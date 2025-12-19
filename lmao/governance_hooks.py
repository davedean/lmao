"""Governance hooks for loop simplification.

This module contains default hook implementations that move governance
logic from inline code in loop.py into configurable hooks.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .hooks import (
    AgentHookContext,
    AgentHookTypes,
    HookContext,
    HookResult,
    ProtocolHookContext,
    ProtocolHookTypes,
    ToolHookContext,
    ToolHookTypes,
    ErrorHookTypes,
    ErrorHookContext,
    LoggingHookTypes,
    LoggingHookContext,
)
from .protocol import ProtocolError
from .runtime_tools import RuntimeContext


# Constants for governance behavior
MESSAGE_PURPOSE_CLARIFICATION = "clarification"
MESSAGE_PURPOSE_CANNOT_FINISH = "cannot_finish"

_MAX_EMPTY_REPLIES = 4
_MAX_INVALID_REPLIES = 2
_MAX_THINK_ONLY_TURNS = 3
_MAX_PROGRESS_ONLY_TURNS = 4

_HEADLESS_INPUT_REQUEST_PATTERNS = (
    "would you like",
    "do you want",
    "can you",
    "could you",
    "please provide",
    "please share",
    "please confirm",
    "please clarify",
    "which one",
    "which should",
    "what should i",
    "should i",
    "let me know",
    "tell me",
    "need more info",
    "need more information",
    "what is your",
    "what are your",
)


class GovernanceHookManager:
    """Manages governance hook implementations for loop simplification."""

    def __init__(self, runtime_context: Optional[RuntimeContext] = None):
        self.runtime_context = runtime_context
        self.empty_replies = 0
        self.invalid_replies = 0
        self.think_only_turns = 0
        self.progress_only_turns = 0

    def register_all_hooks(self, hook_registry) -> None:
        """Register all governance hooks with the given registry."""
        # Protocol recovery hooks
        hook_registry.register(
            ErrorHookTypes.ON_PROTOCOL_PARSE_ERROR,
            self._handle_protocol_parse_error,
            priority=10,
            name="protocol_recovery",
        )

        hook_registry.register(
            ErrorHookTypes.ON_LLM_EMPTY_REPLY_ERROR,
            self._handle_empty_reply,
            priority=10,
            name="empty_reply_recovery",
        )

        hook_registry.register(
            ProtocolHookTypes.POST_MESSAGE_PARSING,
            self._handle_post_message_parsing,
            priority=10,
            name="post_parsing_governance",
        )

        # Headless guardrail hooks
        hook_registry.register(
            ProtocolHookTypes.POST_MESSAGE_VALIDATION,
            self._handle_headless_validation,
            priority=10,
            name="headless_guardrail",
        )

        # Tool result formatting hooks
        hook_registry.register(
            ToolHookTypes.POST_RESULT_FORMATTING,
            self._handle_tool_result_formatting,
            priority=10,
            name="tool_result_formatting",
        )

        # Startup prelude hooks
        hook_registry.register(
            AgentHookTypes.AGENT_STARTUP,
            self._handle_agent_startup,
            priority=10,
            name="startup_prelude",
        )

    def _handle_protocol_parse_error(self, context: ErrorHookContext) -> HookResult:
        """Handle protocol parsing errors with recovery logic."""
        self.invalid_replies += 1

        if self.invalid_replies > _MAX_INVALID_REPLIES:
            # Exhausted retries - fail gracefully
            error_message = (
                "error: model repeatedly returned invalid JSON protocol output.\n"
                f"last error: {context.error_message}\n"
                "last reply (verbatim):\n"
                f"{context.runtime_state.get('assistant_reply', 'N/A')}"
            )

            # Add error message to conversation
            if messages := context.runtime_state.get("messages"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": context.runtime_state.get("assistant_reply", ""),
                    }
                )

            return HookResult(
                success=True,
                data={
                    "action": "fail_conversation",
                    "message": error_message,
                    "should_end": True,
                },
                should_cancel=True,
            )

        # Provide retry instruction
        recovery_message = (
            f"Your reply was not valid for the required JSON assistant protocol.\n"
            f"Error: {context.error_message}\n"
            "Return ONLY a single JSON object matching:\n"
            '{"type":"assistant_turn","version":"2","steps":[...]}\n'
            "No code fences, no extra text. Retry now."
        )

        return HookResult(
            success=True,
            data={
                "action": "insert_user_message",
                "message": recovery_message,
            },
        )

    def _handle_empty_reply(self, context: ErrorHookContext) -> HookResult:
        """Handle empty LLM replies with recovery logic."""
        self.empty_replies += 1

        if self.empty_replies >= _MAX_EMPTY_REPLIES:
            # Exhausted retries - use fallback
            last_tool_summary = context.runtime_state.get(
                "last_tool_summary", "No tool output available."
            )
            fallback_reply = (
                f"(auto-generated fallback) Unable to get a response from the model. "
                f"Based on the latest tool output, here is a summary:\n{last_tool_summary}"
            )

            return HookResult(
                success=True,
                data={
                    "action": "insert_assistant_message_and_end",
                    "message": fallback_reply,
                },
                should_cancel=True,
            )

        # Provide retry instruction
        recovery_message = (
            "Your last reply was empty (or whitespace). This is not allowed.\n"
            f"The user asked: {context.runtime_state.get('last_user', 'unknown')!r}.\n"
            "Return ONLY a single JSON object matching the assistant protocol.\n"
            "Do NOT return whitespace.\n\n"
            "If you are unsure what to do next, emit a tool_call or a short progress message."
        )

        return HookResult(
            success=True,
            data={
                "action": "insert_user_message",
                "message": recovery_message,
            },
        )

    def _handle_post_message_parsing(self, context: ProtocolHookContext) -> HookResult:
        """Handle post-parsing governance for think-only and progress-only turns."""
        if not context.parsed_message:
            return HookResult(success=True)

        steps = context.parsed_message.get("steps", [])
        messages = context.runtime_state.get("messages", [])

        # Parse step types
        thinks = [step for step in steps if step.get("type") == "think"]
        user_messages = [step for step in steps if step.get("type") == "message"]
        tool_call_payloads = [step for step in steps if step.get("type") == "tool_call"]
        has_end = any(step.get("type") == "end" for step in steps)

        input_requested = any(
            msg.get("purpose") == MESSAGE_PURPOSE_CLARIFICATION for msg in user_messages
        )

        # Handle think-only turns
        if thinks and not user_messages and not tool_call_payloads and not has_end:
            self.think_only_turns += 1

            if self.think_only_turns > _MAX_THINK_ONLY_TURNS:
                reminder = (
                    "You have emitted multiple think-only turns. This is not allowed.\n"
                    "Next, emit either tool_call steps or message/end steps.\n\n"
                    "Return ONLY a single JSON object matching the assistant protocol.\n"
                )
            else:
                reminder = "You produced only think steps. Continue immediately with the next action.\n\n"

            example_json = (
                '{"type":"assistant_turn","version":"2","steps":['
                '{"type":"tool_call","call":{"tool":"read","target":"README.md","args":"lines:1-40"}}'
                "]}"
            )

            recovery_message = (
                f"{reminder}"
                "Return ONLY a single JSON object matching the assistant protocol.\n"
                f"Example next action:\n{example_json}"
            )

            return HookResult(
                success=True,
                data={
                    "action": "insert_user_message",
                    "message": recovery_message,
                },
            )

        # Handle progress-only turns
        if (
            user_messages
            and not tool_call_payloads
            and not has_end
            and not input_requested
        ):
            self.progress_only_turns += 1

            purposes = {msg.get("purpose") for msg in user_messages}
            needs_explicit_end = bool(
                purposes.intersection({"final", MESSAGE_PURPOSE_CANNOT_FINISH})
            )

            if self.progress_only_turns >= _MAX_PROGRESS_ONLY_TURNS:
                reminder = (
                    "You have emitted multiple message-only turns without taking an action. "
                    "This is not allowed.\n"
                )
            else:
                reminder = ""

            if needs_explicit_end:
                recovery_message = (
                    f"{reminder}"
                    "You sent a terminal message (purpose='final' or 'cannot_finish') but did not include an end step.\n"
                    "Continue immediately by returning a new assistant_turn that includes an explicit end step.\n"
                    "Return ONLY JSON; do not ask the user to type 'ok' or provide follow-ups."
                )
            else:
                recovery_message = (
                    f"{reminder}"
                    "You sent a progress message but did not call any tools and did not end.\n"
                    "Continue immediately without waiting for user input: either call a tool (tool_call steps) or, if finished, send purpose='final' AND an explicit end step.\n"
                    "Do not ask the user to type 'ok' or otherwise prompt for input unless you truly need clarification (then set purpose='clarification')."
                )

            return HookResult(
                success=True,
                data={
                    "action": "insert_user_message",
                    "message": recovery_message,
                },
            )

        # Reset counters on successful turn
        self.think_only_turns = 0
        self.progress_only_turns = 0

        return HookResult(success=True)

    def _handle_headless_validation(self, context: ProtocolHookContext) -> HookResult:
        """Handle headless mode guardrails for input requests."""
        if not context.runtime_state.get("headless_run", False):
            return HookResult(success=True)

        if not context.parsed_message:
            return HookResult(success=True)

        steps = context.parsed_message.get("steps", [])
        user_messages = [step for step in steps if step.get("type") == "message"]

        explicit_clarification_requested = any(
            msg.get("purpose") == MESSAGE_PURPOSE_CLARIFICATION for msg in user_messages
        )

        implicit_input_requested = self._requests_user_input(
            [
                msg.get("content", "")
                for msg in user_messages
                if msg.get("purpose") != MESSAGE_PURPOSE_CANNOT_FINISH
            ]
        )

        headless_input_requested = (
            explicit_clarification_requested or implicit_input_requested
        )

        if headless_input_requested:
            recovery_message = (
                "Headless mode is active: the human user cannot respond, so do NOT ask questions or request confirmation.\n"
                "Proceed autonomously: pick reasonable defaults, state assumptions briefly, and continue (call tools if helpful).\n"
                "If you are truly blocked, send a message step with purpose='cannot_finish' describing what's missing, then end."
            )

            return HookResult(
                success=True,
                data={
                    "action": "insert_user_message",
                    "message": recovery_message,
                },
            )

        return HookResult(success=True)

    def _handle_tool_result_formatting(self, context: ToolHookContext) -> HookResult:
        """Handle tool result formatting, truncation, and pinning."""
        if not context.tool_result:
            return HookResult(success=True)

        # This would integrate with existing tool result processing logic
        # For now, pass through unchanged to preserve existing behavior
        return HookResult(success=True)

    def _handle_agent_startup(self, context: AgentHookContext) -> HookResult:
        """Handle agent startup prelude via hooks."""
        if not self.runtime_context:
            return HookResult(success=True)

        # Execute policy tool internally
        policy_result = self.runtime_context.call_tool_internal(
            "policy", args={"truncate": True}
        )

        # Check if skills guide should be included
        initial_prompt = context.runtime_state.get("initial_prompt", "")
        if self._should_include_skills_guide_startup(initial_prompt):
            skills_result = self.runtime_context.call_tool_internal("skills_guide")

        return HookResult(
            success=True,
            data={
                "policy_executed": True,
                "skills_guide_executed": self._should_include_skills_guide_startup(
                    initial_prompt
                ),
            },
        )

    def _requests_user_input(self, messages: List[str]) -> bool:
        """Check if messages contain user input requests."""
        for content in messages:
            text = (content or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if any(pat in lowered for pat in _HEADLESS_INPUT_REQUEST_PATTERNS):
                return True
            if "?" in lowered:
                # Question marks alone are too noisy; require at least one "request" indicator.
                import re

                if re.search(
                    r"\b(you|your|please|which|what|confirm|clarif|provide|choose)\b",
                    lowered,
                ):
                    return True
        return False

    def _should_include_skills_guide_startup(self, prompt: str) -> bool:
        """Check if skills guide should be included at startup."""
        lowered = (prompt or "").lower()
        triggers = (
            "skill",
            "skills",
            "skill.md",
            "creating a skill",
            "create a skill",
            "write a skill",
            "list skills",
        )
        return any(trigger in lowered for trigger in triggers)


# Helper functions for hook integration
def register_governance_hooks(
    hook_registry, runtime_context: Optional[RuntimeContext] = None
) -> GovernanceHookManager:
    """Register all governance hooks and return the manager instance."""
    manager = GovernanceHookManager(runtime_context)
    manager.register_all_hooks(hook_registry)
    return manager
