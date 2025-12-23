from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class HookSettings:
    enabled: bool = True
    hook_timeout_s: Optional[float] = None
    max_hook_execution_time_s: Optional[float] = None
    enable_cancellation: bool = True
    execution_order: str = "priority"  # "priority", "registration", "random"
    disabled_hooks: set[str] = field(default_factory=set)


@dataclass
class HookContext:
    hook_type: str
    runtime_state: Dict[str, Any]
    config: Any = None
    session_data: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    cancel_reason: Optional[str] = None

    def cancel(self, reason: str = "Hook execution cancelled") -> None:
        self.cancelled = True
        self.cancel_reason = reason

    def with_hook_type(self, hook_type: str) -> "HookContext":
        return replace(self, hook_type=hook_type)

    def with_runtime_state(self, **updates: Any) -> "HookContext":
        updated = dict(self.runtime_state)
        updated.update(updates)
        return replace(self, runtime_state=updated)


@dataclass
class HookResult:
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    modified_context: Optional[HookContext] = None
    should_cancel: bool = False
    should_skip: bool = False


@dataclass(frozen=True)
class HookSubscription:
    hook_type: str
    hook_func: Callable[[HookContext], Any]
    priority: int
    order: int
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HookExecutionError(RuntimeError):
    def __init__(self, hook_type: str, hook_func: str, original_error: Exception):
        super().__init__(f"hook={hook_type} func={hook_func} error={original_error}")
        self.hook_type = hook_type
        self.hook_func = hook_func
        self.original_error = original_error


class HookTimeoutError(HookExecutionError):
    pass


class HookRegistry:
    """Central registry for managing hook subscriptions and execution."""

    def __init__(self, settings: Optional[HookSettings] = None):
        self._hooks: Dict[str, List[HookSubscription]] = {}
        self._settings = settings or HookSettings()
        self._order_counter = 0

    def settings(self) -> HookSettings:
        return self._settings

    def update_settings(self, **updates: Any) -> None:
        for key, value in updates.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)

    def register(
        self,
        hook_type: str,
        hook_func: Callable[[HookContext], Any],
        *,
        priority: int = 0,
        name: Optional[str] = None,
        **metadata: Any,
    ) -> HookSubscription:
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._order_counter += 1
        raw_name = name if name is not None else getattr(hook_func, "__name__", "hook")
        name_str = str(raw_name) if raw_name is not None else "hook"
        subscription = HookSubscription(
            hook_type=hook_type,
            hook_func=hook_func,
            priority=int(priority),
            order=self._order_counter,
            name=name_str,
            metadata=dict(metadata),
        )
        self._hooks[hook_type].append(subscription)
        return subscription

    def unregister(self, hook_type: str, hook_func: Callable[[HookContext], Any]) -> None:
        if hook_type not in self._hooks:
            return
        self._hooks[hook_type] = [
            sub for sub in self._hooks[hook_type] if sub.hook_func is not hook_func
        ]

    def get_hook_types(self) -> List[str]:
        return sorted(self._hooks.keys())

    def execute_hooks(self, hook_type: str, context: HookContext) -> HookResult:
        settings = self._settings
        if not settings.enabled or hook_type in settings.disabled_hooks:
            return HookResult(success=True, modified_context=context)

        subscriptions = list(self._hooks.get(hook_type, ()))
        if not subscriptions:
            return HookResult(success=True, modified_context=context)

        order = settings.execution_order
        if order == "priority":
            subscriptions.sort(key=lambda sub: (-sub.priority, sub.order))
        elif order == "registration":
            subscriptions.sort(key=lambda sub: sub.order)
        elif order == "random":
            random.shuffle(subscriptions)

        errors: List[str] = []
        combined_data: Dict[str, Any] = {}
        current_context = context
        should_cancel = False
        should_skip = False
        total_elapsed = 0.0

        for sub in subscriptions:
            start = time.monotonic()
            try:
                result = sub.hook_func(current_context)
            except Exception as exc:
                errors.append(f"{sub.name}: {exc}")
                continue
            elapsed = time.monotonic() - start
            total_elapsed += elapsed
            if settings.hook_timeout_s and elapsed > settings.hook_timeout_s:
                errors.append(f"{sub.name}: timeout exceeded ({elapsed:.2f}s)")
                if settings.enable_cancellation:
                    should_cancel = True

            current_context, combined_data, errors, should_cancel, should_skip = _merge_hook_result(
                current_context,
                combined_data,
                errors,
                should_cancel,
                should_skip,
                result,
            )
            if current_context.cancelled and settings.enable_cancellation:
                should_cancel = True
            if should_cancel:
                break
            if settings.max_hook_execution_time_s and total_elapsed > settings.max_hook_execution_time_s:
                errors.append("hook execution time exceeded")
                if settings.enable_cancellation:
                    should_cancel = True
                break

        success = not errors
        return HookResult(
            success=success,
            data=combined_data,
            errors=errors,
            modified_context=current_context,
            should_cancel=should_cancel,
            should_skip=should_skip,
        )


def _merge_hook_result(
    current_context: HookContext,
    combined_data: Dict[str, Any],
    errors: List[str],
    should_cancel: bool,
    should_skip: bool,
    result: Any,
) -> tuple[HookContext, Dict[str, Any], List[str], bool, bool]:
    if isinstance(result, HookResult):
        combined_data.update(result.data)
        errors.extend(result.errors)
        if result.modified_context is not None:
            current_context = result.modified_context
        should_cancel = should_cancel or result.should_cancel
        should_skip = should_skip or result.should_skip
    elif isinstance(result, HookContext):
        current_context = result
    elif isinstance(result, dict):
        combined_data.update(result)
    return current_context, combined_data, errors, should_cancel, should_skip


_GLOBAL_HOOK_REGISTRY: Optional[HookRegistry] = None


def get_global_hook_registry() -> HookRegistry:
    global _GLOBAL_HOOK_REGISTRY
    if _GLOBAL_HOOK_REGISTRY is None:
        _GLOBAL_HOOK_REGISTRY = HookRegistry()
    return _GLOBAL_HOOK_REGISTRY


def reset_global_hook_registry() -> None:
    global _GLOBAL_HOOK_REGISTRY
    _GLOBAL_HOOK_REGISTRY = None


class ToolHookTypes:
    TOOL_DISCOVERY = "tool_discovery"
    TOOL_REGISTRATION = "tool_registration"
    TOOL_VALIDATION_SCHEMA = "tool_validation_schema"
    PRE_TOOL_DISCOVERY = "pre_tool_discovery"
    PRE_TOOL_VALIDATION = "pre_tool_validation"
    PRE_PERMISSION_CHECK = "pre_permission_check"
    PRE_PATH_SAFETY_CHECK = "pre_path_safety_check"
    PRE_ARGUMENT_PARSING = "pre_argument_parsing"
    PRE_TOOL_EXECUTION = "pre_tool_execution"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_PROGRESS = "tool_execution_progress"
    TOOL_EXECUTION_TIMEOUT = "tool_execution_timeout"
    POST_TOOL_EXECUTION = "post_tool_execution"
    POST_RESULT_VALIDATION = "post_result_validation"
    POST_RESULT_FORMATTING = "post_result_formatting"
    POST_RESULT_TRANSFORM = "post_result_transform"
    ON_TOOL_ERROR = "on_tool_error"
    ON_VALIDATION_ERROR = "on_validation_error"
    ON_PERMISSION_ERROR = "on_permission_error"
    ON_PATH_SAFETY_ERROR = "on_path_safety_error"
    ON_EXECUTION_TIMEOUT = "on_execution_timeout"
    ON_TOOL_METRICS = "on_tool_metrics"
    ON_TOOL_PERFORMANCE = "on_tool_performance"


class ErrorHookTypes:
    ON_ERROR_OCCURRED = "on_error_occurred"
    ON_ERROR_RECOVERY_START = "on_error_recovery_start"
    ON_ERROR_RECOVERY_SUCCESS = "on_error_recovery_success"
    ON_ERROR_RECOVERY_FAILED = "on_error_recovery_failed"
    ON_TOOL_ERROR = "on_tool_error"
    ON_TOOL_TIMEOUT = "on_tool_timeout"
    ON_TOOL_VALIDATION_ERROR = "on_tool_validation_error"
    ON_TOOL_PERMISSION_ERROR = "on_tool_permission_error"
    ON_TOOL_EXECUTION_ERROR = "on_tool_execution_error"
    ON_PROTOCOL_PARSE_ERROR = "on_protocol_parse_error"
    ON_PROTOCOL_VALIDATION_ERROR = "on_protocol_validation_error"
    ON_PROTOCOL_VERSION_ERROR = "on_protocol_version_error"
    ON_LLM_API_ERROR = "on_llm_api_error"
    ON_LLM_TIMEOUT_ERROR = "on_llm_timeout_error"
    ON_LLM_RATE_LIMIT_ERROR = "on_llm_rate_limit_error"
    ON_LLM_CONTEXT_LENGTH_ERROR = "on_llm_context_length_error"
    ON_LLM_EMPTY_REPLY_ERROR = "on_llm_empty_reply_error"
    ON_MEMORY_COMPACT_ERROR = "on_memory_compact_error"
    ON_MEMORY_ESTIMATION_ERROR = "on_memory_estimation_error"
    ON_MEMORY_TRUNCATION_ERROR = "on_memory_truncation_error"
    ON_PLUGIN_LOAD_ERROR = "on_plugin_load_error"
    ON_PLUGIN_VALIDATION_ERROR = "on_plugin_validation_error"
    ON_PLUGIN_EXECUTION_ERROR = "on_plugin_execution_error"
    ON_FILE_READ_ERROR = "on_file_read_error"
    ON_FILE_WRITE_ERROR = "on_file_write_error"
    ON_FILE_PERMISSION_ERROR = "on_file_permission_error"
    ON_FILE_NOT_FOUND_ERROR = "on_file_not_found_error"
    ON_RETRY_DECISION = "on_retry_decision"
    ON_RETRY_ATTEMPT = "on_retry_attempt"
    ON_RETRY_SUCCESS = "on_retry_success"
    ON_RETRY_EXHAUSTED = "on_retry_exhausted"
    ON_ERROR_TRANSFORM = "on_error_transform"
    ON_ERROR_LOG = "on_error_log"
    ON_ERROR_ALERT = "on_error_alert"
    ON_ERROR_REPORT = "on_error_report"


class ProtocolHookTypes:
    PRE_MESSAGE_PARSING = "pre_message_parsing"
    POST_MESSAGE_PARSING = "post_message_parsing"
    PRE_MESSAGE_VALIDATION = "pre_message_validation"
    POST_MESSAGE_VALIDATION = "post_message_validation"
    PRE_MESSAGE_TRANSFORMATION = "pre_message_transformation"
    POST_MESSAGE_TRANSFORMATION = "post_message_transformation"
    PRE_STEP_PROCESSING = "pre_step_processing"
    POST_STEP_PROCESSING = "post_step_processing"
    PRE_STEP_VALIDATION = "pre_step_validation"
    POST_STEP_VALIDATION = "post_step_validation"
    PRE_TOOL_CALL_PARSING = "pre_tool_call_parsing"
    POST_TOOL_CALL_PARSING = "post_tool_call_parsing"
    PRE_TOOL_CALL_VALIDATION = "pre_tool_call_validation"
    POST_TOOL_CALL_VALIDATION = "post_tool_call_validation"
    PRE_TOOL_CALL_TRANSFORMATION = "pre_tool_call_transformation"
    POST_TOOL_CALL_TRANSFORMATION = "post_tool_call_transformation"
    PROTOCOL_VERSION_DETECTION = "protocol_version_detection"
    PROTOCOL_VERSION_MIGRATION = "protocol_version_migration"
    PROTOCOL_COMPATIBILITY_CHECK = "protocol_compatibility_check"
    PRE_ASSISTANT_TURN_PARSING = "pre_assistant_turn_parsing"
    POST_ASSISTANT_TURN_PARSING = "post_assistant_turn_parsing"
    PRE_ASSISTANT_TURN_VALIDATION = "pre_assistant_turn_validation"
    POST_ASSISTANT_TURN_VALIDATION = "post_assistant_turn_validation"
    PRE_CONTENT_PROCESSING = "pre_content_processing"
    POST_CONTENT_PROCESSING = "post_content_processing"
    PRE_CONTENT_SANITIZATION = "pre_content_sanitization"
    POST_CONTENT_SANITIZATION = "post_content_sanitization"
    PRE_REQUEST_SENDING = "pre_request_sending"
    POST_REQUEST_SENDING = "post_request_sending"
    PRE_RESPONSE_RECEIVING = "pre_response_receiving"
    POST_RESPONSE_RECEIVING = "post_response_receiving"
    ON_PROTOCOL_ERROR = "on_protocol_error"
    ON_PARSE_ERROR = "on_parse_error"
    ON_VALIDATION_ERROR = "on_validation_error"
    ON_VERSION_ERROR = "on_version_error"
    ON_PROTOCOL_METRICS = "on_protocol_metrics"
    ON_MESSAGE_PROCESSING_METRICS = "on_message_processing_metrics"


class MemoryHookTypes:
    MEMORY_INITIALIZATION = "memory_initialization"
    MEMORY_COMPACT_START = "memory_compact_start"
    MEMORY_COMPACT_END = "memory_compact_end"
    MEMORY_RESET = "memory_reset"
    MEMORY_CLEANUP = "memory_cleanup"
    PRE_MESSAGE_FILTERING = "pre_message_filtering"
    POST_MESSAGE_FILTERING = "post_message_filtering"
    PRE_MESSAGE_RANKING = "pre_message_ranking"
    POST_MESSAGE_RANKING = "post_message_ranking"
    PRE_MESSAGE_TRUNCATION = "pre_message_truncation"
    POST_MESSAGE_TRUNCATION = "post_message_truncation"
    PRE_TOKEN_ESTIMATION = "pre_token_estimation"
    POST_TOKEN_ESTIMATION = "post_token_estimation"
    TOKEN_ESTIMATION_STRATEGY = "token_estimation_strategy"
    TOKEN_COUNT_ADJUSTMENT = "token_count_adjustment"
    COMPACTION_STRATEGY_SELECTION = "compaction_strategy_selection"
    COMPACTION_RULE_APPLICATION = "compaction_rule_application"
    COMPACTION_PRIORITY_CALCULATION = "compaction_priority_calculation"
    COMPACTION_EXECUTION = "compaction_execution"
    PRE_PINNING_SELECTION = "pre_pinning_selection"
    POST_PINNING_SELECTION = "post_pinning_selection"
    CONTEXT_PRESERVE_RULES = "context_preserve_rules"
    CONTEXT_IMPORTANCE_SCORING = "context_importance_scoring"
    PRE_OPTIMIZATION = "pre_optimization"
    POST_OPTIMIZATION = "post_optimization"
    OPTIMIZATION_STRATEGY = "optimization_strategy"
    MEMORY_EFFICIENCY_CHECK = "memory_efficiency_check"
    ON_MEMORY_METRICS = "on_memory_metrics"
    ON_COMPACT_PERFORMANCE = "on_compact_performance"
    ON_MEMORY_USAGE_ALERT = "on_memory_usage_alert"
    ON_MEMORY_PRESSURE = "on_memory_pressure"
    ON_MEMORY_ERROR = "on_memory_error"
    ON_COMPACTION_ERROR = "on_compaction_error"
    ON_ESTIMATION_ERROR = "on_estimation_error"


class LoggingHookTypes:
    ON_LOG_MESSAGE = "on_log_message"
    PRE_LOG_WRITE = "pre_log_write"
    POST_LOG_WRITE = "post_log_write"
    LOG_FORMATTING = "log_formatting"
    LOG_FILTERING = "log_filtering"
    ON_SYSTEM_STARTUP = "on_system_startup"
    ON_SYSTEM_SHUTDOWN = "on_system_shutdown"
    ON_CONFIG_CHANGE = "on_config_change"
    ON_MODE_CHANGE = "on_mode_change"
    ON_SESSION_START = "on_session_start"
    ON_SESSION_END = "on_session_end"
    ON_SESSION_PAUSE = "on_session_pause"
    ON_SESSION_RESUME = "on_session_resume"
    ON_SESSION_IDLE = "on_session_idle"
    ON_PERFORMANCE_METRICS = "on_performance_metrics"
    ON_EXECUTION_TIME = "on_execution_time"
    ON_MEMORY_USAGE = "on_memory_usage"
    ON_CPU_USAGE = "on_cpu_usage"
    ON_DISK_IO = "on_disk_io"
    ON_TOOL_START = "on_tool_start"
    ON_TOOL_SUCCESS = "on_tool_success"
    ON_TOOL_ERROR = "on_tool_error"
    ON_TOOL_TIMEOUT = "on_tool_timeout"
    ON_TOOL_METRICS = "on_tool_metrics"
    ON_AGENT_THINK = "on_agent_think"
    ON_AGENT_ACTION = "on_agent_action"
    ON_AGENT_ERROR = "on_agent_error"
    ON_AGENT_METRICS = "on_agent_metrics"
    ON_LLM_REQUEST = "on_llm_request"
    ON_LLM_RESPONSE = "on_llm_response"
    ON_LLM_ERROR = "on_llm_error"
    ON_LLM_METRICS = "on_llm_metrics"
    ON_MEMORY_COMPACT = "on_memory_compact"
    ON_MEMORY_ESTIMATE = "on_memory_estimate"
    ON_MEMORY_OPTIMIZE = "on_memory_optimize"
    ON_MEMORY_PRESSURE = "on_memory_pressure"
    ON_PLUGIN_LOAD = "on_plugin_load"
    ON_PLUGIN_UNLOAD = "on_plugin_unload"
    ON_PLUGIN_ERROR = "on_plugin_error"
    ON_PLUGIN_METRICS = "on_plugin_metrics"
    ON_ERROR_OCCURRED = "on_error_occurred"
    ON_EXCEPTION_CAUGHT = "on_exception_caught"
    ON_ERROR_RECOVERY = "on_error_recovery"
    ON_ERROR_ESCALATION = "on_error_escalation"
    ON_ALERT_TRIGGER = "on_alert_trigger"
    ON_NOTIFICATION_SEND = "on_notification_send"
    ON_REPORT_GENERATE = "on_report_generate"
    ON_HEALTH_CHECK = "on_health_check"
    ON_METRICS_COLLECT = "on_metrics_collect"
    ON_METRICS_AGGREGATE = "on_metrics_aggregate"
    ON_METRICS_EXPORT = "on_metrics_export"
    ON_METRICS_RESET = "on_metrics_reset"


class AgentHookTypes:
    AGENT_INITIALIZATION = "agent_initialization"
    AGENT_STARTUP = "agent_startup"
    AGENT_SHUTDOWN = "agent_shutdown"
    AGENT_PAUSE = "agent_pause"
    AGENT_RESUME = "agent_resume"
    AGENT_RESET = "agent_reset"
    PRE_AGENT_THINK = "pre_agent_think"
    POST_AGENT_THINK = "post_agent_think"
    THOUGHT_PROCESSING = "thought_processing"
    THOUGHT_VALIDATION = "thought_validation"
    THOUGHT_TRANSFORMATION = "thought_transformation"
    PRE_AGENT_DECISION = "pre_agent_decision"
    POST_AGENT_DECISION = "post_agent_decision"
    DECISION_VALIDATION = "decision_validation"
    DECISION_EXECUTION = "decision_execution"
    DECISION_LEARNING = "decision_learning"
    PRE_AGENT_ACTION = "pre_agent_action"
    POST_AGENT_ACTION = "post_agent_action"
    ACTION_VALIDATION = "action_validation"
    ACTION_EXECUTION = "action_execution"
    ACTION_EVALUATION = "action_evaluation"
    LEARNING_EVENT = "learning_event"
    EXPERIENCE_REPLAY = "experience_replay"
    KNOWLEDGE_UPDATE = "knowledge_update"
    STRATEGY_ADAPTATION = "strategy_adaptation"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    STATE_CHANGE = "state_change"
    STATE_VALIDATION = "state_validation"
    STATE_TRANSITION = "state_transition"
    STATE_PERSISTENCE = "state_persistence"
    STATE_RECOVERY = "state_recovery"
    PRE_COMMUNICATION = "pre_communication"
    POST_COMMUNICATION = "post_communication"
    MESSAGE_PROCESSING = "message_processing"
    RESPONSE_GENERATION = "response_generation"
    CONTEXT_SHARING = "context_sharing"
    PERSONALITY_APPLICATION = "personality_application"
    BEHAVIOR_MODIFICATION = "behavior_modification"
    EMOTIONAL_RESPONSE = "emotional_response"
    STYLE_ADAPTATION = "style_adaptation"
    STRATEGY_SELECTION = "strategy_selection"
    STRATEGY_EXECUTION = "strategy_execution"
    STRATEGY_EVALUATION = "strategy_evaluation"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    COLLABORATION_START = "collaboration_start"
    COLLABORATION_END = "collaboration_end"
    TASK_DELEGATION = "task_delegation"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    COORDINATION = "coordination"
    ON_AGENT_METRICS = "on_agent_metrics"
    ON_BEHAVIOR_ANALYSIS = "on_behavior_analysis"
    ON_PERFORMANCE_REVIEW = "on_performance_review"
    ON_HEALTH_CHECK = "on_health_check"


@dataclass
class ToolHookContext(HookContext):
    tool_name: str = ""
    tool_target: Optional[str] = None
    tool_args: Any = None
    tool_call: Any = None
    tool_result: Optional[str] = None
    plugin_info: Optional[Dict[str, Any]] = None
    runtime_tool: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class ErrorHookContext(HookContext):
    error_type: str = ""
    error_message: str = ""
    error_exception: Optional[Exception] = None
    error_severity: str = "error"
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: Optional[str] = None
    original_context: Optional[HookContext] = None


@dataclass
class ProtocolHookContext(HookContext):
    protocol_version: str = ""
    message_type: str = ""
    step_type: str = ""
    raw_message: Optional[str] = None
    parsed_message: Any = None
    validation_result: Optional[bool] = None
    validation_errors: List[str] = field(default_factory=list)
    transformation_result: Optional[Dict[str, Any]] = None


@dataclass
class MemoryHookContext(HookContext):
    messages: List[Dict[str, Any]] = field(default_factory=list)
    original_messages: List[Dict[str, Any]] = field(default_factory=list)
    memory_budget: Optional[int] = None
    current_token_count: Optional[int] = None
    target_token_count: Optional[int] = None
    compaction_ratio: Optional[float] = None


@dataclass
class LoggingHookContext(HookContext):
    log_level: str = "info"
    log_message: str = ""
    log_data: Dict[str, Any] = field(default_factory=dict)
    log_source: str = ""
    formatted_message: Optional[str] = None


@dataclass
class AgentHookContext(HookContext):
    agent_id: str = ""
    agent_type: str = ""
    agent_state: Dict[str, Any] = field(default_factory=dict)
    current_task: Optional[str] = None
