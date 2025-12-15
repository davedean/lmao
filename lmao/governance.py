from __future__ import annotations

from typing import Sequence

from .protocol import MessageStep
from .task_list import TaskListManager

MESSAGE_PURPOSE_PROGRESS = "progress"
MESSAGE_PURPOSE_CLARIFICATION = "clarification"
MESSAGE_PURPOSE_CANNOT_FINISH = "cannot_finish"
MESSAGE_PURPOSE_FINAL = "final"

ALLOWED_MESSAGE_PURPOSES = {
    MESSAGE_PURPOSE_PROGRESS,
    MESSAGE_PURPOSE_CLARIFICATION,
    MESSAGE_PURPOSE_CANNOT_FINISH,
    MESSAGE_PURPOSE_FINAL,
}


def has_incomplete_tasks(task_manager: TaskListManager) -> bool:
    return any(not task.done for task in task_manager.tasks)

def should_render_user_messages(task_manager: TaskListManager, messages: Sequence[MessageStep]) -> bool:
    """
    Enforce: do not message the user while tasks remain, unless the message is a
    clarification request or an explanation that the request cannot be completed.
    """
    if not messages:
        return True
    if not has_incomplete_tasks(task_manager):
        return True
    purposes = {msg.purpose for msg in messages}
    return bool(purposes.intersection({MESSAGE_PURPOSE_CLARIFICATION, MESSAGE_PURPOSE_CANNOT_FINISH}))


def can_end_conversation(task_manager: TaskListManager, messages: Sequence[MessageStep]) -> bool:
    if not has_incomplete_tasks(task_manager):
        return True
    return any(msg.purpose == MESSAGE_PURPOSE_CANNOT_FINISH for msg in messages)
