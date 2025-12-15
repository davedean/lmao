from unittest import TestCase

from lmao.governance import (
    can_end_conversation,
    should_render_user_messages,
)
from lmao.protocol import MessageStep
from lmao.task_list import TaskListManager


class GovernanceTests(TestCase):
    def test_withholds_progress_messages_until_tasks_complete(self) -> None:
        manager = TaskListManager()
        manager.new_list()
        manager.add_task("do something")
        msg = MessageStep(type="message", content="progress", purpose="progress")
        self.assertFalse(should_render_user_messages(manager, [msg]))

    def test_allows_clarification_with_incomplete_tasks(self) -> None:
        manager = TaskListManager()
        manager.new_list()
        manager.add_task("do something")
        msg = MessageStep(type="message", content="question", purpose="clarification")
        self.assertTrue(should_render_user_messages(manager, [msg]))

    def test_allows_cannot_finish_end_with_incomplete_tasks(self) -> None:
        manager = TaskListManager()
        manager.new_list()
        manager.add_task("do something")
        msg = MessageStep(type="message", content="can't do it", purpose="cannot_finish")
        self.assertTrue(can_end_conversation(manager, [msg]))
