from unittest import TestCase

from lmao.task_list import TaskListManager


class TaskListManagerTests(TestCase):
    def setUp(self) -> None:
        self.manager = TaskListManager()
        self.manager.new_list()

    def test_adds_and_lists_tasks(self) -> None:
        self.manager.new_list()
        self.manager.add_task("first")
        self.manager.add_task("second")
        rendered = self.manager.render_tasks()
        self.assertIn("[ ] 1 first", rendered)
        self.assertIn("[ ] 2 second", rendered)

    def test_complete_and_delete_tasks(self) -> None:
        task = self.manager.add_task("do it")
        complete_result = self.manager.complete_task(task.id)
        self.assertIn("completed", complete_result)
        rendered = self.manager.render_tasks()
        self.assertIn("[x] 1 do it", rendered)
        delete_result = self.manager.delete_task(task.id)
        self.assertIn("deleted task", delete_result)
        remaining = self.manager.render_tasks()
        self.assertNotIn("create a plan to respond", remaining)

    def test_no_active_list_error(self) -> None:
        manager = TaskListManager()
        self.assertNotEqual("no active task list", manager.render_tasks())
