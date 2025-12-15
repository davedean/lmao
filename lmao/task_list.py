from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Task:
    id: int
    text: str
    done: bool = False


class TaskListManager:
    def __init__(self) -> None:
        self.tasks: List[Task] = []
        self._next_id: int = 1
        self._active: bool = False
        # Always start with an active list.
        self.new_list()

    def has_list(self) -> bool:
        return self._active

    def new_list(self) -> None:
        self.tasks = []
        self._next_id = 1
        self._active = True

    def add_task(self, text: str) -> Task:
        if not self.has_list():
            self.new_list()
        task = Task(id=self._next_id, text=text, done=False)
        self.tasks.append(task)
        self._next_id += 1
        return task

    def _find_task(self, task_id: int) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def complete_task(self, task_id: int) -> str:
        if not self.has_list():
            return "error: no active task list"
        task = self._find_task(task_id)
        if not task:
            return f"error: task {task_id} not found"
        task.done = True
        return f"completed task {task.id}: {task.text}"

    def delete_task(self, task_id: int) -> str:
        if not self.has_list():
            return "error: no active task list"
        task = self._find_task(task_id)
        if not task:
            return f"error: task {task_id} not found"
        self.tasks = [t for t in self.tasks if t.id != task_id]
        return f"deleted task {task_id}"

    def to_payload(self) -> dict:
        """Structured representation of the current task list."""
        return {
            "active": self._active,
            "tasks": [
                {"id": task.id, "text": task.text, "done": task.done}
                for task in self.tasks
            ],
        }

    def render_tasks(self) -> str:
        if not self.has_list():
            return "no active task list"
        lines: List[str] = []
        for task in self.tasks:
            box = "[x]" if task.done else "[ ]"
            lines.append(f"{box} {task.id} {task.text}")
        return "\n".join(lines) if lines else "(empty task list)"
