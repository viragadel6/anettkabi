from __future__ import annotations

__all__ = [
    "DomainError",
    "HandlerNotRegisteredError",
    "InvalidStateTransitionError",
    "TaskNotFoundError",
    "TaskOwnershipError",
]


class DomainError(Exception):
    pass


class InvalidStateTransitionError(DomainError):
    def __init__(self, current_state: object, target_state: object) -> None:
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(
            f"Invalid task state transition from {current_state!s} to {target_state!s}"
        )


class TaskNotFoundError(DomainError):
    def __init__(self, task_id: object) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id} was not found")


class TaskOwnershipError(DomainError):
    def __init__(self, task_id: object, expected_holder: object, actual_holder: object) -> None:
        self.task_id = task_id
        self.expected_holder = expected_holder
        self.actual_holder = actual_holder
        super().__init__(
            f"Task {task_id} is held by {actual_holder!r}, not by {expected_holder!r}"
        )


class HandlerNotRegisteredError(DomainError):
    def __init__(self, task_type: str) -> None:
        self.task_type = task_type
        super().__init__(f"No task handler is registered for task type {task_type!r}")
