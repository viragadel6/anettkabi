from __future__ import annotations

from enum import Enum

from domain.exceptions import InvalidStateTransitionError

__all__ = [
    "TaskState",
    "TaskStateMachine",
    "TERMINAL_STATES",
    "can_transition",
    "validate_transition",
]


class TaskState(str, Enum):
    PENDING = "PENDING"
    ACQUIRED = "ACQUIRED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.DEAD_LETTER}
)


class TaskStateMachine:
    TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
        TaskState.PENDING: frozenset({TaskState.ACQUIRED}),
        TaskState.ACQUIRED: frozenset({TaskState.RUNNING, TaskState.PENDING}),
        TaskState.RUNNING: frozenset(
            {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.DEAD_LETTER,
                TaskState.PENDING,
            }
        ),
        TaskState.FAILED: frozenset({TaskState.PENDING}),
        TaskState.COMPLETED: frozenset(),
        TaskState.DEAD_LETTER: frozenset({TaskState.PENDING}),
    }

    @classmethod
    def can_transition(cls, current_state: TaskState, target_state: TaskState) -> bool:
        return target_state in cls.TRANSITIONS[current_state]

    @classmethod
    def validate_transition(
        cls, current_state: TaskState, target_state: TaskState
    ) -> None:
        if not cls.can_transition(current_state, target_state):
            raise InvalidStateTransitionError(current_state, target_state)

    @classmethod
    def next_states(cls, current_state: TaskState) -> frozenset[TaskState]:
        return cls.TRANSITIONS[current_state]

    @classmethod
    def is_terminal(cls, state: TaskState) -> bool:
        return state in TERMINAL_STATES


def can_transition(current_state: TaskState, target_state: TaskState) -> bool:
    return TaskStateMachine.can_transition(current_state, target_state)


def validate_transition(current_state: TaskState, target_state: TaskState) -> None:
    TaskStateMachine.validate_transition(current_state, target_state)
