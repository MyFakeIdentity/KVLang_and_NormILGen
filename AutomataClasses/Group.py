from typing import Any, Callable
from AutomataClasses.SemiAutomaton import SemiAutomaton


class Group(SemiAutomaton):
    def __init__(self, initial_state, state_set: set, function: Callable):
        super().__init__(state_set)
        self.state = initial_state
        self.function = function

    def get_state(self) -> Any:
        return self.state

    def update(self, token: Any) -> None:
        super().update(token)
        self.state = self.function(self.state, token)
