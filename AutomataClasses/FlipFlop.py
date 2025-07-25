from typing import Any
from AutomataClasses.SemiAutomaton import SemiAutomaton


class FlipFlop(SemiAutomaton):
    def __init__(self, initial_state="Low"):
        super().__init__({"Read", "Set", "Reset"})
        self.initial_state = initial_state
        self.state = self.initial_state

    def get_state(self) -> Any:
        return self.state

    def update(self, token: Any) -> None:
        super().update(token)

        match token:
            case "Read":
                pass
            case "Set":
                self.state = "High"
            case "Reset":
                self.state = "Low"
            case _:
                raise ValueError("Unrecognised token.")

    def reset(self):
        self.state = self.initial_state

