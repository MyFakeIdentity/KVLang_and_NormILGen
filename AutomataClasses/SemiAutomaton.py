from typing import Any


class SemiAutomaton:
    def __init__(self, alphabet: set):
        self.alphabet = alphabet

    def get_state(self) -> Any:
        pass

    def update(self, token: Any) -> None:
        if token not in self.alphabet:
            raise ValueError(f"Token {token} not in alphabet.")

    def reset(self):
        pass
