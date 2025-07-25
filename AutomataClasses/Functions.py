from typing import Any


class Function:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def apply(self, token: Any) -> Any:
        if token in self.mapping:
            return self.mapping[token]
        raise ValueError(f"Function cannot parse token: {token}. Accepted tokens are: {list(self.mapping.keys())}.")


class Identity(Function):
    def __init__(self):
        super().__init__(self)

    def apply(self, token: Any) -> Any:
        return token
