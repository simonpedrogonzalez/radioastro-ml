from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Literal

@dataclass(frozen=True)
class TimeGrid:
    solint: str | int = "int"  # can become int (seconds)
    interp: Literal["linear"] = "linear" # idk if there could be another one that makes sense

    def __post_init__(self):
        if self.solint == "int":
            return

        if isinstance(self.solint, int):
            return

        value = self.solint.strip().lower()

        if value.endswith("s"):
            seconds = int(value[:-1])
        elif value.endswith("m"):
            seconds = int(value[:-1]) * 60
        else:
            raise ValueError(
                f"Invalid solint '{self.solint}'. Use 'int', '#s', or '#m'."
            )

        object.__setattr__(self, "solint", seconds)