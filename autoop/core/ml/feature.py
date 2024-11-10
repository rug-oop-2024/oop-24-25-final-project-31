from pydantic import BaseModel
from typing import Literal


class Feature(BaseModel):
    def __init__(self, name: str, type: str) -> None:
        self.type = type
        self.name = name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value: Literal["numerical", "categorical"]) -> None:
        if value != Literal["numerical", "categorical"]:
            raise ValueError(f"{value} is not a valid type!")
        else:
            self._type = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if isinstance(value.strip(), str):
            self._name = value
        else:
            raise ValueError(f"{value} is not a valid name!")
