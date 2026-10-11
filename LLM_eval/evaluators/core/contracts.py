from typing import Protocol, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class EvalResult:
    answer: str = ""
    trajectory_path: str = ""
    contexts: list[str] = field(default_factory=list)

class System(Protocol):
    def run(self, case: dict) -> EvalResult:
        pass

class Metric(ABC):
    name: str
    required_fields: list[str]

    def score(self, case: dict, result: EvalResult) -> dict:
        if not self._check_fields(case): return self._give_error()
        return self._getScore(case, result)
    
    def _check_fields(self, case:dict) -> bool:
        for field in self.required_fields:
            if field not in case: return False
        return True
    
    def _give_error(self) -> dict:
        return {"error": f"Invalid schema provided. Please provide: {self.required_fields}"}
    
    @abstractmethod
    def _getScore(self, case: dict, result: EvalResult) -> dict:
        pass

class GoldenSet(ABC):
    name: str
    source: str

    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source

    @abstractmethod
    def load_set(self, version: int = -1) -> list[dict]:
        pass

    @abstractmethod
    def add_testcases(self, testcases: list[dict]) -> bool:
        pass
