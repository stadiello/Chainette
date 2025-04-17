from dataclasses import dataclass
from typing import Any

@dataclass
class Node:
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    neighbors: list[str]
