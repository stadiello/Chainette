class Node:
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, str]
    neighbors: list[str]
