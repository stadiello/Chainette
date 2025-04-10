from rag.node import Node
from typing import List
import uuid

class SimpleIndexer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def index_from_text(self, text: str, metadata: dict = None) -> List[Node]:
        # Naïf : split par paragraphes
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        embeddings = self.embedding_model.encode(chunks).tolist()

        nodes = []
        for i, chunk in enumerate(chunks):
            node_id = str(uuid.uuid4())
            node = Node(
                id=node_id,
                content=chunk,
                embedding=embeddings[i],
                metadata=metadata or {},
                neighbors=[],
            )
            nodes.append(node)

        return nodes