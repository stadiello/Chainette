import numpy as np
from typing import List
from rag.node import Node

class GraphRetriever:
    def __init__(self, graph_builder, embedding_model, top_k=3, expand_neighbors=True):
        self.graph_builder = graph_builder
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.expand_neighbors = expand_neighbors

    def retrieve(self, query: str) -> List[str]:
        query_embedding = self.embedding_model.encode([query])[0]
        all_nodes = [self.graph_builder.get_node(n_id) for n_id in self.graph_builder.graph.nodes]

        scored = [
            (node, self.cosine_similarity(query_embedding, node.embedding))
            for node in all_nodes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_nodes = [node for node, score in scored[:self.top_k]]

        if self.expand_neighbors:
            neighbor_ids = {
                n_id
                for node in top_nodes
                for n_id in node.neighbors
            }
            neighbor_nodes = [self.graph_builder.get_node(n_id) for n_id in neighbor_ids]
            top_nodes += neighbor_nodes

        return [node.content for node in top_nodes]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        a = np.array(vec1)
        b = np.array(vec2)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))