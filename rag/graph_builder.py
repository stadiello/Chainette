import networkx as nx
from rag.node import Node
import numpy as np
from typing import List

class GraphBuilder:
    def __init__(self, similarity_threshold: float = 0.8):
        self.graph = nx.Graph()
        self.similarity_threshold = similarity_threshold

    def add_node(self, node: Node):
        self.graph.add_node(node.id, data=node)

        for other_id in self.graph.nodes:
            if other_id == node.id:
                continue

            other_node = self.graph.nodes[other_id]["data"]
            sim = self.cosine_similarity(node.embedding, other_node.embedding)

            if sim >= self.similarity_threshold:
                self.graph.add_edge(node.id, other_id, weight=sim)
                node.neighbors.append(other_id)
                other_node.neighbors.append(node.id)

    def get_node(self, node_id: str) -> Node:
        return self.graph.nodes[node_id]["data"]

    def get_neighbors(self, node_id: str) -> List[Node]:
        neighbor_ids = list(self.graph.neighbors(node_id))
        return [self.get_node(n_id) for n_id in neighbor_ids]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        a = np.array(vec1)
        b = np.array(vec2)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))