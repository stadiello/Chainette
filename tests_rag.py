from rag.graph_builder import GraphBuilder
from rag.node import Node
import matplotlib.pyplot as plt
import networkx as nx



# Embeddings fictifs
embedding_a = [1.0, 0.0, 0.0]
embedding_b = [0.9, 0.1, 0.0]
embedding_c = [0.0, 1.0, 0.0]

# Trois nodes
node_a = Node(id="A", content="Bonjour tout le monde", embedding=embedding_a, metadata={}, neighbors=[])
node_b = Node(id="B", content="Salut à tous", embedding=embedding_b, metadata={}, neighbors=[])
node_c = Node(id="C", content="Je parle anglais", embedding=embedding_c, metadata={}, neighbors=[])

# Création du builder avec seuil bas pour test
builder = GraphBuilder(similarity_threshold=0.85)

builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)

# Test affichage voisins
print("Voisins de A:", builder.get_neighbors("A"))
print("Voisins de B:", builder.get_neighbors("B"))
print("Voisins de C:", builder.get_neighbors("C"))

nx.draw(builder.graph, with_labels=True)
plt.show()