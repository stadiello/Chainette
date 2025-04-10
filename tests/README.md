💅 Ah, le README.
# 🧠 `chainette.rag` — RAG Graph Contextualisé

Un module de Retrieval-Augmented Generation (RAG) basé sur une **structure de graphe sémantique**.

---

## Fonctionnalités

- Stockage de documents sous forme de `Node` (texte + embedding)
- Construction automatique d’un **graphe contextuel** entre les chunks
- Récupération de documents enrichie via leurs **voisins sémantiques**
- Prêt pour l’intégration dans une pipeline LLM ou un agent

---

## Structure du module

rag/
├── node.py             # structure de base des “noeuds de connaissance”
├── graph_builder.py    # ajoute les noeuds, construit les liens entre eux
├── retriever.py        # récupère les chunks + voisins
├── indexer.py          # découpe et embedd les documents
├── rag_chain.py        # classe de haut niveau qui gère tout le flux RAG

---

## Installation

Ce module fait partie de la bibliothèque `chainette`. Si tu bosses dans un environnement local :

```bash
pip install -r requirements.txt
```

Tu auras besoin de :
	•	networkx
	•	numpy
	•	sentence-transformers (ou autre lib d’embedding)

⸻

Utilisation rapide

1. Créer et ajouter des Nodes

```python
from node import Node
from graph_builder import GraphBuilder

builder = GraphBuilder(similarity_threshold=0.85)

node = Node(
    id="chunk_42",
    content="Ceci est un extrait de document.",
    embedding=[0.23, 0.76, 0.11, ...],
    metadata={"source": "doc1.md"},
)

builder.add_node(node)
```


⸻

2. Récupérer les voisins sémantiques

```python
neighbors = builder.get_neighbors("chunk_42")
for n in neighbors:
    print(n.id, n.content)
```

⸻

3. Intégrer dans une chaîne RAG

```python
from rag_chain import RAGChain

rag = RAGChain(retriever=yourRetriever, llm=yourLLM, prompt_template=yourPrompt)
response = rag.run("Qu'est-ce que le GraphBuilder fait exactement ?")

```

⸻

Tests

Un test de base est fourni dans test_graph_builder.py :

```bash
python test_graph_builder.py
```

Tu peux y injecter tes propres Nodes et visualiser le graphe avec matplotlib.

⸻

TODO (coming soon™)
	•	Ajout d’un backend vector store (Chroma, Qdrant)
	•	Graphe orienté avec poids ajustables
	•	Visualisation en HTML avec pyvis
	•	Export GraphML

⸻

Pourquoi ce module s’appelle-t-il chainette ?

Parce que c’est une chaîne légère, élégante, modulaire — pas une usine comme LangChain.
Et surtout parce que le .graph ici n’est pas un sapin de Noël, c’est un cerveau miniature.

⸻
