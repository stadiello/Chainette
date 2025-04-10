from sentence_transformers import SentenceTransformer
from rag.indexer import SimpleIndexer
from rag.graph_builder import GraphBuilder
from rag.retriever import GraphRetriever
from rag.rag_chain import RAGChain

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Index
text = """Le soleil est une étoile.
La Terre tourne autour du Soleil.
Les planètes du système solaire sont huit.
Pluton a été rétrogradée au rang de planète naine."""

indexer = SimpleIndexer(embedding_model)
nodes = indexer.index_from_text(text)

# 2. Build graph
graph = GraphBuilder()
for node in nodes:
    graph.add_node(node)

# 3. Retrieve + RAG
retriever = GraphRetriever(graph, embedding_model)
class DummyLLM:
    def generate(self, prompt): return f"Réponse (fake) : {prompt[:100]}..."

class DummyPrompt:
    def render(self, context, question): return f"Contexte:\n{context}\nQuestion: {question}"

rag = RAGChain(retriever, DummyLLM(), DummyPrompt())

print(rag.run("Qu'est-ce que le Soleil ?"))