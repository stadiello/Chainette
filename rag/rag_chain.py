class RAGChain:
    def __init__(self, retriever, llm, prompt_template):
        ...

    def run(self, query):
        docs = self.retriever.retrieve(query)
        prompt = self.prompt_template.render(context=docs, question=query)
        return self.llm.generate(prompt)
    

# # 1. Indexation
# indexer = SimpleIndexer(embedding_model)
# nodes = indexer.index_from_text(document)
# for node in nodes:
#     graph_builder.add_node(node)

# # 2. RAG
# retriever = GraphRetriever(graph_builder, embedding_model)
# rag = RAGChain(retriever, llm=myLLM, prompt_template=myPrompt)

# response = rag.run("C'est quoi un graphe de connaissance ?")
# print(response)