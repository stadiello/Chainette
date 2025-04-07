class RAGChain:
    def __init__(self, retriever, llm, prompt_template):
        ...

    def run(self, query):
        docs = self.retriever.retrieve(query)
        prompt = self.prompt_template.render(context=docs, question=query)
        return self.llm.generate(prompt)