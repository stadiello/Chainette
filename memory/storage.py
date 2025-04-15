import chromadb

class Storage:
    def __init__(self, persist_directory):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="memory")

