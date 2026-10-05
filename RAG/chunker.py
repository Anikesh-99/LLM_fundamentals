import os
import pickle
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize

EMBED_MODEL = "all-MiniLM-L6-v2"

def window_sentences(sentences, size=5, overlap=1):
    """Group sentences into overlapping chunks so each chunk carries context."""
    chunks, step = [], max(1, size - overlap)
    for i in range(0, len(sentences), step):
        piece = sentences[i:i + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if i + size >= len(sentences):
            break
    return chunks

class Chunker:
    def __init__(self, doc_path=None, collection_name="AI_research"):
        here = os.path.dirname(os.path.abspath(__file__))
        self.directory_path = doc_path or os.path.join(here, "corpus")
        self.client = chromadb.PersistentClient(os.path.join(here, "chroma_db"))
        self.collection_name = collection_name
        # One embedding function, used for BOTH ingest and query (set on the collection).
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self.MAX_BATCH_SIZE = 5000

    def generate_BM25(self, collection):
        all_data = collection.get(include=["documents"])
        chroma_ids = all_data["ids"]
        chroma_documents = all_data["documents"]
        tokenized_corpus = [doc.lower().split() for doc in chroma_documents]
        bm25 = BM25Okapi(tokenized_corpus)
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "bm25_index.pkl"), "wb") as f:
            pickle.dump(bm25, f)
        with open(os.path.join(here, "ordered_doc_ids.pkl"), "wb") as f:
            pickle.dump(chroma_ids, f)

    def chunk_documents(self):
        # idempotent: drop any existing collection so re-runs don't crash
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        collection = self.client.create_collection(
            name=self.collection_name, embedding_function=self.embedding_fn
        )
        offset = 0
        for filename in sorted(os.listdir(self.directory_path)):
            full_path = os.path.join(self.directory_path, filename)
            if not full_path.lower().endswith(".txt"):
                continue
            with open(full_path, "r", errors="replace", encoding="utf-8") as f:
                print(f"Chunking: {filename}")
                sentences = sent_tokenize(f.read())
            chunks = window_sentences(sentences)
            ids = [f"id_{i + offset}" for i in range(len(chunks))]
            metadatas = [{"source": filename} for _ in chunks]
            offset += len(chunks)
            for i in range(0, len(ids), self.MAX_BATCH_SIZE):
                end = i + self.MAX_BATCH_SIZE
                # documents only: the collection's embedding_function embeds them,
                # and the SAME function embeds queries at search time.
                collection.add(ids=ids[i:end], documents=chunks[i:end], metadatas=metadatas[i:end])
        self.generate_BM25(collection)
        print(f"done: {collection.count()} chunks indexed")

if __name__ == "__main__":
    Chunker().chunk_documents()
