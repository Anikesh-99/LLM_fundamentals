import os
import pickle
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
EMBED_MODEL = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, collection_name="AI_research", chunks=5):
        self.k = chunks
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        client = chromadb.PersistentClient(path=os.path.join(HERE, "chroma_db"))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        # same embedding function as ingest, so query and doc vectors are comparable
        self.collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)
        with open(os.path.join(HERE, "bm25_index.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(HERE, "ordered_doc_ids.pkl"), "rb") as f:
            self.bm25_ids = pickle.load(f)

    def retrieve(self, query):
        # dense
        dense = self.collection.query(query_texts=[query], n_results=self.k * 4, include=["documents"])
        dense_ids = dense["ids"][0]
        # sparse (BM25), tokenized the same way as at index time
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25 = np.argsort(bm25_scores)[::-1][:self.k * 2]
        sparse_ids = [self.bm25_ids[i] for i in top_bm25]
        # weighted reciprocal rank fusion (0.3 sparse / 0.7 dense)
        rrf = {}
        for rank, did in enumerate(sparse_ids):
            rrf[did] = rrf.get(did, 0.0) + 0.3 * (1.0 / (60 + rank + 1))
        for rank, did in enumerate(dense_ids):
            rrf[did] = rrf.get(did, 0.0) + 0.7 * (1.0 / (60 + rank + 1))
        fused_ids = [d for d, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:self.k * 2]]
        # hydrate, then cross-encoder rerank to top-k (rerank re-sorts, so input order is irrelevant)
        hydrated = self.collection.get(ids=fused_ids, include=["documents"])
        docs, ids = hydrated["documents"], hydrated["ids"]
        if not docs:
            return []
        scores = self.reranker.predict([[query, d] for d in docs])
        order = np.argsort(scores)[::-1][:self.k]
        return [[ids[i], docs[i]] for i in order]

if __name__ == "__main__":
    for cid, doc in Retriever(chunks=5).retrieve("What is self attention"):
        print(cid, "::", doc[:140])
