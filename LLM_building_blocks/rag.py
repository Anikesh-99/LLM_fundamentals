from sentence_transformers import SentenceTransformer
import numpy as np
import heapq
# Load a lightweight, popular embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "This is the first document",
    "This is the second document",
    "This is the third document",
    "This is the first file",
    "This is the fourth document"
]

def chunker(text, size = 200, overlap = 40):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start: start + size]
        if not piece: break
        chunks.append((start, " ".join(piece)))
        if start + size >= len(words): break
    return chunks

embeddings = []
for doc in docs:
    for start, chunk in chunker(docs):
        embeddings.append(model.encode(doc))

def length(x):
    ln = 0
    for i in x:
        ln += i**2
    return np.sqrt(ln)

def cosine_sim(x, y):
    return np.dot(x, y)/(length(x) * length(y))

s = "What is in the 1st folder"
query_embedding = model.encode(s)
scores = []
for i in range(len(embeddings)):
    scores.append(cosine_sim(query_embedding, embeddings[i]))

k = 2
top_scores = []
for i, score in enumerate(scores):
    heapq.heappush(top_scores, (score, docs[i]))
    if len(top_scores) > k:
        heapq.heappop(top_scores)
top_scores = sorted(top_scores)
print([doc for score, doc in top_scores])