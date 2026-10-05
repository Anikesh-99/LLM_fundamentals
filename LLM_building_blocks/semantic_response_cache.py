from sentence_transformers import SentenceTransformer, util
from anthropic import Anthropic
import os 

model = SentenceTransformer('all-MiniLM-L6-v2')

cache = []
hits = 0
total_queries = 0
queries = ["What is the closest planet to earth?", "When was the galaxy formed?", "what is the nearest object to Earth", "how old is the galaxy", "what is the size of the universe", "how old is the universe"]
threshold = 0.8
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def generate_answer(query):
    global hits, total_queries
    embedding = model.encode(query)
    total_queries += 1
    for k, answer in cache:
        # this is the brute force way. We could do the more optimal hnsw but not for a small corpus
        if util.cos_sim(embedding, k).item() >= threshold:
            hits += 1
            return answer
    resp = client.messages.create(
        model="claude-sonnet-4-5-202201010",
        max_tokens=1000,
        temperature=0.2,
        system="You are a knowledgable professor in the world of science answering questions from your students",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    answer = resp.content[0].text
    cache.append(embedding, answer)
    return answer
        
for query in queries:
    generate_answer(query)

hit_rate = hits/total_queries