import os
import json
from anthropic import Anthropic
from retriever import Retriever

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"

GOLDEN_SET = [
    {"query": "What is self-attention and how is it computed?",
     "source": "Attention_is_all_you_need.txt",
     "relevant": ["attention", "query", "key", "value"]},
    {"query": "What objectives is BERT pre-trained with?",
     "source": "BERT.txt",
     "relevant": ["masked", "language model", "next sentence"]},
    {"query": "What is chain-of-thought prompting?",
     "source": "Chain_of_thought.txt",
     "relevant": ["chain of thought", "reasoning", "intermediate"]},
    {"query": "How does LoRA reduce the number of trainable parameters?",
     "source": "LoRa.txt",
     "relevant": ["low-rank", "rank", "trainable"]},
    {"query": "What are the main steps of RLHF?",
     "source": "RLHF.txt",
     "relevant": ["reward model", "human feedback", "reinforcement"]},
]

NL = "\n"

def generate_answer(query, contexts):
    ctx = "\n---\n".join(contexts)
    resp = client.messages.create(
        model=MODEL, max_tokens=500, temperature=0.0,
        system="Answer the question using ONLY the provided context. If the answer is not in the context, say you don't know.",
        messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}],
    )
    return resp.content[0].text

def _judge(prompt):
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=256, temperature=0.0,
            system="You are a strict, objective RAG evaluation judge. Output exactly one JSON object, nothing else.",
            messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "{"}],
        )
        data = json.loads("{" + resp.content[0].text)
        return max(0.0, min(1.0, float(data.get("score", 0.0))))
    except Exception as e:
        print("judge error:", e)
        return 0.0

def faithfulness(contexts, answer):
    return _judge(
        f"Context:{NL}{NL.join(contexts)}{NL}{NL}Answer:{NL}{answer}{NL}{NL}"
        "Score 1.0 if every claim in the Answer is directly supported by the Context, "
        "0.0 if any claim is unsupported by or contradicts it. Judge grounding only, not truth or relevance."
        f'{NL}Return only JSON: {{"reason":"...","score":0.0}}'
    )

def answer_relevance(query, answer):
    return _judge(
        f"Question: {query}{NL}Answer: {answer}{NL}{NL}"
        "Score 1.0 if the Answer directly and completely addresses the Question, 0.0 if it is off-topic or evasive. "
        "Ignore factual truth."
        f'{NL}Return only JSON: {{"reason":"...","score":0.0}}'
    )

def context_relevance(query, contexts):
    return _judge(
        f"Question: {query}{NL}Context:{NL}{NL.join(contexts)}{NL}{NL}"
        "Score 1.0 if the Context contains the information needed to answer the Question, 0.0 if it is off-topic or useless. "
        "Judge the context, not any answer."
        f'{NL}Return only JSON: {{"reason":"...","score":0.0}}'
    )

def retrieval_metrics(docs, gold):
    rels = [s.lower() for s in gold["relevant"]]
    first = None
    for rank, d in enumerate(docs, start=1):
        dl = d.lower()
        if any(s in dl for s in rels):
            first = rank
            break
    return (1.0 if first else 0.0), (1.0 / first if first else 0.0)

def evaluate(k=5):
    r = Retriever(chunks=k)
    agg = {"recall": 0.0, "mrr": 0.0, "faith": 0.0, "ans": 0.0, "ctx": 0.0}
    print(f"{'query':46}recall  mrr   faith  ans   ctx")
    print("-" * 80)
    for item in GOLDEN_SET:
        docs = [doc for _id, doc in r.retrieve(item["query"])]
        recall, mrr = retrieval_metrics(docs, item)
        answer = generate_answer(item["query"], docs)
        f_ = faithfulness(docs, answer)
        a_ = answer_relevance(item["query"], answer)
        c_ = context_relevance(item["query"], docs)
        for key, val in zip(agg, [recall, mrr, f_, a_, c_]):
            agg[key] += val
        print(f"{item['query'][:46]:46}{recall:.2f}   {mrr:.2f}  {f_:.2f}   {a_:.2f}  {c_:.2f}")
    n = len(GOLDEN_SET)
    print("-" * 80)
    print(f"{'AVERAGE':46}{agg['recall']/n:.2f}   {agg['mrr']/n:.2f}  "
          f"{agg['faith']/n:.2f}   {agg['ans']/n:.2f}  {agg['ctx']/n:.2f}")

if __name__ == "__main__":
    evaluate(k=5)
