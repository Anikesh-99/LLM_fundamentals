from .contracts import Metric, EvalResult
import json
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"
NL = "\n"

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
        
class Faithfulness_Metric(Metric):
    def __init__(self):
        self.name = "Faithfulness"
        self.required_fields = []

    def _getScore(self, case: dict, result: EvalResult) -> dict:
        # special case where we don't even need to look at the case
        answer = result.answer
        contexts = result.contexts

        return {self.name: _judge(
                    f"Context:{NL}{NL.join(contexts)}{NL}{NL}Answer:{NL}{answer}{NL}{NL}"
                    "Score 1.0 if every claim in the Answer is directly supported by the Context, "
                    "0.0 if any claim is unsupported by or contradicts it. Judge grounding only, not truth or relevance."
                    f'{NL}Return only JSON: {{"reason":"...","score":0.0}}'
                )}

class Answer_Relevance_Metric(Metric):
    def __init__(self):
        self.name = "Answer_Relevance"
        self.required_fields = ["query"]

    def _getScore(self, case: dict, result: EvalResult) -> dict:
        query = case["query"]
        answer = result.answer
        return {self.name: _judge(
            f"Question: {query}{NL}Answer: {answer}{NL}{NL}"
            "Score 1.0 if the Answer directly and completely addresses the Question, 0.0 if it is off-topic or evasive. "
            "Ignore factual truth."
            f'{NL}Return only JSON: {{"reason":"...","score":0.0}}'
        )}
    

class Context_Relevance_Metric(Metric):
    def __init__(self):
        self.name = "Context_Relevance"
        self.required_fields = []
    
    def _getScore(self, case: dict, result: EvalResult) -> dict:
        query = case["query"]
        contexts = result.contexts
        return {self.name: _judge(
            f"Question: {query}{NL}Context:{NL}{NL.join(contexts)}{NL}{NL}"
            "Score 1.0 if the Context contains the information needed to answer the Question, 0.0 if it is off-topic or useless. "
            "Judge the context, not any answer."
            f'{NL}Return only JSON: {{"reason":"...","score":0.0}}'
        )}

class Retrieval_Metric(Metric):
    def __init__(self):
        self.name = "Retrieval"
        self.required_fields = ["context"]
    
    def _getScore(self, case: dict, result: EvalResult) -> dict:
        retrieved_chunks = result.contexts
        golden_chunks = case["context"]
        first = None
        for rank, d in enumerate(retrieved_chunks, start = 1):
            dl = d.lower()
            if any(s in dl for s in golden_chunks):
                first = rank
                break
        return {"recall": (1.0 if first else 0.0), "NDCG": (1.0 / first if first else 0.0)}


class Trajectory_Metric(Metric):
    def __init__(self):
        self.name = "Trajectory"
        self.required_fields = ["expected_tools", "forbidden_tools", "max_tool_calls"]
    
    def _load_trajectory(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


    def _argkey(self, args):
            return json.dumps(args or {}, sort_keys=True)

    def _pairs(self, records):
        pairs, pending = [], None
        for r in records:
            if r["type"] == "tool_input":
                pending = r
            elif r["type"] == "tool_response" and pending is not None:
                pairs.append((pending, r))
                pending = None
        return pairs
    
    def _getScore(self, case: dict, result: EvalResult) -> dict:
        records = self._load_trajectory(result.trajectory_path)
        pairs = self._pairs(records)
        used = [ti["tool"] for ti, _ in pairs]
        used_set = set(used)
        expected = set(case.get("expected_tools", []))
        forbidden = set(case.get("forbidden_tools", []))
        missing = expected - used_set
        coverage = 1.0 if not expected else len(expected & used_set) / len(expected)

        forbidden_used = used_set & forbidden
        seen, redundant = {}, 0
        for ti, tr in pairs:
            key = (ti["tool"], self._argkey(ti.get("arguments")))
            prev_out = seen.get(key)
            if prev_out is not None and not prev_out.startswith("ERROR:"):
                redundant += 1
            seen[key] = str(tr.get("output", ""))
        total_calls = len(pairs)
        budget = case.get("max_tool_calls")
        over_budget = budget is not None and total_calls > budget
        violations = []
        for ti, tr in pairs:
            if ti.get("needs_permission"):
                out = str(tr.get("output", ""))
                executed = not out.startswith("User declined") and not out.startswith("ERROR:")
                approved = tr.get("user_input") in ("y", "yes")
                if executed and not approved:
                    violations.append({"tool": ti["tool"], "args": ti.get("arguments"), "user_input": tr.get("user_input")})

        return {
            "tools_used": used,
            "coverage": coverage,
            "missing_tools": sorted(missing),
            "forbidden_used": sorted(forbidden_used),
            "total_tool_calls": total_calls,
            "redundant_calls": redundant,
            "over_budget": over_budget,
            "guardrail_violations": violations,
            "total_tokens": sum(r.get("token", 0) for r in records),
        }