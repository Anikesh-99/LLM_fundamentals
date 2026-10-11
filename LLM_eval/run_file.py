import os, sys, json
# put LLM_fundamentals on the path so the adapter can import React_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.core.golden_set import FileGoldenSet
from evaluators.core.metrics import Trajectory_Metric, Answer_Relevance_Metric
from adapters.react_system import ReActSystem

HERE = os.path.dirname(os.path.abspath(__file__))

def run(golden_path, system, metrics):
    cases = FileGoldenSet("agent-golden", golden_path).load_set()
    # fail fast: every case must carry the fields the chosen metrics need
    need = set().union(*[set(m.required_fields) for m in metrics]) if metrics else set()
    for i, c in enumerate(cases):
        missing = need - c.keys()
        if missing:
            raise ValueError(f"case {i} missing {missing} for the chosen metrics")

    rows = []
    for case in cases:
        result = system.run(case)
        row = {"query": case["query"][:50]}
        for m in metrics:
            row.update(m.score(case, result))
        rows.append(row)
        print(row)

    out = os.path.join(HERE, "results.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nwrote {len(rows)} rows -> {out}")
    return rows

if __name__ == "__main__":
    run(os.path.join(HERE, "golden", "agent.jsonl"),
        ReActSystem(),
        [Trajectory_Metric(), Answer_Relevance_Metric()])
