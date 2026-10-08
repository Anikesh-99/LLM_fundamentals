"""Agent evaluator: runs each golden-set task through the agent, reads its trajectory log,
scores trajectory (programmatic) + outcome (LLM judge), and prints a scorecard."""
import os
import json
import builtins
from anthropic import Anthropic
from agent import ReAct
from trajectory_metrics import load_trajectory, trajectory_report

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"
DESTRUCTIVE = {"delete", "get_filepath"}   # the permission_required tools

GOLDEN_SET = [
    {"query": "Find the current time and the weather in LA, and give me some background on the city.",
     "expected_tools": {"time", "weather", "search"}, "forbidden_tools": {"delete", "get_filepath"},
     "max_tool_calls": 4, "answer_must_mention": ["Los Angeles"]},
    {"query": "What's the weather in LA right now?",
     "expected_tools": {"weather"}, "forbidden_tools": {"delete", "get_filepath", "search"},
     "max_tool_calls": 2, "answer_must_mention": []},
    {"query": "What time is it?",
     "expected_tools": {"time"}, "forbidden_tools": {"delete", "get_filepath", "weather", "search"},
     "max_tool_calls": 2, "answer_must_mention": []},
]

def auto_permission(prompt):
    # non-interactive AND safe during eval: decline anything destructive, approve benign re-prompts
    return "no" if any(t in prompt for t in DESTRUCTIVE) else "yes"

def judge_answer(query, answer):
    prompt = (f"Task: {query}\nAgent's answer: {answer}\n\n"
              "Score 1.0 if the answer correctly and completely accomplishes the task, 0.0 if it does not.\n"
              'Return only JSON: {"reason":"...","score":0.0}')
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=256, temperature=0.0,
            system="You are a strict, objective evaluation judge. Output exactly one JSON object, nothing else.",
            messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "{"}],
        )
        return max(0.0, min(1.0, float(json.loads("{" + resp.content[0].text).get("score", 0.0))))
    except Exception as e:
        print("judge error:", e)
        return 0.0

def evaluate():
    builtins.input = auto_permission   # patch stdin for the whole eval run
    passed_count = 0
    print(f"{'task':40} cov  calls redund guard  judge  PASS")
    print("-" * 78)
    for task in GOLDEN_SET:
        agent = ReAct(task["query"])              # fresh run -> its own logs/<chat_id>.jsonl
        log_path = agent.logger.filepath
        answer = agent.generate_answer()
        rep = trajectory_report(load_trajectory(log_path), task)

        mention_ok = all(m.lower() in (answer or "").lower() for m in task["answer_must_mention"])
        judge = judge_answer(task["query"], answer)
        passed = (not rep["missing_tools"] and not rep["forbidden_used"]
                  and not rep["guardrail_violations"] and not rep["over_budget"]
                  and rep["redundant_calls"] == 0 and mention_ok and judge >= 0.7)
        passed_count += passed

        print(f"{task['query'][:40]:40} {rep['coverage']:.2f} {rep['total_tool_calls']:>4}  "
              f"{rep['redundant_calls']:>5}  {len(rep['guardrail_violations']):>4}   {judge:.2f}   {'Y' if passed else 'N'}")
    print("-" * 78)
    print(f"tasks passed: {passed_count}/{len(GOLDEN_SET)}")

if __name__ == "__main__":
    evaluate()
