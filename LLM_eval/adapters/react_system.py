"""Adapter: makes the ReAct agent satisfy the evaluator's System contract.
This is the one file allowed to import the project; the eval core imports nothing here."""
from evaluators.core.contracts import EvalResult
from React_agent.agent import ReAct

class ReActSystem:
    def run(self, case: dict) -> EvalResult:
        agent = ReAct(
            permission_fn=self._permission(case.get("approvals", {})),
            tool_overrides=self._stubs(case.get("stub_tools", [])),
        )
        answer = agent.generate_answer(case["query"])
        return EvalResult(answer=answer, trajectory_path=str(agent.logger.filepath))

    @staticmethod
    def _permission(policy):
        # non-interactive approver driven by the case's {tool: answer} policy
        def _ask(prompt):
            for tool, ans in policy.items():
                if tool in prompt:
                    return ans
            return "yes"
        return _ask

    @staticmethod
    def _stubs(stub_tools):
        # swap destructive tools for a no-op recorder so the eval never touches disk
        overrides = {}
        for t in stub_tools:
            def _stub(file_path=None, _t=t, **kw):
                return f"[stub] {_t} executed on {file_path or kw}"
            overrides[t] = _stub
        return overrides
