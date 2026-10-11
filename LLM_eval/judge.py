from database_manager.manager import TestcaseManager, ProjectManager
from evaluators.agent_eval import agent_eval
from evaluators.rag_eval import rag_eval
import pandas as pd
from typing import Protocol

class System(Protocol):
    def run(self, case: dict) -> dict:
        pass

class Judge:
    def __init__(self, project_id):
        self.project_id = project_id
        self.testcaseManager = TestcaseManager(project_id)
        self.project_type = ProjectManager(project_id).get_project_type()
        self.answer_generator = {"AGENT": agent_eval, "RAG": rag_eval}.get(self.project_type)
    
    def get_golden_set(self):
        if not self.answer_generator: return "Unable to find evaluator"
        all_testcases = self.testcaseManager.get_testcases()
        df = pd.DataFrame([vars(p) for p in all_testcases])
        sample = df.groupby('type', group_keys=False).apply(lambda x: x.sample(frac=0.2))
        golden_set = []
        for s in sample.to_dict("records"):
            s["golden_answer"]["query"] = s["query"]
            golden_set.append(s["golden_answer"])
        return golden_set

    def evaluate_agents(self):
        if not self.answer_generator: return "Unable to find evaluator"
        golden_set = self.get_golden_set()
        if self.project_type == "AGENT":
            agent_eval(golden_set).evaluate()
        elif self.project_type == "RAG":
            rag_eval(golden_set).evaluate()
        else:
            print("Unsupported project type")
    

if __name__ == "__main__":
    Judge("ReAct").evaluate_agents()
    
