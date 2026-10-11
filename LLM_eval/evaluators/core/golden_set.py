from .contracts import GoldenSet
from database_manager.manager import TestcaseManager
from database_manager.model import Testcase
import json

# top-level columns on Testcase; everything else in a case is packed into golden_answer
_META = ("query", "type", "version", "project_id", "id")

class FileGoldenSet(GoldenSet):
    def load_set(self, version: int = -1) -> list[dict]:
        with open(self.source, encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]
        if not cases:
            return []
        if version == -1:
            version = max(c.get("version", 0) for c in cases)
        return [c for c in cases if c.get("version", 0) == version]

    def add_testcases(self, testcases) -> bool:
        try:
            with open(self.source, "a", encoding="utf-8") as f:
                for tc in testcases:          # cases are plain dicts
                    f.write(json.dumps(tc) + "\n")
            return True
        except Exception as e:
            print("add_testcases (file) failed:", e)
            return False

class DatabaseGoldenSet(GoldenSet):
    def load_set(self, version: int = -1) -> list[dict]:
        # flatten golden_answer out to the top level; drop ORM internals
        rows = TestcaseManager(self.source).get_testcases(version)
        return [{**r.golden_answer, "query": r.query, "type": r.type, "version": r.version} for r in rows]

    def add_testcases(self, testcases: list[dict]) -> bool:
        # pack the task fields back into golden_answer (mirror of load_set)
        rows = []
        for c in testcases:
            golden = {k: v for k, v in c.items() if k not in _META}
            rows.append(Testcase(
                project_id=self.source,
                query=c["query"],
                type=c["type"],
                version=c["version"],
                golden_answer=golden,
            ))
        return TestcaseManager(self.source).add_new_version(rows)
