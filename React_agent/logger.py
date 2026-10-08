import json
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, chat_id, directory=""):
        base = Path(directory) if directory else Path.cwd()
        self.filepath = base / "logs" / f"{chat_id}.jsonl"      # JSONL: one JSON object per line
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.chat_id = str(chat_id)

    def log_action(self, type="unknown", resp="", token=0, tool="unknown", **extra):
        record = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chat_id": self.chat_id,
            "type": type,
            "token": token,
        }
        record.update(self._format_content(type, resp, tool))
        record.update(extra)                                    # confidence, needs_permission, user_input, step...
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")     # default=str keeps UUIDs/odd args serializable

    def _format_content(self, type, resp, tool):
        if type == "tool_input":      return {"tool": tool, "arguments": resp}
        elif type == "tool_response": return {"tool": tool, "output": resp}
        elif type == "start_chat":    return {"query": resp}
        elif type == "answer":        return {"answer": resp}
        else:                         return {"data": resp}
