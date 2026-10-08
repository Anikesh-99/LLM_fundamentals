from anthropic import Anthropic
import os
import json
import uuid
from tools import wikipedia_search_tool, get_current_weather, get_current_time, delete_files, get_current_file_path
from logger import Logger

class ReAct:
    def __init__(self, query: str):
        self.required_confidence = 0.8
        self.query = query
        self.MAX_STEPS = 5
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.tools = {
            "time": {
                "func": get_current_time,
                "description": "Returns the current date and time of the user",
                "arguments": "none",
                "permission_required": False,
                "returns": "string with date and time as %Y-%m-%d %H:%M:%S",
            },
            "weather": {
                "func": get_current_weather,
                "description": "Returns the weather at a location (defaults to Los Angeles)",
                "arguments": "lat -> float, lon -> float (both optional)",
                "permission_required": False,
                "returns": "string with temperature in Celsius and conditions",
            },
            "search": {
                "func": wikipedia_search_tool,
                "description": "Searches Wikipedia for content regarding the query",
                "arguments": "query -> str",
                "permission_required": False,
                "returns": "string showing search results",
            },
            "delete": {
                "func": delete_files,
                "description": "Deletes a file from disk",
                "arguments": "file_path -> str",
                "permission_required": True,
                "returns": "string status: deleted / not found / refused if outside the sandbox",
            },
            "get_filepath": {
                "func": get_current_file_path,
                "description": "Resolves the absolute path of a file name",
                "arguments": "file_name -> str",
                "permission_required": True,
                "returns": "string absolute path",
            },
        }
        self.chat_id = uuid.uuid4()
        self.logger = Logger(self.chat_id)
        self.logger.log_action("start_chat", resp=self.query)      # LOG: the run's query

    def create_tool_string(self):
        # strip the un-serializable 'func' before handing the schema to the model
        view = {name: {k: v for k, v in meta.items() if k != "func"} for name, meta in self.tools.items()}
        return json.dumps(view, indent=2)

    def generate_answer(self):
        system_prompt = (
            "You are a smart assistant with access to tools.\n"
            "To use a tool, respond with ONLY a JSON object in this exact format:\n"
            '{"action": "<tool_name>", "action_input": {<named arguments as a JSON object>}, '
            '"confidence": <number between 0 and 1>}\n'
            "Always include a numeric confidence. For a tool that takes no arguments use an empty "
            'object: "action_input": {}.\n'
            f"The tools you have access to are:\n{self.create_tool_string()}\n"
            "When you have enough to answer the user, respond with ONLY:\n"
            '{"final_answer": "<your answer>"}'
        )

        messages = [{"role": "user", "content": self.query}]
        steps = 0
        while steps < self.MAX_STEPS:
            steps += 1
            resp = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                temperature=0.0,
                system=system_prompt,
                messages=messages,
            )
            response_text = resp.content[0].text
            tokens = resp.usage.input_tokens + resp.usage.output_tokens   # LOG: token usage of this model call
            messages.append({"role": "assistant", "content": response_text})

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                print("Error: model output was not valid JSON.")
                messages.append({"role": "user", "content": "Observation: your output was not valid JSON. Reply with a single JSON object only."})
                continue

            if "final_answer" in data:
                self.logger.log_action("answer", resp=data["final_answer"], token=tokens)   # LOG: final answer + tokens
                print(data["final_answer"])
                return data["final_answer"]

            action_name = data.get("action")
            action_input = data.get("action_input") or {}        # no-arg tools -> {}
            # fail-safe: missing or unparseable confidence is treated as LOW (so we ask)
            try:
                confidence = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            if action_name not in self.tools:
                print(f"Error: unknown tool {action_name}")
                messages.append({"role": "user", "content": f"Observation: unknown tool {action_name}"})
                continue

            # ask the human when the tool is destructive OR the model is not confident enough
            needs_permission = self.tools[action_name]["permission_required"] or confidence < self.required_confidence

            # LOG: tool decision, with confidence + whether it needs a human gate
            self.logger.log_action("tool_input", resp=action_input, token=tokens, tool=action_name,
                                   confidence=confidence, needs_permission=needs_permission)

            user_decision = None   # what the human answered, if we asked
            try:
                if needs_permission:
                    user_decision = input(f"Allow {action_name} with {action_input} (confidence {confidence})? [y/N] ").strip().lower()
                    if user_decision in ("y", "yes"):
                        obs = self.tools[action_name]["func"](**action_input)
                    else:
                        obs = f"User declined {action_name} on {action_input}"
                else:
                    obs = self.tools[action_name]["func"](**action_input)
            except Exception as e:
                obs = f"ERROR: {e}"

            # LOG: observation, plus the human's decision (None when no gate was shown)
            self.logger.log_action("tool_response", resp=obs, tool=action_name,
                                   needs_permission=needs_permission, user_input=user_decision)
            print(f"Observed result from {action_name}: {obs}")
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        return "Exceeded max steps so stopped generating answer"


if __name__ == "__main__":
    ReAct("Find the current time and weather in LA and give me some information about the city itself").generate_answer()
