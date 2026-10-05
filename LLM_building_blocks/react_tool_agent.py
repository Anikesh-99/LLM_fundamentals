import datetime
import json
import os
from anthropic import Anthropic

def calculator_tool(x, y, op):
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    if op not in ops: raise Exception("Invalid operation")
    try:
        return op(x, y)
    except Exception as e:
        raise e

def search_tool(x):
    return x

def current_time():
    return datetime.datetime.now()

tools = {
    "calculator": {
        "func": calculator_tool,
        "description": "Performs operation on 2 numbers",
        "arguments": "x -> float, y -> float, op -> ['+', '-', '*', '/']",
        "returns": "float with operation or raises value exception if arguments are invald"
    },
    "search": {
        "func": search_tool,
        "description": "Searches a corpus for the text",
        "arguments": "x -> str",
        "returns": "x if x is found inside corpus"
    },
    "current_time": {
        "func": current_time,
        "description": "Finds the current time in PST",
        "arguments": "None",
        "returns": "current datetime object"
    }
}

steps = 0
max_steps = 4
system_prompt = "You are a smart assistant with access to tools. To use a tool, respond with JSON in the following format: {\"action\": \"tool_name\", \"action_input\": \"arguments\"}. If you have enough to answer the user respond with: {\"final_answer\": \"your answer here\"}"

messages = [
    {"role": "user", "content": "Find whether xtr is inside a corpus, find the current time and add the day and month of the current time together"}
]
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

while steps <= max_steps:
    steps += 1
    resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=256,
                temperature=0.0,
                system=system_prompt,
                messages=messages
            )
    response_text = resp.content[0].content
    messages.append({"role": "assistant", "content": response_text})
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error: Model output was not valid JSON.")
        break

    if "final_answer" in data:
        print(f"Final Answer: {data['final_answer']}")
        break
    
    action_name = data.get("action")
    action_input = data.get("action_input")

    if action_name in tools:
        # the action_input as of now is coming back as a string i assume or a list which would need to be handled in the function
        try:
            obs = tools[action_name]["func"](**action_input)
        except Exception as e:
            obs = f"ERROR: {e}"
        print(f"Observed result from {action_name}: {obs}")
        messages.append(
            {"role": "user", "content": f"Observation: {obs}"}
        )
    else:
        print(f"Error: Unknown tool {action_name}")
        break

