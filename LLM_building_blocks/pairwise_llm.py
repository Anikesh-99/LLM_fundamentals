import os
from anthropic import Anthropic
import json

user_prompt = "What is the size of the world?"
response_A = "The world is 90 Million square kilometers big"
response_B = "The world has 8 billion people living on it"

class Pairwise_Judge:
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-5"
    
    def _llm_judge(self, prompt):
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.0,
                system="You are a strict, objective RAG evaluation judge. Output exactly one JSON object, nothing else.",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"}
                ],
            )
            data = json.loads("{" + resp.content[0].text)
            return data
        except Exception as e:
            print(f"judge error: {e}")
            return {"winner": "unknown", "reason": f"Error in API response: {e}"}
    
    def evaluate_answer_pair(self, user_prompt, A, B):
        prompt = f"""
        You are an impartial AI response evaluator. Based on the following user prompt: {user_prompt}, 
        judge which response is better suited to answering it. A's response is: {A} and the B's response is: {B}. 
        Respond only in this JSON format: {{"winner": "C", "reason": "brief explanation"}}
        """
        MAX_RETRIES = 3
        for retry in range(MAX_RETRIES):
            response = self._llm_judge(prompt)
            try:
                winner = response["winner"]
                if winner == "unknown": 
                    prompt += f"The response was inconclusive due to {response['reason']}"
                    continue
                return winner
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                prompt += f"\nThe previous failure reasoning was a decoding error: {e}"
            except KeyError as e:
                print(f"Invalid key provided: {e}")
                prompt += f"""\nThe previous failure reasoning was an issue with the schema you returned as a response which did not contain the key: "Winner" """
        return ""

    def test_responses(self, user_prompt, A, B):
        winner1 = self.evaluate_answer_pair(user_prompt, A, B)
        winner2 = self.evaluate_answer_pair(user_prompt, B, A)
        winner1 = "A" if winner1 == "A" else "B" if winner1 == "B" else None
        winner2 = "B" if winner1 == "A" else "A" if winner1 == "B" else None
        if not winner1 or not winner2: return "inconclusive"
        return winner1 if winner1 == winner2 else "tie"

judge = Pairwise_Judge()
print(judge.test_responses(user_prompt, response_A, response_B))