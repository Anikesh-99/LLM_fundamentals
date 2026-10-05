import os
from anthropic import Anthropic
import json

golden_set = {
    "user_input": [
        "What is the document in which NVIDIA's yearly balance sheet rests in?",
        "How does TSMC generate its revenue in 2026?"
    ],
    "response": [
        "NVIDIA's yearly balance sheet rests in the 10-K filing",
        "TSMC generates 90%% of it's revenue from corporate deals"
    ],
    "retrieved_context":[
        ["NVIDIA's balance sheet shows 10%% growth in revenue in its 10-K filing", "NVIDIA's revenue is reflected in it's 10-k filing"],
        ["TSMC's revenue has been more concentrated into it's corporate endevours", "Almost 90%% of TSMC's revenue comes from company deals"]
    ],
    "reference": [
        "NVIDIA's balance sheet",
        "TSMC's revenue growth"
    ]
}

LLM_answers = {
    "answer": [
        "NVIDIA's balance sheet is in the 10-Q filing",
        "TSMC's revenue is concentrated in corporate deals with around 90%% coming from them"
    ],
    "retrieved_chunks":[
        ["NVIDIA's balance sheet represents a 20%% growth in revenue in its 10-Q filing", "NVIDIA's quarterly revenue is reflected in it's 10-Q filing"],
        ["TSMC's revenue has been more concentrated into it's corporate endevours", "Almost 90%% of TSMC's revenue comes from company deals"]
    ]
}


class RAG_Eval:
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-5"

    def _llm_judge(self, prompt):
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=256,                       # required
                temperature=0.0,
                system="You are a strict, objective RAG evaluation judge. Output exactly one JSON object, nothing else.",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"},   # prefill: forces the reply to start as JSON
                ],
            )
            data = json.loads("{" + resp.content[0].text)   # prepend the prefilled "{"
            return max(0.0, min(1.0, float(data.get("score", 0.0))))  # clamp to [0,1]
        except Exception as e:
            print(f"judge error: {e}")
            return 0.0
        
    def evaluate_answer_relevance(self, question, generated_answer):
        prompt = f"""
            Determine if the Generated Answer directly addresses the user's initial Question.
            Ignore factual truth for this metric—only measure if it stays on topic and addresses the user's intent.
            
            Question: {question}
            Generated Answer: {generated_answer}
            
            Provide a relevance score between 0.0 (completely avoids the question or talks about something else) 
            and 1.0 (directly and completely answers the question).
            Respond ONLY in this JSON format: {{"reason": "brief explanation", "score": 0.95}}
            """
        return self._llm_judge(prompt)
    
    def evaluate_faithfulness(self, retrieved_context, generated_answer):
        context = "\n---\n".join(retrieved_context)
        prompt = f"""
            Determine if the Generated Answer is supported by the retrieved context.
            
            Context: {context}
            Generated Answer: {generated_answer}
            
            Score 1.0 if every claim in the answer is directly supported by the context; 0.0 if any claim is unsupported by or contradicts the context. Judge grounding only — not whether the answer is true or on-topic.
            Respond ONLY in this JSON format: {{"reason": "brief explanation", "score": 0.95}}
            """
        return self._llm_judge(prompt)

    def evaluate_context_relevance(self, question, retrieved_context):
        context = "\n---\n".join(retrieved_context)
        prompt = f"""
            Determine if the retrieved context actually pertains to the question and adds meaningful value to it.
            
            Context: {context}
            Question: {question}
            
            Score 1.0 if the context contains the information needed to answer the question; 0.0 if it's off-topic or useless. Judge the context, not the answer.
            Respond ONLY in this JSON format: {{"reason": "brief explanation", "score": 0.95}}
            """
        return self._llm_judge(prompt)
    
    def get_recall(self, actual_context, retrieved_context):
        correct = 0
        for actual in actual_context:
            if actual in retrieved_context:
                correct += 1
        return correct/(1.0 * len(retrieved_context))


    # retrieval can be graded better using recall than an LLM approach in my opinion so i left it out for now
    def evaluator(self, golden_set, LLM_answers):
        answer_relevances, answer_faithfulness, answer_context_relevance, recall = [[]] * 4
        print("Question | Relevance | Faithfulness | Context Relevance")
        for i, question in enumerate(golden_set["user_input"]):
            answer_relevances.append(self.evaluate_answer_relevance(question, LLM_answers["answer"][i]))
            answer_faithfulness.append(self.evaluate_faithfulness(LLM_answers["retrieved_chunks"][i], LLM_answers["answer"][i]))
            answer_context_relevance.append(self.evaluate_context_relevance(question, LLM_answers["retrieved_chunks"][i]))
            recall.append(self.get_recall(golden_set["retrieved_context"][i], LLM_answers["retrieved_chunks"][i]))
            print(question, answer_relevances[-1], answer_faithfulness[-1], answer_context_relevance[-1])

RAG_Eval().evaluator(golden_set, LLM_answers)