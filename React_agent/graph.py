from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from tools import wikipedia_search_tool, get_current_weather, get_current_time, delete_files, get_current_file_path
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

class AgentGraph:
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000
        )
        self.sensitive_tools = ["delete_files", "get_current_file_path"]
        self.tools = [wikipedia_search_tool, get_current_weather, get_current_time, get_current_file_path, delete_files]
        self.llm_with_tools = self.model.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    def _agent(self, state):
        return {"messages": [self.llm_with_tools.invoke(state["messages"])]}
    
    def _route_after_agent(self, state):
        last = state["messages"][-1]
        return "review" if getattr(last, "tool_calls", None) else "end"
    
    def _human_review_node(self, state: AgentState):
        last_message = state["messages"][-1]
        for tool_call in last_message.tool_calls:
            if tool_call["name"] in self.sensitive_tools:
                human_response = interrupt({
                    "tool": tool_call["name"],
                    "args": tool_call["args"]
                })

                if human_response.get("decision") != "approve":
                    return Command(goto="__end__")
        return {}

    def _build_graph(self):
        builder = StateGraph(AgentState)
        tool_node = ToolNode(self.tools)
        builder.add_node("agent", self._agent)
        builder.add_node("human_review", self._human_review_node)
        builder.add_node("tools", tool_node)

        builder.add_edge(START, "agent")
        builder.add_edge("agent", "human_review")
        builder.add_edge("human_review", "tools")
        builder.add_edge("tools", "agent")
        builder.add_conditional_edges("agent", self._route_after_agent,
                              {"review": "human_review", "end": END})
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
        return graph

    def run_query(self, query: str):
        config = {"configurable": {"thread_id": str(uuid4())}}
        stream_input = {
            "messages": [("user", query)]
        }
        while True:
            for event in self.graph.stream(stream_input, config, stream_mode="updates"):
                for node, update in event.items():
                    if isinstance(update, dict) and update.get("messages"):
                        print(f"[{node}] {update['messages'][-1].content}")
            snap = self.graph.get_state(config)
            if not snap.next: break
            payload = snap.tasks[0].interrupts[0].value
            answer = input(f"Approve {payload}? [approve/reject]")
            stream_input = Command(resume = {"decision": answer})