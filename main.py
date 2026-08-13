from typing import TypedDict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END




class State(TypedDict):
    messages: Sequence[BaseMessage]


state = State(messages=[HumanMessage("Could you tell me a grook by Piet Hein?")])



chat = ChatOpenAI(
    model="google/gemma-3-4b",
    base_url="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    max_tokens=3000,
    temperature=0.7,
)

response  = chat.invoke(state["messages"])

def chatbot(state:State)->State:
    print(f"\n ----> ENTERING chatbot:")
    response  = chat.invoke(state["messages"])
    return  State(messages= [response])


graph = StateGraph(State)

graph.add_node("chatbot",chatbot)
graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)

graph_compiled =  graph.compile()

print(graph_compiled.invoke(state))
