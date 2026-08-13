from typing import TypedDict, Sequence, Literal

from langchain_core.messages import BaseMessage, HumanMessage
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


def ask_question(state: State) -> State:
    print(f"\n ----> ENTERING ask_question:")

    print("what is your question?")
    return State(messages=[HumanMessage(input())])


def ask_another_question(state: State) -> State:
    print(f"\n ----> ENTERING ask_another_question:")

    print("Whould you like to ask another question?")
    return State(messages=[HumanMessage(input())])



def chatbot(state:State)->State:
    print(f"\n ----> ENTERING chatbot:")
    response  = chat.invoke(state["messages"])
    print(response.content)
    return  State(messages= [response])


def routing_function(state:State)-> Literal["ask_question","__end__"]:
    print(f"\n ----> ENTERING routing_function:")
    if state["messages"][0].content == "yes":
        return "ask_question"
    else:
        return "__end__"



graph = StateGraph(State)

graph.add_node("ask_question",ask_question)
graph.add_node("chatbot",chatbot)
graph.add_node("ask_another_question",ask_another_question)

graph.add_edge(START,"ask_question")
graph.add_edge("ask_question","chatbot")
graph.add_edge("chatbot","ask_another_question")
graph.add_conditional_edges(source="ask_another_question",path = routing_function,path_map= {
        "ask_question": "ask_question",
        "__end__": "__end__"
    })

graph_compiled = graph.compile()


# print(graph_compiled.get_graph().draw_ascii())


graph_compiled.invoke(State(messages=[]))




