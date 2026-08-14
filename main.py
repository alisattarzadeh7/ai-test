from typing import TypedDict, Sequence, Literal, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages, MessagesState



chat = ChatOpenAI(
    model="google/gemma-3-4b",
    base_url="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    max_tokens=3000,
    temperature=0.7,
)


def ask_question(state: MessagesState) -> MessagesState:
    print(f"\n ----> ENTERING ask_question:")
    for i in state["messages"]:
        i.pretty_print()
    question = "what is your question?"
    print(question)
    return MessagesState(messages=[HumanMessage(input())])



def chatbot(state:MessagesState)->MessagesState:
    print(f"\n ----> ENTERING chatbot:")
    response  = chat.invoke(state["messages"])
    print(response.pretty_print())
    return  MessagesState(messages= [response])


def ask_another_question(state: MessagesState) -> MessagesState:
    print(f"\n ----> ENTERING ask_another_question:")

    question = "Whould you like to ask another question?"
    print(question)
    return MessagesState(messages=[HumanMessage(input())])




def routing_function(state:MessagesState)-> Literal["trim_messages","__end__"]:
    print(f"\n ----> ENTERING routing_function:")
    if state["messages"][-1].content == "yes":
        return "trim_messages"
    else:
        return "__end__"


def trim_messages(state:MessagesState)->MessagesState:
    print(f"\n ----> ENTERING trim_messages:")
    remove_messages = [RemoveMessage(id = i.id) for i in state["messages"][:-5]]

    return MessagesState(messages=remove_messages)



graph = StateGraph(MessagesState)

graph.add_node("ask_question",ask_question)
graph.add_node("chatbot",chatbot)
graph.add_node("ask_another_question",ask_another_question)
graph.add_node("trim_messages",trim_messages)

graph.add_edge(START,"ask_question")
graph.add_edge("ask_question","chatbot")
graph.add_edge("chatbot","ask_another_question")
graph.add_conditional_edges(source="ask_another_question",path = routing_function)

graph.add_edge("trim_messages","ask_question")

graph_compiled = graph.compile()


# print(graph_compiled.get_graph().draw_ascii())


graph_compiled.invoke(MessagesState(messages=[]))




