from typing import TypedDict, Sequence, Literal, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, add_messages, MessagesState
import sqlite3
from pathlib import Path

chat = ChatOpenAI(
    model="google/gemma-3-4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    max_tokens=3000,
    temperature=0.7,
    timeout=120,
    max_retries=1,
)


class State(MessagesState):
    summary:str



def ask_question(state: State) -> State:
    print(f"\n ----> ENTERING ask_question:")
    for i in state["messages"]:
        i.pretty_print()
    question = "what is your question?"
    print(question)
    return State(messages=[HumanMessage(input())])



def chatbot(state:State)->State:
    print(f"\n ----> ENTERING chatbot:")

    system_message =f'''
        Here's a quick summary of what's been discussed so far:
        {state.get("summary","")}
        
        Keep this in mind as you answer the next question.
        '''

    response  = chat.invoke([SystemMessage(system_message)] + state["messages"])
    print(response.pretty_print())
    return  State(messages= [response])


def ask_another_question(state: State) -> State:
    print(f"\n ----> ENTERING ask_another_question:")

    question = "Whould you like to ask another question?"
    print(question)
    return State(messages=[HumanMessage(input())])




def routing_function(state:State)-> Literal["summarize_and_delete_messages","__end__"]:
    print(f"\n ----> ENTERING routing_function:")
    if state["messages"][-1].content == "yes":
        return "summarize_and_delete_messages"
    else:
        return "__end__"


def summarize_and_delete_messages(state:State)->State:
    print(f"\n ----> ENTERING trim_messages:")
    new_conversation = ""
    for i in state["messages"]:
        new_conversation += f"{i.type}: {i.content}\n\n"

    summary_instructions = f'''
        update the ongoing summary bt incorporating the new lines of conversation below.
        build upon the previous summary rather than repeating it so that the result reflects the most recent context and developments.
        
        Previous Summary:
        {state.get("summary","")}
        
        New Conversation:
        {new_conversation}
        
    '''

    print(summary_instructions)

    summary = chat.invoke([HumanMessage(summary_instructions)])

    remove_messages = [RemoveMessage(id = i.id) for i in state["messages"][:]]

    return State(messages=remove_messages, summary = summary.content)



graph = StateGraph(State)

graph.add_node("ask_question",ask_question)
graph.add_node("chatbot",chatbot)
graph.add_node("ask_another_question",ask_another_question)
graph.add_node("summarize_and_delete_messages",summarize_and_delete_messages)

graph.add_edge(START,"ask_question")
graph.add_edge("ask_question","chatbot")
graph.add_edge("chatbot","ask_another_question")
graph.add_conditional_edges(source="ask_another_question",path = routing_function)

graph.add_edge("summarize_and_delete_messages","ask_question")

db_path = Path(__file__).resolve().with_name("langgraph.db")
con = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(con)
graph_compiled = graph.compile(checkpointer=checkpointer)


print(graph_compiled.get_graph().draw_ascii())


config1 = {
    "configurable":{
        "thread_id": "1"
    }
}


graph_compiled.invoke(State(messages=[]),config1)



