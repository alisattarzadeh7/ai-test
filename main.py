import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")


# -------------------------
# Model
# -------------------------

llm = ChatOpenAI(
    model="google/gemma-3-4b",
    base_url="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    max_tokens=3000,
    temperature=0.7,
)


# -------------------------
# Prompts
# -------------------------

chat_template_tools = ChatPromptTemplate.from_template("""
What are the five most important tools a {job_title} needs?

Answer only by listing the tools.
""")


chat_template_strategy = ChatPromptTemplate.from_template("""
Considering the tools provided, develop a strategy for effectively
learning and mastering them:

{tools}
""")


# -------------------------
# Chains
# -------------------------

tools_chain = chat_template_tools | llm | StrOutputParser() | {'tools': RunnablePassthrough()}

strategy_chain = (
    chat_template_strategy
    | llm
    | StrOutputParser()
)


# -------------------------
# Run first chain
# -------------------------

chain_combined = tools_chain | strategy_chain



# -------------------------
# Feed result into second
# -------------------------

strategy = chain_combined.invoke({'job_title': 'frontend developer'})



print(strategy)

chain_combined.get_graph().print_ascii()