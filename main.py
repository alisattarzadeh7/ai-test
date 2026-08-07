import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
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

chat_template_books = ChatPromptTemplate.from_template("""
suggest three of the best intermediate-level {programming_language} books.
answer only by listing the books.
""")


chat_template_projects = ChatPromptTemplate.from_template("""
suggest three interesting {programming_language} projects suitable for intermediate-level programmers.
answer only by listing the projects.
""")



string_parser = StrOutputParser()

chain_books = chat_template_books | llm | string_parser

chain_projects = chat_template_projects | llm | string_parser

chain_parrallel = RunnableParallel({
    'books': chain_books,
    'projects': chain_projects,
})

output = chain_parrallel.invoke({
    'programming_language':'Python'
})
print(output)
chain_parrallel.get_graph().print_ascii()