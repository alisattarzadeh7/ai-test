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


chat_template_time = ChatPromptTemplate.from_template('''
    I'm an intermediate level programmer.
    Consider the following literature:
    {books}
    
    Also, consider the following projects:
    {projects}
    
    roughly how much time would it take me to complete the literature and the projects?.
''')


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

chain_time = (RunnableParallel({
    'books': chain_books,
    'projects': chain_projects,
}) | chat_template_time | llm | string_parser)

print(chain_time.invoke({
    'programming_language':'Python'
}))

chain_parrallel.get_graph().print_ascii()