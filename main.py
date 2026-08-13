from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-mxbai-embed-large-v1",
    base_url="http://192.168.56.1:1234/v1",
    api_key="lm-studio",
    check_embedding_ctx_length=False,  # Important for LM Studio
)

vectorstore = Chroma(collection_name="data_science_course",persist_directory='./chroma_db',embedding_function=embeddings)


retriever = vectorstore.as_retriever(search_type='mmr',search_kwargs={'k':3,'lambda_mult':0.7})


TEMPLATE = '''
Answer the question:
{question}

To answer the question, use only the following context:
{context}

At the end of the response, specify the name of the lecture this context is taken from in the format:
Resources: *Lecture Title*
where *Lecture Title* should be substituted with the title of all resource lectures.
'''


prompt_template = PromptTemplate.from_template(TEMPLATE)

chat = ChatOpenAI(
    model="google/gemma-3-4b",
    base_url="http://192.168.56.1:1234/v1",
    api_key="lm-studio",
    max_tokens=3000,
    temperature=0.7,
)


question  = "What software do data scientists use?"

chain = {'context':retriever,'question': RunnablePassthrough()} | prompt_template | chat | StrOutputParser()

result = chain.invoke(question)

print(result)


