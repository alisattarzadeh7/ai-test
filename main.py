from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-mxbai-embed-large-v1",
    base_url="http://192.168.56.1:1234/v1",
    api_key="lm-studio",
    check_embedding_ctx_length=False,  # Important for LM Studio
)

vectorstore = Chroma(collection_name="data_science_course",persist_directory='./chroma_db',embedding_function=embeddings)


question  = "What programming languages do data scientists use?"
retrieved_docs = vectorstore.similarity_search(query=question,k = 5)

print(retrieved_docs)

