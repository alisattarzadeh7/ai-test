import os
import sys

os.environ["USER_AGENT"] = "my-langchain-app"
sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import WebBaseLoader
from litellm import completion
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 1. Choose the webpage containing your custom data.
URL = "https://vakilnoor.ir/posts"

# 2. Load the webpage as a LangChain document.
raw_documents = WebBaseLoader(URL).load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5",
    base_url="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    check_embedding_ctx_length=False,
)

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
)


question = "what is latest post of vakilnoor website?"


results = vector_store.similarity_search(question, k=3)

webpage_text = "\n\n".join(
    document.page_content for document in results
)
# 3. Ask a question about that data.

# 4. Give the webpage text and question to the model.
response = completion(
    model="openai/google/gemma-3-4b",
    api_base="http://192.168.244.67:1234/v1",
    api_key="lm-studio",
    messages=[
        {
            "role": "system",
            "content": "Answer only from the provided webpage content.",
        },
        {
            "role": "user",
            "content": f"Webpage content:\n{webpage_text}\n\nQuestion: {question}",
        },
    ],
    max_tokens=256,
    timeout=60,
)

# 5. Print the answer.
print(response.choices[0].message.content)
