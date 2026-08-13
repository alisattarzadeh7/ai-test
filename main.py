from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter


loader = Docx2txtLoader('Introduction_to_Data_and_Data_Science_2.docx')

pages = loader.load()


md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on= [("#","Course Title"),("##","Lecture title")])

pages_md_split = md_splitter.split_text(pages[0].page_content)




for i in range(len(pages_md_split)):
    pages_md_split[i].page_content = " ".join(pages_md_split[i].page_content.split())


char_splitter = CharacterTextSplitter(separator=".", chunk_size= 500, chunk_overlap=0)

page_char_split = char_splitter.split_documents(pages_md_split)

text = page_char_split[3].page_content

embeddings = OpenAIEmbeddings(
    model="text-embedding-mxbai-embed-large-v1",
    base_url="http://192.168.56.1:1234/v1",
    api_key="lm-studio",
    check_embedding_ctx_length=False,  # Important for LM Studio
)

vector_store = Chroma.from_documents(
    documents=page_char_split,
    embedding=embeddings,
    collection_name="data_science_course",
    persist_directory="./chroma_db",
)

