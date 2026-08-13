from langchain_community.document_loaders import Docx2txtLoader
import copy

from langchain_text_splitters import CharacterTextSplitter

loader = Docx2txtLoader('Introduction_to_Data_and_Data_Science.docx')

pages = loader.load()

print(len(pages[0].page_content))


for i in range(len(pages)):
    pages[i].page_content = ''.join(pages[i].page_content.split())

print(len(pages[0].page_content))

char_splitter = CharacterTextSplitter(separator=".", chunk_size= 500, chunk_overlap=0)

page_char_split = char_splitter.split_documents(pages)

print(len(page_char_split))