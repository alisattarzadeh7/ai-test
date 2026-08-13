from langchain_community.document_loaders import PyPDFLoader
import copy


loader_pdf = PyPDFLoader('Introduction_to_Data_and_Data_Science.pdf')

pages_pdf = loader_pdf.load()

pages_pdf_cut = copy.deepcopy(pages_pdf)


for i in pages_pdf_cut:
    i.page_content = ''.join(pages_pdf_cut[i].page_content.split())

print(pages_pdf_cut)