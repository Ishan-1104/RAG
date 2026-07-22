from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vector_store.similarity_search("What is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    print()

retriever = vector_store.as_retriever()

docs = retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)