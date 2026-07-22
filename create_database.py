import re
import unicodedata

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

loader = PyPDFLoader("document loaders/deeplearning.pdf")
docs = loader.load()

print(f"Loaded {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")

clean_chunks = []

for chunk in chunks:

    text = chunk.page_content

    if text is None:
        continue

    text = str(text)

    # Remove invalid UTF-16 surrogate characters
    text = re.sub(r"[\ud800-\udfff]", "", text)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove invalid UTF-8 bytes
    text = text.encode(
        "utf-8",
        "ignore"
    ).decode(
        "utf-8",
        "ignore"
    )

    text = text.strip()

    if not text:
        continue

    clean_chunks.append(
        Document(
            page_content=text,
            metadata=chunk.metadata
        )
    )

print(f"Valid chunks: {len(clean_chunks)}")

# -------------------------------------------------
# Embedding model
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded.")

# -------------------------------------------------
# Verify every chunk
# -------------------------------------------------

print("\nChecking every chunk...")

good_chunks = []

for i, doc in enumerate(clean_chunks):

    text = doc.page_content

    try:
        embedding_model.embed_query(text)

        print(f"✅ Chunk {i} OK")

        good_chunks.append(doc)

    except Exception as e:

        print(f"\n❌ Chunk {i} FAILED")

        print("Length :", len(text))
        print(repr(text[:500]))
        print(e)

print(f"\nGood chunks : {len(good_chunks)}")
print(f"Bad chunks  : {len(clean_chunks)-len(good_chunks)}")

# -------------------------------------------------
# Create Chroma DB
# -------------------------------------------------

print("\nCreating Chroma Database...")

vectorstore = Chroma.from_documents(
    documents=good_chunks,
    embedding=embedding_model,
    persist_directory="Chroma_db"
)

print("\n✅ Chroma Database Created Successfully!")