import os
import re
import tempfile
import unicodedata
import shutil
import gc

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI


load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader(
    "Upload a PDF book",
    type="pdf"
)

# -------------------------------------------------------
# CREATE VECTOR DATABASE
# -------------------------------------------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Reading PDF..."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            st.write(f"Loaded Pages : {len(docs)}")
            st.write(f"Chunks Created : {len(chunks)}")

            # ------------------------------------------
            # CLEAN CHUNKS
            # ------------------------------------------

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

            st.write(f"Valid Chunks : {len(clean_chunks)}")

            # ------------------------------------------
            # EMBEDDING MODEL
            # ------------------------------------------

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # ------------------------------------------
            # VERIFY CHUNKS
            # ------------------------------------------

            good_chunks = []

            progress = st.progress(0)

            for i, doc in enumerate(clean_chunks):

                try:
                    embeddings.embed_query(doc.page_content)
                    good_chunks.append(doc)

                except Exception as e:

                    st.error(f"Chunk {i} failed")

                    st.code(repr(doc.page_content[:500]))

                    st.exception(e)

                progress.progress((i + 1) / len(clean_chunks))

            st.write(f"Good Chunks : {len(good_chunks)}")

            # ------------------------------------------
            # DELETE OLD DATABASE
            # ------------------------------------------

            try:
                gc.collect()

                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db")

            except PermissionError:

                st.warning(
                    "Old Chroma database is currently in use.\n\n"
                    "Please stop Streamlit, delete the 'chroma_db' folder manually, "
                    "and run the app again."
                )

                st.stop()

            # ------------------------------------------
            # CREATE DATABASE
            # ------------------------------------------

            vectorstore = Chroma.from_documents(
                documents=good_chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            try:
                vectorstore.persist()
            except:
                # persist() is not required in newer versions
                pass

            st.success("✅ Vector Database Created Successfully!")

# -------------------------------------------------------
# LOAD DATABASE
# -------------------------------------------------------

if os.path.exists("chroma_db"):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
reply exactly:

I could not find the answer in the document.
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()

    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        response = llm.invoke(final_prompt)

        st.write("### AI Answer")

        st.write(response.content)