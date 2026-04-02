import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def ingest_pdf(pdf_path: str, output_dir: str, chunk_size: int = 1000, chunk_overlap: int = 400):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    os.makedirs(output_dir, exist_ok=True)
    vector_store.save_local(output_dir)
    return vector_store


