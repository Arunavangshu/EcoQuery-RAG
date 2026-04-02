import os
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def load_retriever(vector_store_path: str, k: int = 5):
    store_path = Path(vector_store_path)
    if store_path.suffix == ".faiss" or store_path.is_file():
        store_path = store_path.parent

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    )

    return FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    ).as_retriever(search_kwargs={"k": k})


def query_retriever(retriever, query: str):
    return retriever.get_relevant_documents(query)
