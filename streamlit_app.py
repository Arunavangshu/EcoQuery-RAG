import os
from pathlib import Path
import re

import streamlit as st
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from src.ecoquery.retriever import load_retriever, query_retriever

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"

PROMPT_TEMPLATE = """You are a helpful assistant. Use only the information provided in the context to answer the user's question.
Use percent symbols (%) instead of writing out the word percent when reporting percentages.
Always include source page citations in the answer in brackets, like [Page 12], when the information comes from the context.
If the answer cannot be found in the context, respond with: "I don't know based on the provided context."
If the user's question is a follow-up, use the conversation history to understand what the user is referring to.

Conversation history:
{history}

Context:
{context}

Question: {question}

Answer:"""


@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@st.cache_resource
def get_retriever(vector_store_path: str):
    return load_retriever(vector_store_path)


def build_context(docs):
    pieces = []
    for idx, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        page_number = doc.metadata.get("page")
        page_label = f"Page {page_number + 1}" if page_number is not None else f"Document {idx}"
        pieces.append(f"[{page_label}]\n{content}")
    return "\n\n---\n\n".join(pieces)


def build_history(history):
    if not history:
        return "No previous conversation."
    return "\n".join(
        [f"User: {user}\nAssistant: {assistant}" for user, assistant in history]
    )


def normalize_percent_answer(text: str) -> str:
    return re.sub(r"(\d+)\s+percent", r"\1%", text, flags=re.IGNORECASE)


def generate_answer(question: str, docs, history, model: str):
    context = build_context(docs)
    prompt = PROMPT_TEMPLATE.format(
        history=build_history(history),
        context=context,
        question=question,
    )
    client = OpenAI()
    response = client.responses.create(model=model, input=prompt)

    if hasattr(response, "output_text") and response.output_text:
        return normalize_percent_answer(response.output_text.strip())

    if getattr(response, "output", None):
        output_blocks = response.output
        if isinstance(output_blocks, list) and len(output_blocks) > 0:
            first_block = output_blocks[0]
            if isinstance(first_block, dict) and "content" in first_block:
                content_items = first_block["content"]
                if isinstance(content_items, list) and len(content_items) > 0:
                    first_content = content_items[0]
                    if isinstance(first_content, dict) and "text" in first_content:
                        return normalize_percent_answer(first_content["text"].strip())
                    return normalize_percent_answer(str(first_content).strip())
    return normalize_percent_answer(str(response))


def format_document_card(doc, index):
    page_number = doc.metadata.get("page")
    page_label = f"Page {page_number + 1}" if page_number is not None else f"Document {index}"
    snippet = doc.page_content.strip().replace("\n", " ")
    return page_label, snippet[:500]


def setup_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = []


def main():
    config = load_config()
    retriever = get_retriever(config["vector_store_path"])
    model_name = config.get("llm_model", "gpt-4o-mini")

    st.set_page_config(
        page_title="EcoQuery-RAG",
        page_icon="🍏",
        layout="wide",
    )

    setup_session_state()

    st.markdown("# 🍏 EcoQuery-RAG Streamlit UI")
    st.markdown(
        "A beautiful interactive interface for querying Apple’s Environmental Progress Report with the existing EcoQuery RAG engine."
    )

    with st.sidebar:
        st.header("Quick Controls")
        st.write("Use the form below to ask any question about Apple’s environmental report.")
        st.markdown("---")
        st.subheader("Sample questions")
        sample_questions = [
            "What percentage of the materials in Apple products came from recycled sources last year?",
            "Which product is Apple's first to be made with more than 50 percent recycled material?",
            "How much can ocean freight reduce emissions compared to air transport?",
            "What is Apple's target for recycled cobalt in batteries and what is the deadline?",
            "How many metric tons of waste did Apple’s suppliers divert from landfills in 2023?",
        ]
        for question in sample_questions:
            if st.button(question, key=question):
                st.session_state.last_question = question
        st.markdown("---")
        st.write(f"**Model:** {model_name}")
        st.write("**Vector store:** data/processed/vector_store.faiss")
        st.write("**Source PDF:** data/raw/Apple_Environmental_Progress_Report_2024.pdf")
        st.markdown("---")
        st.markdown("Built with Streamlit and OpenAI." )

    with st.form(key="query_form"):
        question = st.text_area("Ask a question about Apple's environmental report:",
                                 value=st.session_state.get("last_question", ""), height=130)
        submit = st.form_submit_button("Submit")

    if submit and question.strip():
        with st.spinner("Retrieving context and generating EcoQuery answer..."):
            docs = query_retriever(retriever, question)
            answer = generate_answer(question, docs, st.session_state.history, model_name)
            st.session_state.history.append((question, answer))
            st.session_state.conversation.append({
                "question": question,
                "answer": answer,
                "docs": docs,
            })

    if st.session_state.conversation:
        last = st.session_state.conversation[-1]
        st.markdown("## EcoQuery Answer")
        st.markdown(f"> {last['answer']}" )

        st.markdown("---")
        st.markdown("### Retrieved context")
        for idx, doc in enumerate(last["docs"], 1):
            page_label, snippet = format_document_card(doc, idx)
            st.markdown(f"**{page_label}**")
            st.write(snippet)
            st.markdown("---")

    if st.session_state.history:
        with st.expander("Conversation history"):
            for item in st.session_state.history[::-1]:
                st.markdown(f"**Q:** {item[0]}")
                st.markdown(f"**EcoQuery Answer:** {item[1]}")
                st.markdown("---")

    st.markdown("---")
    st.markdown("Powered by EcoQuery-RAG and Streamlit.")


if __name__ == "__main__":
    main()
