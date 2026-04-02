import sys
import yaml
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.ecoquery.retriever import load_retriever, query_retriever


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


def build_context(docs):
    pieces = []
    for idx, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        page_number = doc.metadata.get("page")
        if page_number is not None:
            page_label = f"Page {page_number + 1}"
        else:
            page_label = f"Document {idx}"
        pieces.append(f"[{page_label}]\n{content}")
    return "\n\n---\n\n".join(pieces)


def build_history(history):
    if not history:
        return "No previous conversation."
    return "\n".join(
        [f"User: {user}\nAssistant: {assistant}" for user, assistant in history]
    )


def generate_answer(question: str, docs, history, model: str):
    context = build_context(docs)
    prompt = PROMPT_TEMPLATE.format(
        history=build_history(history),
        context=context,
        question=question,
    )
    client = OpenAI()
    response = client.responses.create(model=model, input=prompt)

    def normalize_percent_answer(text: str) -> str:
        import re

        text = re.sub(r"(\d+)\s+percent", r"\1%", text, flags=re.IGNORECASE)
        return text

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


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    retriever = load_retriever(config["vector_store_path"])
    history = []

    print("Enter your query. Type 'quit' or 'exit' to end the session.")
    while True:
        question = input("Query: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Closing the conversation. Goodbye!")
            break

        docs = query_retriever(retriever, question)

        print("\nTop retrieved text chunks:\n")
        for i, doc in enumerate(docs, 1):
            page_number = doc.metadata.get("page")
            page_label = f"Page {page_number + 1}" if page_number is not None else f"Document {i}"
            print(f"[{i}] {page_label}: {doc.page_content[:600]}\n")

        print("Generating final answer from the LLM...\n")
        answer = generate_answer(
            question,
            docs,
            history,
            config.get("llm_model", "gpt-4o-mini"),
        )

        print("Final answer:\n")
        print(answer)
        print("\n---\n")

        history.append((question, answer))


if __name__ == "__main__":
    main()
