# 🍏 EcoQuery-RAG

> Turning Apple’s 100+ page Environmental Progress Report into an actionable, queryable intelligence engine.

EcoQuery-RAG is a Retrieval-Augmented Generation project that loads Apple’s 2024 environmental report from PDF, builds a local FAISS vector store, and answers business and ESG questions using OpenAI.

## Architecture

- **Orchestration:** Custom Python pipeline using LangChain patterns
- **LLM:** OpenAI GPT-4o-mini (configurable in `configs/config.yaml`)
- **Vector Database:** FAISS via `langchain_community`
- **Embedding Model:** OpenAI `text-embedding-3-large`
- **Chunking Strategy:** Recursive character splitting with `chunk_size=1000` and `chunk_overlap=200`
- **Storage:** Source PDF in `data/raw/`, vector store saved to `data/processed/`

## Project structure

- `data/raw/` - source documents and original files
- `data/processed/` - parsed chunks, embeddings, and vector store data
- `src/ecoquery/` - ingestion, retrieval, and query modules
- `scripts/` - runnable ingestion/query scripts
- `tests/` - unit and integration test scaffolds
- `configs/` - configuration settings for models and file paths
- `notebooks/` - exploratory and analysis notebooks

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key stored in `.env`

### Installation

```bash
cd EcoQuery-RAG
python -m pip install -r requirements.txt
cp .env.example .env
# then set OPENAI_API_KEY in .env
```

### Ingest the PDF

```bash
python scripts/ingest_pdf.py
```

### Start the interactive query session

```bash
python scripts/query_pdf.py
```

Type questions directly into the prompt. Enter `quit` or `exit` to close the conversation.

The system formats percentages with the `%` symbol and includes page citations where available.

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

This launches a polished web interface where you can ask questions, see retrieved document snippets, and get an `EcoQuery Answer:` response with citation context.

## 📊 Sample Output

**Query:** What percentage of the materials in Apple products came from recycled sources last year?

**EcoQuery Answer:** More than 20% of the materials shipped in Apple products came from recycled sources last year [Page 3].

**Source Attribution:** `Apple_Environmental_Progress_Report_2024.pdf` (recycled materials and sourcing sections)

## Tested questions

The RAG system has been tested with the following sample questions:

- What percentage of the materials in Apple products came from recycled sources last year?
- Which product is Apple's first to be made with more than 50 percent recycled material?
- How much can ocean freight reduce emissions compared to air transport?
- What is Apple’s specific target for using recycled cobalt in its batteries, and what is the deadline for this goal?
- What percentage of recycled tungsten, rare earth elements, and aluminum did Apple use in its products in the most recent reporting period?
- How much did Apple reduce its overall greenhouse gas emissions across Scopes 1, 2, and 3 compared to its 2015 baseline?
- How many metric tons of waste did Apple’s suppliers divert from landfills in 2023, and how many facilities are participating in the Zero Waste program?
- What is the percentage of recycled polymers used in the Vision Pro according to the 2024 report?

## 🛠️ Key Challenges & Solutions

- **Table parsing:** Apple’s report contains complex ESG tables and product specifications. The project uses chunking and similarity search to preserve table-relevant information without losing structural meaning.
- **Context management:** Top chunks are retrieved and passed into the prompt to avoid overwhelming the LLM, while preserving the most relevant numeric and product-specific details.
- **Conversation state:** The query script supports multi-turn context, allowing follow-up questions like product-specific comparisons and clarifications.
- **Secure local storage:** All data is kept locally in `data/processed/`, and the OpenAI key is loaded from `.env`.- `Citation support:` The query engine includes source page numbers from the PDF chunks in the returned context.
## Notes

- Place `Apple_Environmental_Progress_Report_2024.pdf` in `data/raw/`.
- Use `OPENAI_API_KEY` in `.env` for OpenAI embeddings and LLM access.
- The current proof-of-concept does not yet include formal page-level citation output but can be extended with source attribution rules.
