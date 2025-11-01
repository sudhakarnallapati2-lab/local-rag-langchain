# Local RAG LangChain Project (Modular, Streamlit UI, Multi-file awareness)

## What you get
- `ingest.py` — ingest documents from `data/` and build FAISS index
- `qa_engine.py` — simple terminal QA using the FAISS index
- `app.py` — Streamlit app: upload PDFs, build index, conversational RAG, chat history, download, **multi-file selection**
- `requirements.txt` — Python dependencies
- `.env.example` — example env file for OpenAI key (optional)
- `data/` — folder where uploads are saved at runtime
- `vectorstore/` — FAISS index is written here at runtime

## Quick start (OpenAI mode)
1. Create and activate a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` -> `.env` and add your `OPENAI_API_KEY` (if using OpenAI):
   ```bash
   cp .env.example .env
   # edit .env and set OPENAI_API_KEY=sk-...
   ```

4. Run Streamlit UI (recommended):
   ```bash
   streamlit run app.py
   ```

5. Upload PDFs, select which files to include for a query (multi-file awareness), ask questions, download chat history.

## Offline (Ollama) mode
See comments in `app.py` for how to switch to Ollama (fully offline). You'll need Ollama installed and local models pulled.

## Notes on multi-file awareness
The Streamlit app allows selecting a subset of uploaded/ingested files to restrict the retrieval scope. For small collections the app recreates a temporary FAISS index from the selected documents to ensure accurate, file-scoped retrieval.

## Troubleshooting
- If FAISS fails to load, delete `vectorstore/` and re-run ingestion via the UI or `python ingest.py`
- For large PDFs consider increasing chunk size or running ingestion on a machine with enough memory/CPU.

Enjoy — open an issue or ask for features in the chat!


## Extras included in this ZIP

- `Dockerfile` and `docker-compose.yml` to run the Streamlit app in a container.
- `app.py` now defaults to Ollama (offline) if `OLLAMA=true` is set or if `USE_OLLAMA` env var exists; otherwise it uses OpenAI.
- `.github/workflows/ci.yml` — basic CI to run `pip install -r requirements.txt` and run tests.
- `tests/test_app_smoke.py` — a minimal pytest file to ensure imports run.

