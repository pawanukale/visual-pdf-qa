# Visual PDF Question Answering (AIT Global POC)

A local-first Visual RAG proof of concept. PDF pages are rendered as images, indexed with ColPali/ColQwen2 multi-vector embeddings, retrieved with late-interaction MaxSim scoring, and passed (top 2–3 pages only) to Gemini for grounded answers with page citations.

## Architecture selected: Option A — Hybrid Local

- **Local:** PDF rendering (PyMuPDF), visual embeddings (ColPali Engine), persistent index, MaxSim retrieval, Streamlit UI, logs and RAM measurements.
- **Cloud:** Gemini receives only the user question and top retrieved page images for answer generation.
- **No OCR or text chunking:** every page is handled as an image.

## Requirements

- Windows 10/11, Linux, or macOS; 64-bit Python 3.10 or 3.11 recommended.
- RAM: 16 GB or more is recommended. This package is configured conservatively for an 8 GB experiment (100 DPI, batch size 1), but the ColQwen2 checkpoint may still exceed available RAM on some 8 GB laptops. A successful full model load is not guaranteed; verify before committing to the CPU-only path.
- Internet for the first model download and Gemini API calls.
- Gemini API key.

## Setup (Windows PowerShell)

```powershell
cd visual-pdf-qa
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CPU PyTorch build first if this machine has no NVIDIA GPU:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Create the environment file and add your key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `GEMINI_API_KEY` to your own key. Never commit `.env` to GitHub.

Run the application:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). Upload 3–4 PDFs (about 10–15 pages total), click **Index uploaded PDFs**, then ask questions. Three synthetic demo PDFs (10 pages total, with tables and charts) are included in `sample_documents/` for an initial smoke test. On an 8 GB machine, begin with one small PDF (2–3 pages), close memory-heavy applications, keep DPI at 100 and batch size at 1, and monitor RAM before indexing all 10–15 pages.

## How the pipeline works

1. `pdf_processor.py` renders each PDF page at 100–150 DPI using PyMuPDF. It does not extract text or run OCR.
2. `colpali_engine.py` loads `vidore/colqwen2-v1.0` on CPU and generates page multi-vectors.
3. `embedding_store.py` persists vectors in `.pt` and page metadata in JSON. Existing vectors are reused between questions.
4. `retriever.py` embeds the question and uses the processor's multi-vector scoring (late-interaction MaxSim) to rank pages.
5. `vlm_generator.py` sends the top retrieved page images and question to Gemini. The prompt requires citations for every factual claim and the exact not-found response if unsupported.
6. `performance_monitor.py` measures resident memory; application logs record stage timestamps and durations.

## Performance reporting

The app reports retrieval latency, generation latency, current RAM and indexing time. For the final benchmark, record the indexing time, total pages indexed, query latency, and peak observed RAM on the actual ThinkPad. Do not use invented or estimated values in `ARCHITECTURE.md`. The current RAM reading is process RSS; it is a sampled observed value, not a guaranteed operating-system peak.

## Project layout

```text
app.py
src/
  config.py
  pdf_processor.py
  colpali_engine.py
  embedding_store.py
  retriever.py
  vlm_generator.py
  performance_monitor.py
  logger.py
data/{pdfs,page_images,embeddings}/
logs/
ARCHITECTURE.md
AI_PROMPT_LOG.md
requirements.txt
```

## Troubleshooting

- **Model download / Hugging Face access (`Background writer channel closed`):** The base model (`vidore/colqwen2-base`) requires ~8.8 GB of disk space. If your primary `C:` drive has limited free space, the Hugging Face `hf-xet` background writer will fail. Set `HF_HOME=D:\huggingface_cache` and `HF_HUB_DISABLE_XET=1` in your `.env` to redirect the cache to a drive with ample space (e.g., `D:`) and use standard reliable HTTP streaming downloads.
- **RAM exhaustion:** close other applications, keep `BATCH_SIZE=1`, set DPI to 100, and begin with 2–3 pages. If the model itself cannot load, reducing page count will not solve the model-weight memory requirement; use a machine with more RAM or obtain approval for a smaller retrieval model before changing the assignment architecture. Restart the app after a failed model load.
- **Gemini authentication:** confirm the API key and model name in `.env`; restart Streamlit after changing `.env`.
- **Changed model:** if `COLPALI_MODEL` changes, delete the old files under `data/embeddings/` and re-index.
- **Model/API changes:** model identifiers and SDK behavior can change; verify the configured Gemini model is enabled for your API key.

## Important limitations to validate before the review

This is a POC baseline, not a production document platform. Run it on the target laptop, validate the installed ColPali package/model combination, verify Gemini model availability, and test answer grounding with known questions and deliberately unanswerable questions. Keep original PDFs private; top retrieved page images are transmitted to the configured Gemini API.
