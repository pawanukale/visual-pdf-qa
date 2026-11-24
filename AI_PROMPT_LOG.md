# AI Coding Assistant Prompt Log

> This file starts the required prompt log. Keep the complete, verbatim prompts used with any AI coding assistant during the actual implementation and debugging. The entries below document the baseline project-generation request used to create this starter. Add every later prompt and response summary during development; do not claim that prompts not actually used were run.

## Baseline project-generation prompt (key prompt 1)

"Read the AIT Global AI Proof of Concept brief for Visual PDF Question Answering. Build a modular Python Streamlit application using Option A Hybrid Local: PyMuPDF page rendering, ColPali/ColQwen2 multi-vector page embeddings on CPU, persistent `.pt` embeddings and JSON metadata, MaxSim retrieval of top 2–3 pages, and Gemini vision-language answer generation using only retrieved page images. Do not use OCR or text chunking. Include citations `[Filename, Page X]` for every factual claim and the exact fallback `Not found in the provided documents.`. Add timestamped logs, indexing/query latency and psutil RAM measurements, requirements, setup instructions, architecture documentation and prompt log. Keep API keys in environment variables."

## Key prompt 2 — to use only if needed

"Review the current ColPali integration against the installed package version and official model documentation. Identify API mismatches, correct the implementation, and explain the exact reason for each change. Do not replace visual multi-vector retrieval with text embeddings, OCR or a different retrieval architecture."

## Key prompt 3 — to use only if needed

"Test the PDF ingestion and persistent index pipeline with a small sample PDF. Check page count, rendered image readability, embedding count, metadata alignment, saved index reload, and model identifier validation. Fix any reproducible defects and report the test evidence."

## Key prompt 4 — to use only if needed

"Audit MaxSim retrieval: verify the query and page embeddings use the same model and compatible tensor format, confirm score ordering, return the top 2–3 pages with filenames/page numbers/scores, and add a deterministic unit test using mocked embeddings."

## Key prompt 5 — to use only if needed

"Evaluate answer grounding with questions whose answers are visible in retrieved pages and questions whose answers are absent. Ensure each factual statement cites the supplied filename and page, and unsupported questions return exactly `Not found in the provided documents.`. Record failures and fixes."

## Actual development and debugging log

Append entries in chronological order using this format:

- Date/time:
- Exact prompt:
- Tool/model used:
- What it changed or suggested:
- What I verified manually:

## AI coding assistant review (complete after observing a real issue)

- Mistake/hallucination observed: **TO COMPLETE FROM AN ACTUAL DEVELOPMENT EVENT**
- Why it was wrong:
- How it was detected:
- Correction made:
- How the correction was tested:
