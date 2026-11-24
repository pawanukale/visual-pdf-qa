# Architecture — Visual PDF Question Answering

## Objective

Build a local command-line or lightweight web application that answers questions about PDFs containing tables, charts and scanned layouts. Each page is treated as an image and indexed using ColPali late-interaction multi-vector embeddings. No traditional OCR or text chunking is used. A vision-language model generates answers grounded in the retrieved page images.

## Architecture decision: Option A — Hybrid Local

PDF rendering, ColPali embedding, persistent storage and MaxSim retrieval run locally on CPU. Only the top two or three retrieved page images and the user's question are sent to Gemini for answer generation. This follows the recommended hybrid architecture in the assignment and avoids running both retrieval and generation models locally on a memory-constrained ThinkPad. It does require network access and an API key, and selected page images leave the local machine.

## Retrieval and generation flow

1. PyMuPDF renders 3–4 input PDFs (target 10–15 pages total) at configurable 100–150 DPI.
2. ColPali/ColQwen2 creates multi-vector embeddings for every page image.
3. Page vectors (`.pt`) and metadata (filename, page number, image path) are persisted on disk and reused for later queries.
4. The query is embedded; late-interaction MaxSim scores every stored page. The top 2–3 pages and scores are returned.
5. The top page images and question are passed to Gemini. The instruction requires each factual claim to cite `[Filename, Page X]`. If the answer is not visible in the retrieved pages, the exact response must be `Not found in the provided documents.`
6. Logs capture stage timestamps, indexing duration, retrieval latency, generation latency and process RSS memory.

## Why visual retrieval instead of standard text RAG?

Standard text RAG commonly extracts text, splits it into chunks and embeds those chunks. That can lose spatial relationships and make charts, tables, scanned pages and complex layouts difficult to represent. ColPali treats the whole page as an image and represents visual/textual page content with multi-vector embeddings, allowing late-interaction retrieval against the query without building an OCR-and-chunking pipeline.

## Storage and RAM trade-offs

The persistent `.pt` index avoids re-embedding pages for every question, reducing repeated indexing work at the cost of disk space proportional to the number and size of page vectors. CPU inference reduces GPU dependency but is slower and can consume substantial RAM. Batch size 1 and 100 DPI are the conservative settings for this 8 GB ThinkPad. These settings reduce page-processing memory but do not guarantee that the ColQwen2 model weights and runtime will fit in 8 GB; this must be tested on the target laptop. The app records observed process RSS; final peak RAM must be measured on the target laptop.

## Benchmark results (fill after running on the ThinkPad)

| Metric | Measured result |
|---|---|
| Hardware / RAM | ThinkPad, 8 GB RAM (user-provided; CPU/GPU details to record) |
| PDFs / pages indexed | **TO MEASURE** |
| Render DPI / batch size | **TO RECORD** |
| Indexing time | **TO MEASURE** |
| Average query retrieval latency | **TO MEASURE** |
| Average end-to-end query latency | **TO MEASURE** |
| Peak observed process RAM | **TO MEASURE** |

No benchmark values are fabricated. Record at least three representative queries and report the mean, noting whether model warm-up is included.

## Grounding and known limitations

The prompt enforces source-only answers and page citations, but prompting alone cannot mathematically guarantee zero hallucinations. Test answerable and unanswerable questions, manually inspect citations, and treat the VLM output as a POC result requiring verification. Only retrieved pages are provided to the VLM, so a relevant page missed by retrieval cannot support the answer. The app sends retrieved page images to Gemini; do not use confidential documents without authorization.
