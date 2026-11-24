# Test Plan

| ID | Test | Expected result |
|---|---|---|
| T01 | Render a 2-page PDF | Two readable page images; page numbers 1 and 2 |
| T02 | Index 10–15 pages | One embedding and one metadata record per page |
| T03 | Save and reload index | Same model identifier, vector count and metadata count |
| T04 | Ask a question answered on one page | Relevant page appears in top 2–3 retrieval results |
| T05 | Ask an unanswerable question | Exact fallback text returned |
| T06 | Ask about a chart/table | Answer matches visible values and includes correct page citation |
| T07 | Repeat a question | Saved embeddings are reused; no page re-indexing |
| T08 | Remove Gemini key | Clear configuration error; no secret printed to logs |
| T09 | Measure performance | Index duration, retrieval/generation latency and process RSS recorded |
| T10 | Inspect logs | Timestamped stages for render, embedding, retrieval and VLM |

Record test date, machine specs, PDF names/page counts, observed results and screenshots before submission.
