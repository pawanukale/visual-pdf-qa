import time
from src.logger import logger

def retrieve(question, index, engine, top_k=2):
    start = time.perf_counter()
    query_embedding = engine.embed_query(question)
    scores = engine.score(query_embedding, index["embeddings"])
    # score_multi_vector returns a [queries, documents] tensor.
    flat_scores = scores[0].float().cpu()
    k = min(max(1, int(top_k)), len(index["metadata"]))
    values, indices = flat_scores.topk(k)
    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        item = dict(index["metadata"][idx])
        item["score"] = float(score)
        results.append(item)
    elapsed = time.perf_counter() - start
    logger.info("RETRIEVAL_COMPLETE question=%r top_k=%s elapsed_seconds=%.3f results=%s", question, k, elapsed, [(r['filename'], r['page_number'], round(r['score'], 4)) for r in results])
    return results, elapsed
