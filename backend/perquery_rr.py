"""
Per-query reciprocal rank for BM25 and Hybrid-RRF on multi-hop queries.
Prints raw RR values so the caller can run a bootstrap significance test.
"""
import json, sys, time
import numpy as np

sys.path.append("/Users/krishsapru/aethel-clean")
from backend.public_benchmark import SimpleDocument, _SparseRetriever
from backend.evaluate_aethel import _NERv3Retriever, rrf_fuse, _RRF_POOL

CHUNKS  = "/Users/krishsapru/aethel-clean/backend/data/processed_chunks.json"
QUERIES = "/Users/krishsapru/aethel-clean/backend/data/eval_queries_gold.json"


def rr(retrieved_cids, gold_ids):
    for rank, cid in enumerate(retrieved_cids):
        if cid in gold_ids:
            return 1.0 / (rank + 1)
    return 0.0


def main():
    print("Loading data...")
    with open(CHUNKS)  as f: chunks  = json.load(f)
    with open(QUERIES) as f: queries = json.load(f)
    multi_hop = [q for q in queries if q["query_type"] == "multi-hop"]
    print(f"  {len(chunks)} chunks, {len(multi_hop)} multi-hop queries\n")

    docs = [SimpleDocument(page_content=c["content"],
                           metadata={"chunk_id": c["chunk_id"]})
            for c in chunks]

    print("Init BM25...")
    bm25 = _SparseRetriever(docs)
    print("Init NERv3 (takes ~2 min cold)...")
    ner3 = _NERv3Retriever(docs)
    print()

    print(f"{'Q':<5} {'BM25_RR':>8} {'RRF_RR':>8}  {'delta':>8}  query")
    print("-" * 95)

    bm25_rrs, rrf_rrs = [], []
    for qi, q in enumerate(multi_hop, start=21):
        qtxt  = q["query"]
        golds = set(q["gold_chunk_ids"])

        bm25_full  = bm25.query(qtxt, k=_RRF_POOL)
        ner3_full  = ner3.query(qtxt, k=_RRF_POOL)
        fused_idxs = rrf_fuse([bm25_full, ner3_full], top_k=10)

        bm25_cids  = [docs[i].metadata["chunk_id"] for i in bm25_full[:10]]
        fused_cids = [docs[i].metadata["chunk_id"] for i in fused_idxs]

        rr_b = rr(bm25_cids, golds)
        rr_r = rr(fused_cids, golds)
        d    = rr_r - rr_b
        sign = "+" if d > 0 else ("=" if d == 0 else "")

        bm25_rrs.append(rr_b)
        rrf_rrs.append(rr_r)

        print(f"Q{qi:<4} {rr_b:>8.4f} {rr_r:>8.4f}  {sign}{d:>7.4f}  {qtxt[:55]}")

    print("-" * 95)
    mb, mr = np.mean(bm25_rrs), np.mean(rrf_rrs)
    print(f"{'MRR':<5} {mb:>8.4f} {mr:>8.4f}  {mr-mb:>+8.4f}")
    print()
    print("BM25 per-query RRs:", [round(x, 4) for x in bm25_rrs])
    print("RRF  per-query RRs:", [round(x, 4) for x in rrf_rrs])
    print("Deltas (RRF-BM25): ", [round(r-b, 4) for b, r in zip(bm25_rrs, rrf_rrs)])


if __name__ == "__main__":
    main()
