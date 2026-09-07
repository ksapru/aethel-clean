"""
Per-query reciprocal rank for BM25 and Hybrid-RRF on multi-hop queries,
plus the paired bootstrap CI on their MRR difference.

The bootstrap was previously run outside the repository, which left the
confidence interval quoted in the paper unreproducible. It is implemented
here so the interval regenerates deterministically from the same per-query
data, and is written to backend/data/multihop_rr_bootstrap.json.
"""
import json, sys, time
import numpy as np

OUT = "/Users/krishsapru/aethel-clean/backend/data/multihop_rr_bootstrap.json"
N_BOOT = 10000
SEED = 42

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


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over queries of mean(b) - mean(a).

    Both systems are scored on the SAME resampled query set in each iteration,
    which is what makes the interval paired: it removes per-query difficulty
    as a source of variance. Returns the observed difference, the percentile
    95% CI, and a two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    observed = float(b.mean() - a.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided p: fraction of resamples on the far side of zero, doubled
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "observed": round(observed, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "p_two_sided": round(float(min(p, 1.0)), 4),
        "n_queries": int(n),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }


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
    print()

    bs = paired_bootstrap(bm25_rrs, rrf_rrs)
    print(f"Paired bootstrap (N={bs['n_queries']} queries, B={bs['n_boot']}, seed={bs['seed']}):")
    print(f"  observed MRR delta (RRF - BM25): {bs['observed']:+.4f}")
    print(f"  95% CI: [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]")
    print(f"  two-sided p: {bs['p_two_sided']:.4f}")
    print(f"  CI crosses zero: {bs['crosses_zero']}")

    payload = {
        "meta": {
            "description": "Paired bootstrap on the multi-hop MRR difference "
                           "between Hybrid-RRF and BM25 over the open-corpus "
                           "financial benchmark.",
            "produced_by": "backend/perquery_rr.py",
        },
        "mrr": {"bm25": round(float(mb), 4), "hybrid_rrf": round(float(mr), 4)},
        "per_query_rr": {
            "bm25": [round(x, 4) for x in bm25_rrs],
            "hybrid_rrf": [round(x, 4) for x in rrf_rrs],
        },
        "bootstrap": bs,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
