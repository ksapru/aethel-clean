# Aethel: Bipartite Graph-Walk Retrieval for Multi-Hop Financial Diligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

Aethel is a graph-augmented retrieval framework for multi-hop question answering over unstructured financial documents. It models document collections as bipartite entity–passage graphs and executes Personalized PageRank (PPR) random walks to resolve cross-document, multi-hop queries. An **alias-expanded seeding (AES)** layer expands entity mentions via alias matching and substring overlap, improving hit rate coverage at the cost of top-1 precision. Note this is surface-form matching, not coreference resolution: no mention-clustering or pronoun resolution is performed.


## Architecture

The framework has three components:

1. **Bipartite Entity–Passage Graph** — entities (`V_e`) and passages (`V_p`) are vertices in a bipartite graph `G = (V_p ∪ V_e, E)`. Query entities seed a PPR random walk over sparse adjacency paths to rank passages.

2. **Alias-Expanded Seeding (AES)** — expands the PPR personalization vector with alias matches and substring overlaps, seeding additional start nodes to improve hit rate coverage across entity variants (e.g., "Apollo" → "Apollo Global Management"). This is surface-form matching only, not coreference resolution.

3. **Orchestrated Specialist Swarm** — retrieved passages are forwarded to domain-specialist agents (Liquidity, Valuation, Diligence Auditor) coordinated by a central Orchestrator. The swarm is described in `backend/agents/` and `backend/main.py`; it is not quantitatively evaluated in the current paper.


## Empirical Results

All numbers below come from `backend/public_benchmark.py` running against the official HuggingFace validation splits. Results are cached in `eval_cache.json` (committed to the repository) for instant reproducibility.

**200-question random samples, seed=42, from official validation splits of MuSiQue and 2WikiMultiHopQA.**

| Method | 2Wiki HR@1 | 2Wiki HR@3 | 2Wiki HR@5 | 2Wiki MRR | MuSiQue HR@1 | MuSiQue HR@3 | MuSiQue HR@5 | MuSiQue MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Sparse (TF-IDF) | 0.790 | 0.965 | 0.990 | 0.873 | 0.580 | 0.780 | 0.845 | 0.681 |
| Dense (MiniLM, 22M) | 0.900 | 0.990 | 0.995 | 0.940 | 0.765 | 0.935 | 0.960 | 0.849 |
| Dense (E5-base, 109M) | 0.960 | **1.000** | **1.000** | 0.980 | 0.740 | 0.915 | 0.965 | 0.834 |
| Dense (BGE-base, 109M) | **0.985** | **1.000** | **1.000** | **0.993** | 0.795 | 0.925 | **0.985** | 0.865 |
| Dense (GTE-base, 109M) | **0.985** | **1.000** | **1.000** | **0.993** | **0.825** | **0.950** | 0.980 | **0.888** |
| Bipartite PPR   | 0.830 | 0.970 | 0.995 | 0.900 | 0.630 | 0.800 | 0.875 | 0.721 |
| Graph (PPR + AES) | 0.785 | 0.980 | **1.000** | 0.877 | 0.570 | 0.785 | 0.885 | 0.687 |

**Key result:** AES improves HR@5 over Bipartite PPR on both datasets, at the cost of HR@1 — but **this does not make it competitive with modern dense retrieval.** All three base-size encoders match Aethel's 1.000 HR@5 on 2Wiki while scoring 0.960–0.985 HR@1 against Aethel's 0.785, and on MuSiQue BGE-base leads HR@5 by 10 points (0.985 vs 0.885). An earlier version of this README claimed HR@5 coverage as Aethel's contribution; that claim held only against MiniLM and is withdrawn. What the graph retains is an explicit, auditable entity–passage traversal path — not superior retrieval quality.

BGE and E5 use their authors' prescribed query/passage instruction prefixes; omitting them materially understates both models.

### AES ablation

AES bundles three mechanisms. Adding one at a time (mean PPR seeds per query in parentheses):

| Configuration | 2Wiki HR@1 | 2Wiki HR@5 | MuSiQue HR@1 | MuSiQue HR@5 |
| :--- | :---: | :---: | :---: | :---: |
| Exact match only (= Bipartite PPR) | **0.830** (3.28) | 0.995 | **0.630** (4.04) | 0.875 |
| + alias expansion | 0.780 (3.82) | **1.000** | 0.490 (5.42) | 0.880 |
| + substring overlap | 0.775 (3.87) | **1.000** | 0.490 (5.42) | 0.880 |
| + weighted teleport (= full AES) | 0.785 (3.87) | **1.000** | 0.570 (5.42) | **0.885** |

- **Substring overlap is inert** — bit-identical on MuSiQue; on 2Wiki it costs 0.005 HR@1 and gains nothing. It should be removed.
- **Alias expansion supplies the entire HR@5 gain** but costs 0.050 (2Wiki) and 0.140 (MuSiQue) HR@1.
- **Weighted teleport is the most valuable component** — it adds no seeds yet recovers 0.080 of MuSiQue HR@1 and contributes the final +0.005 HR@5.
- Total AES coverage gain is **+0.005 / +0.010 HR@5 — one and two questions out of 200.** Not a robust effect at this sample size.

### Open-corpus and scaling

On the 4,123-chunk financial corpus (20 multi-hop queries), BM25 still leads (HR@5 0.700). Aethel-NER3 reaches 0.600, which **ties** BGE-base (0.600) rather than beating dense retrieval as previously reported against MiniLM (0.450).

The paper's central scaling claim **survives the stronger baselines and generalizes**. From 100 → 4,123 chunks, multi-hop HR@5 changes by:

| BM25 | Aethel-NER3 | MiniLM | BGE-base | E5-base | GTE-base |
| :---: | :---: | :---: | :---: | :---: | :---: |
| −0.100 | −0.200 | −0.390 | −0.310 | −0.350 | **−0.500** |

Every dense encoder degrades 3–5× faster than BM25, so the collapse is a property of dense retrieval on keyword-dense financial text, not an artifact of an outdated model. Note that **GTE-base is the best encoder on MuSiQue and the worst on the financial corpus** — closed-pool benchmark rank does not predict open-corpus behavior.


## Reproducing the Results

```bash
git clone https://github.com/anonymous/aethel-clean.git
cd aethel-clean
pip install -r requirements.txt

# Run the full benchmark (requires 'datasets' library, streams from HuggingFace, ~5 min)
# Note: Results are cached to eval_cache.json
PYTHONPATH=. python3 backend/public_benchmark.py
```

Results are cached to `eval_cache.json` for instant reproduction. To recompute from scratch, delete the cache file.

## Running the Diligence System

To run the full multi-agent diligence pipeline against your own documents:

1. Configure your `.env` file (based on `.env.example`). Note that `OPENAI_API_KEY` is required at startup because `RAGService.__init__` instantiates HippoRAG unconditionally upon initialization:
   ```env
   OPENAI_API_KEY=your-api-key-here
   ```
2. Place documents for ingestion. Create a `docs/` directory at the project root and place your PDF/text files there (otherwise the ingestion script will exit silently with an error):
   ```bash
   mkdir -p docs
   # [Place actual PDF or text files inside ./docs/]
   PYTHONPATH=. python3 backend/ingest.py
   ```
3. Run the orchestrator:
   ```bash
   PYTHONPATH=. python3 backend/main.py
   ```

---

## License

MIT License. Academic reference: see `paper.tex`.
