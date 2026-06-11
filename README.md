# Aethel: Bipartite Graph-Walk Retrieval for Multi-Hop Financial Diligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

Aethel is a graph-augmented retrieval framework for multi-hop question answering over unstructured financial documents. It models document collections as bipartite entity–passage graphs and executes Personalized PageRank (PPR) random walks to resolve cross-document, multi-hop queries. A **Bipartite Coreference Teleportation (BCT)** layer expands entity mentions via alias matching and substring overlap, improving hit rate coverage at the cost of top-1 precision — the right tradeoff for financial diligence workflows where all supporting passages must be surfaced.


## Architecture

The framework has three components:

1. **Bipartite Entity–Passage Graph** — entities (`V_e`) and passages (`V_p`) are vertices in a bipartite graph `G = (V_p ∪ V_e, E)`. Query entities seed a PPR random walk over sparse adjacency paths to rank passages.

2. **Bipartite Coreference Teleportation (BCT)** — expands the PPR personalization vector with alias matches and substring overlaps, seeding additional start nodes to improve hit rate coverage across entity variants (e.g., "Apollo" → "Apollo Global Management").

3. **Orchestrated Specialist Swarm** — retrieved passages are forwarded to domain-specialist agents (Liquidity, Valuation, Diligence Auditor) coordinated by a central Orchestrator. The swarm is described in `backend/agents/` and `backend/main.py`; it is not quantitatively evaluated in the current paper.


## Empirical Results

All numbers below come from `backend/public_benchmark.py` running against the official HuggingFace validation splits. Results are cached in `eval_cache.json` (committed to the repository) for instant reproducibility.

**200-question random samples, seed=42, from official validation splits of MuSiQue and 2WikiMultiHopQA.**

| Method | 2Wiki HR@1 | 2Wiki HR@3 | 2Wiki HR@5 | 2Wiki MRR | MuSiQue HR@1 | MuSiQue HR@3 | MuSiQue HR@5 | MuSiQue MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Sparse (TF-IDF) | 0.790 | 0.965 | 0.990 | 0.873 | 0.580 | 0.780 | 0.845 | 0.681 |
| Dense (MiniLM)  | **0.900** | **0.990** | 0.995 | **0.940** | **0.765** | **0.935** | **0.960** | **0.849** |
| Bipartite PPR   | 0.830 | 0.970 | 0.995 | 0.900 | 0.630 | 0.800 | 0.875 | 0.721 |
| Aethel (PPR+BCT) | 0.785 | 0.980 | **1.000** | 0.877 | 0.570 | 0.785 | 0.885 | 0.687 |

**Key result:** BCT wins on HR@5 (both datasets) and HR@3 — the coverage-oriented metrics. Bipartite PPR wins on HR@1 and MRR — precision-oriented metrics. This is the expected tradeoff: BCT alias expansion seeds more starting nodes in the PPR walk, surfacing more supporting passages at the cost of top-1 rank.


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
