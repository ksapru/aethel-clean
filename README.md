# Dense Retrieval Degrades Faster Than BM25 at Scale

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

Code, corpus and evaluation harness for the paper of the same name (`paper.tex`). The project is codenamed **Aethel**.

We measure how lexical, dense, and graph-based retrievers degrade as a corpus grows, using multi-hop queries over real financial disclosures. **The headline result: every one of four dense bi-encoders spanning 22M–109M parameters loses far more multi-hop HR@5 than BM25 as the pool grows from 100 to 4,123 chunks (−0.310 to −0.500 versus BM25's −0.100).** Gold passages are held in every subsample, so only distractor density varies. Testing four encoders rather than one establishes this as a property of dense retrieval on keyword-dense text, not an artifact of an outdated baseline.

Against that backdrop we report a **negative result for graph retrieval**: bipartite Personalized PageRank over an entity–passage graph, in the style of HippoRAG v2, does not beat a well-chosen dense encoder or BM25 at open-corpus scale.

> **Scope note.** The graph retriever is a reimplementation of the bipartite PPR retrieval stage of HippoRAG v2, not an extension of it. It does not use HippoRAG's LLM-based entity extraction, continual-memory components, or query-to-node linking. Differences from published HippoRAG numbers reflect the reimplementation and our regex-based entity extraction.

## The retriever under test

The system evaluated in the paper has two components:

1. **Bipartite Entity–Passage Graph** — entities (`V_e`) and passages (`V_p`) are vertices in a bipartite graph `G = (V_p ∪ V_e, E)`. Query entities seed a PPR random walk over sparse adjacency paths to rank passages.

2. **Alias-Expanded Seeding (AES)** — expands the PPR personalization vector with alias matches and substring overlaps, seeding additional start nodes across entity variants (e.g. "Apollo" → "Apollo Global Management"). This is surface-form matching only: no mention-clustering, no pronoun resolution, no access to discourse structure. The ablation below shows its net contribution is one to two questions in 200.

## Results

Every table below is generated from committed result files — `eval_cache.json`, `backend/data/scaling_results.json`, and `backend/data/open_corpus_results.json` — and matches the corresponding table in `paper.tex`.

### 1. Scale sensitivity (headline result)

Multi-hop HR@5 as the corpus grows. Gold passages are retained in every subsample, so only distractor density varies.

| System | 100 | 500 | 1K | 2K | 4.1K | Δ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.800 | 0.760 | 0.710 | 0.650 | 0.700 | **-0.100** |
| Graph (Aethel-NER3) | 0.800 | 0.670 | 0.660 | 0.610 | 0.600 | **-0.200** |
| Dense (BGE-base, 109M) | 0.910 | 0.790 | 0.700 | 0.620 | 0.600 | **-0.310** |
| Dense (E5-base, 109M) | 0.800 | 0.650 | 0.590 | 0.470 | 0.450 | **-0.350** |
| Dense (MiniLM, 22M) | 0.840 | 0.660 | 0.610 | 0.560 | 0.450 | **-0.390** |
| Dense (GTE-base, 109M) | 0.800 | 0.670 | 0.520 | 0.410 | 0.300 | **-0.500** |

**Every dense encoder degrades 3–5× faster than BM25**, across a 5× parameter range and four independently trained models. GTE-base collapses hardest (−0.500) despite being the strongest encoder on the closed-pool benchmarks — closed-pool rank does not predict open-corpus behavior.

### 2. Open-corpus financial retrieval

4,123-chunk corpus, 40 annotated queries (20 single-hop, 20 multi-hop). Gold labels frozen before any retriever ran.

| System | Multi HR@1 | Multi HR@5 | Multi MRR | Multi R@5 | All HR@5 | All MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 | 0.350 | **0.700** | 0.468 | 0.525 | 0.650 | 0.367 |
| Dense (MiniLM) | 0.200 | 0.450 | 0.318 | 0.275 | 0.450 | 0.311 |
| Dense (BGE-base) | 0.250 | 0.600 | 0.408 | 0.425 | 0.525 | 0.404 |
| Dense (E5-base) | 0.200 | 0.450 | 0.315 | 0.300 | 0.475 | 0.319 |
| Dense (GTE-base) | 0.200 | 0.300 | 0.248 | 0.200 | 0.375 | 0.324 |
| Aethel-Reg | 0.050 | 0.400 | 0.175 | 0.200 | 0.275 | 0.121 |
| Aethel-NER3 | 0.150 | 0.600 | 0.360 | 0.350 | 0.425 | 0.250 |
| Hybrid-RRF | 0.350 | 0.650 | **0.479** | 0.475 | 0.475 | 0.367 |

**BM25 leads on multi-hop HR@5 (0.700).** The graph retriever reaches 0.600 — which *ties* BGE-base rather than beating dense retrieval, as earlier versions reported against MiniLM alone (0.450).

### 3. Closed-pool benchmarks

200-question random samples, seed=42, from official MuSiQue and 2WikiMultiHopQA validation splits.

| Method | 2Wiki HR@1 | 2Wiki HR@3 | 2Wiki HR@5 | 2Wiki MRR | MuSiQue HR@1 | MuSiQue HR@3 | MuSiQue HR@5 | MuSiQue MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Sparse (TF-IDF) | 0.790 | 0.965 | 0.990 | 0.873 | 0.580 | 0.780 | 0.845 | 0.681 |
| Dense (MiniLM, 22M) | 0.900 | 0.990 | 0.995 | 0.940 | 0.765 | 0.935 | 0.960 | 0.849 |
| Dense (E5-base, 109M) | 0.960 | **1.000** | **1.000** | 0.980 | 0.740 | 0.915 | 0.965 | 0.834 |
| Dense (BGE-base, 109M) | **0.985** | **1.000** | **1.000** | **0.993** | 0.795 | 0.925 | **0.985** | 0.865 |
| Dense (GTE-base, 109M) | **0.985** | **1.000** | **1.000** | **0.993** | **0.825** | **0.950** | 0.980 | **0.888** |
| Graph (PPR, exact match) | 0.830 | 0.970 | 0.995 | 0.900 | 0.630 | 0.800 | 0.875 | 0.721 |
| Graph (PPR + AES) | 0.785 | 0.980 | **1.000** | 0.877 | 0.570 | 0.785 | 0.885 | 0.687 |

BGE and E5 use their authors' prescribed query/passage instruction prefixes; omitting them materially understates both models.

**All three base-size encoders match the graph retriever's 1.000 HR@5 on 2Wiki while scoring 20+ points higher on HR@1**, and BGE-base leads MuSiQue HR@5 by 10 points. Earlier versions of this README claimed HR@5 coverage as the graph's contribution; that held only against MiniLM and is withdrawn.

### 4. AES ablation

AES bundles three mechanisms. Each row adds exactly one, on top of exact entity matching.

| Configuration | 2Wiki seeds | 2Wiki HR@1 | 2Wiki HR@5 | MuSiQue seeds | MuSiQue HR@1 | MuSiQue HR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Exact match only (= Graph PPR) | 3.28 | 0.830 | 0.995 | 4.04 | 0.630 | 0.875 |
| + alias expansion | 3.82 | 0.780 | 1.000 | 5.42 | 0.490 | 0.880 |
| + substring overlap | 3.87 | 0.775 | 1.000 | 5.42 | 0.490 | 0.880 |
| + weighted teleport (= full AES) | 3.87 | 0.785 | 1.000 | 5.42 | 0.570 | 0.885 |

- **Substring overlap is inert** — bit-identical on MuSiQue; on 2Wiki it adds 0.05 seeds and costs 0.005 HR@1 while gaining nothing. It should be removed.
- **Alias expansion supplies the entire HR@5 gain** but costs 0.050 (2Wiki) and 0.140 (MuSiQue) HR@1.
- **Weighted teleport is the most valuable component** — it adds no seeds, yet recovers 0.080 of MuSiQue HR@1 and contributes the final +0.005 HR@5.
- Total AES coverage gain is **+0.005 / +0.010 HR@5 — one and two questions out of 200.** Not a robust effect at this sample size.

## Limitations

Open-corpus conclusions rest on **40 queries labelled by a single annotator**, with no inter-annotator agreement statistic. At N=20 multi-hop, the paired-bootstrap CI on the Hybrid-RRF vs BM25 MRR difference is [−0.163, +0.192] — wide enough to accommodate a substantial effect in either direction. The scaling result is a **single-corpus finding** on proper-noun-dense financial text, precisely the condition that favours BM25; we do not claim it generalizes to corpora with different lexical properties. See the Limitations section of `paper.tex` for the full list.

## Reproducing

```bash
git clone https://github.com/ksapru/aethel-clean.git
cd aethel-clean
pip install -r requirements.txt

# Closed-pool benchmarks + AES ablation (tables 3 and 4)
PYTHONPATH=. python3 backend/public_benchmark.py

# Open-corpus evaluation (table 2)
PYTHONPATH=. python3 backend/evaluate_aethel.py

# Scale-sensitivity curve (table 1)
PYTHONPATH=. python3 backend/scaling_curve.py
```

Closed-pool results are cached in `eval_cache.json` for instant reproduction; delete it to recompute. PPR is deterministic power iteration over a fixed graph and seed vector, so graph results are bit-identical across runs. spaCy NER introduces ≤0.025 variation in Aethel-NER3 figures across environments.

**On low-memory machines** (8 GB or less), the four encoders will exhaust RAM and swap-thrash. Force CPU and a smaller batch:

```bash
AETHEL_DENSE_DEVICE=cpu AETHEL_DENSE_BATCH=8 PYTHONPATH=. python3 backend/evaluate_aethel.py
```

### Verifying the AES refactor

AES is implemented as flags on a single retriever. `backend/test_aes_refactor.py` is a differential test proving the flag refactor reproduces the pre-refactor implementation exactly:

```bash
git show 3863346:backend/public_benchmark.py > /tmp/orig_pb.py
PYTHONPATH=. python3 backend/test_aes_refactor.py --orig /tmp/orig_pb.py
```

## Also in this repo

Neither of these is part of the paper, and neither is evaluated. They are a demo application built on top of the retriever, not a contribution.

- **Orchestrated specialist swarm** (`backend/agents/`, `backend/main.py`) — forwards retrieved passages to domain-specialist agents (Liquidity, Valuation, Diligence Auditor) under an Orchestrator.
- **Web frontend** (`frontend/`) — Next.js UI for the diligence app.

<details>
<summary>Running the diligence demo</summary>

1. Configure `.env` (see `.env.example`). `OPENAI_API_KEY` is required at startup because `RAGService.__init__` instantiates HippoRAG unconditionally:
   ```env
   OPENAI_API_KEY=your-api-key-here
   ```
2. Place PDF/text files in a `docs/` directory at the project root (the ingestion script exits silently without it):
   ```bash
   mkdir -p docs
   PYTHONPATH=. python3 backend/ingest.py
   ```
3. Run the orchestrator:
   ```bash
   PYTHONPATH=. python3 backend/main.py
   ```

</details>

---

## License

MIT License. Academic reference: see `paper.tex`.
