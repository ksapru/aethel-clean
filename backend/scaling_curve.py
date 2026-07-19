"""
scaling_curve.py — HR@5 vs. corpus size for BM25, Dense, Aethel-NER3.

Design decisions (state in caption):
  - Gold passages are ALWAYS retained in every subsample.
    HR@5 degrades only because distractors grow, not because the answer
    is absent. This measures retriever precision as the haystack expands.
  - Random seed: np.random.default_rng(42). Report in paper.
  - N_TRIALS = 5 resamples per non-full size; full corpus = 1 trial
    (deterministic, no randomness).
  - Phase 1 (spaCy NER vocab) run ONCE on the full corpus and reused
    across all subsamples. Only Phase 2 (adjacency) is rebuilt per subset.
  - Dense passage embeddings pre-computed once; sliced per subset.

Output: scaling_results.json
"""
import json, re, sys, time
import numpy as np
from scipy.sparse import lil_matrix, diags
from typing import List, Dict, Tuple, Set

sys.path.append("/Users/krishsapru/aethel-clean")
from backend.public_benchmark import SimpleDocument, _SparseRetriever
from backend.evaluate_aethel import (
    KEEP_LABELS, _is_clean_entity, _RRF_POOL,
)

CHUNKS  = "/Users/krishsapru/aethel-clean/backend/data/processed_chunks.json"
QUERIES = "/Users/krishsapru/aethel-clean/backend/data/eval_queries_gold.json"
OUT     = "/Users/krishsapru/aethel-clean/backend/data/scaling_results.json"

SIZES    = [100, 500, 1000, 2000, 4123]
N_TRIALS = 5    # resamples per non-full size
RNG      = np.random.default_rng(42)


# ── Subset-aware Dense Retriever ─────────────────────────────────────────────
class _SubsetDenseRetriever:
    """Dense retriever that slices pre-computed embeddings — no re-encoding."""

    def __init__(self, subset_orig_indices: List[int],
                 all_embeddings: np.ndarray,
                 docs: List[SimpleDocument],
                 model):
        self._model = model
        self._orig  = subset_orig_indices          # indices into full docs list
        self._embs  = all_embeddings[subset_orig_indices]   # (|subset|, dim)
        self._docs  = docs

    def query(self, q: str, k: int = 5) -> List[str]:
        """Return up to k chunk_ids from the subset."""
        qv   = self._model.encode([q], normalize_embeddings=True)[0]
        sims = self._embs @ qv                     # cosine (embeddings normalised)
        top  = np.argsort(sims)[::-1][:k]
        return [self._docs[self._orig[i]].metadata["chunk_id"] for i in top]


# ── Subset BM25 Retriever ─────────────────────────────────────────────────────
class _SubsetSparseRetriever:
    """Rebuild TF-IDF index over only the subset (IDF must reflect subset IDF)."""

    def __init__(self, subset_docs: List[SimpleDocument]):
        self._inner = _SparseRetriever(subset_docs)
        self._docs  = subset_docs

    def query(self, q: str, k: int = 5) -> List[str]:
        idxs = self._inner.query(q, k=k)
        return [self._docs[i].metadata["chunk_id"] for i in idxs]


# ── Subset NER3 Retriever ─────────────────────────────────────────────────────
class _SubsetNER3Retriever:
    """
    Reuses pre-computed entity vocab (Phase 1 result) and rebuilds only the
    adjacency matrix (Phase 2) over the given subset of passages.
    Query logic is byte-for-byte identical to _NERv3Retriever.query() in
    evaluate_aethel.py — three seeding passes + weighted teleport.
    """

    def __init__(self,
                 subset_docs: List[SimpleDocument],
                 key_to_idx: Dict[str, int],
                 key_to_text: Dict[str, str]):
        self._docs = subset_docs
        self._key_to_idx = key_to_idx
        np_ = len(subset_docs)
        ne  = len(key_to_idx)
        self.np_ = np_
        self.ne  = ne

        # Ordered entity list (needed for Pass 2 substring scan)
        self.entities: List[str] = [""] * ne
        for k, i in key_to_idx.items():
            self.entities[i] = key_to_text[k]

        # Alias map — identical to evaluate_aethel.py lines 150-156
        self._alias_map: Dict[str, int] = {}
        for ei, ent in enumerate(self.entities):
            key = ent.lower()
            self._alias_map[key] = ei
            words = ent.split()
            if len(words) > 1 and len(words[-1]) >= 4:
                self._alias_map[words[-1].lower()] = ei

        # Phase 2: rebuild adjacency on this subset (same algorithm as evaluate_aethel.py)
        total = np_ + ne
        self.total = total
        adj   = lil_matrix((total, total), dtype=np.float32)
        contents_lower = [d.page_content.lower() for d in subset_docs]
        entity_keys    = list(key_to_idx.keys())

        for ekey in entity_keys:
            eidx = key_to_idx[ekey]
            for pi, cl in enumerate(contents_lower):
                if ekey in cl:
                    adj[pi, np_ + eidx] = 1.0
                    adj[np_ + eidx, pi] = 1.0

        adj_csr = adj.tocsr()

        # Transition matrix — identical to evaluate_aethel.py lines 161-165
        rs     = np.asarray(adj_csr.sum(axis=1)).ravel()
        inv_rs = np.zeros_like(rs)
        nz     = rs > 0
        inv_rs[nz] = 1.0 / rs[nz]
        self._transition = (diags(inv_rs) @ adj_csr).T.tocsr()

    def query(self, q: str, k: int = 5) -> List[str]:
        """3-pass seeding + weighted teleport — identical to _NERv3Retriever.query()."""
        if self.ne == 0 or self.total == 0:
            return [self._docs[i].metadata["chunk_id"]
                    for i in range(min(k, len(self._docs)))]

        seeds: List[int] = []
        nlp = _nlp_singleton()

        # Pass 1: NER on query
        q_doc = nlp(q)
        for ent in q_doc.ents:
            text = ent.text.strip()
            if not _is_clean_entity(text):
                continue
            key = text.lower()
            if key in self._key_to_idx:
                ei = self._key_to_idx[key]
                if ei not in seeds:
                    seeds.append(ei)

        # Pass 2: substring match of entity vocab against query string
        q_lower = q.lower()
        for ei, ent in enumerate(self.entities):
            if ent.lower() in q_lower and ei not in seeds:
                seeds.append(ei)

        # Pass 3: alias map (4-char word hits)
        for w in re.findall(r'[a-z]{4,}', q_lower):
            if w in self._alias_map:
                ei = self._alias_map[w]
                if ei not in seeds:
                    seeds.append(ei)

        if not seeds:
            return [self._docs[i].metadata["chunk_id"]
                    for i in range(min(k, len(self._docs)))]

        # Weighted teleport — identical to evaluate_aethel.py lines 212-222
        u = np.zeros(self.total, dtype=np.float32)
        if len(seeds) > 1:
            q_words = set(re.findall(r'[a-z]{4,}', q_lower))
            weights = {}
            for s in seeds:
                ew = set(re.findall(r'[a-z]{4,}', self.entities[s].lower()))
                weights[s] = len(ew & q_words) + 1
            tw = sum(weights.values())
            for s, w in weights.items():
                u[self.np_ + s] = w / tw
        else:
            u[self.np_ + seeds[0]] = 1.0

        # PPR power iteration — identical to evaluate_aethel.py lines 226-229
        alpha = 0.85
        v = u.copy()
        for _ in range(20):
            v = alpha * self._transition.dot(v) + (1 - alpha) * u

        ranked = np.argsort(v[:self.np_])[::-1]
        return [self._docs[int(i)].metadata["chunk_id"] for i in ranked[:k]]


# Singleton spaCy model (load once)
_NLP = None
def _nlp_singleton():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        _NLP.max_length = 2_000_000
    return _NLP


# ── HR@5 scorer ───────────────────────────────────────────────────────────────
def hr_at_k(retrieved_cids: List[str], gold_ids: Set[str], k: int = 5) -> float:
    return float(any(c in gold_ids for c in retrieved_cids[:k]))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Scaling curve: HR@5 vs corpus size — BM25 / Dense / Aethel-NER3")
    print(f"  Sizes: {SIZES}   N_TRIALS: {N_TRIALS}   seed: 42")
    print(f"  Gold retention: YES (gold passages always in subset)")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\nLoading data...")
    with open(CHUNKS)  as f: chunks  = json.load(f)
    with open(QUERIES) as f: queries = json.load(f)
    multi_hop = [q for q in queries if q["query_type"] == "multi-hop"]
    print(f"  {len(chunks)} chunks, {len(multi_hop)} multi-hop queries")

    all_docs = [SimpleDocument(page_content=c["content"],
                               metadata={"chunk_id": c["chunk_id"]})
                for c in chunks]
    all_ids  = [c["chunk_id"] for c in chunks]
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    # Collect all gold chunk_ids across multi-hop queries
    all_gold_ids: Set[str] = set()
    gold_map: Dict[str, Set[str]] = {}
    for q in multi_hop:
        gids = set(q["gold_chunk_ids"])
        gold_map[q["query"]] = gids
        all_gold_ids |= gids
    gold_orig_indices = [id_to_idx[g] for g in all_gold_ids if g in id_to_idx]
    print(f"  {len(all_gold_ids)} unique gold chunks across queries")

    # ── Phase 1: NER vocab (once on full corpus) ─────────────────────────
    print("\nPhase 1: extracting NER vocab from full corpus (runs once)...")
    t0  = time.time()
    nlp = _nlp_singleton()
    key_to_idx: Dict[str, int] = {}
    key_to_text: Dict[str, str] = {}
    raw_count = 0
    for di, d in enumerate(all_docs):
        if di % 1000 == 0 and di > 0:
            print(f"  {di}/{len(all_docs)}, {len(key_to_idx)} entities")
        doc_nlp = nlp(d.page_content)
        for ent in doc_nlp.ents:
            if ent.label_ not in KEEP_LABELS:
                continue
            raw_count += 1
            text = ent.text.strip()
            if not _is_clean_entity(text):
                continue
            key = text.lower()
            if key not in key_to_idx:
                key_to_idx[key] = len(key_to_idx)
                key_to_text[key] = text
    ne = len(key_to_idx)
    print(f"  Phase 1 done: {ne} clean entities ({raw_count} raw). "
          f"{time.time()-t0:.1f}s")

    # ── Pre-compute Dense embeddings (once on full corpus) ────────────────
    print("\nPre-computing Dense embeddings (once)...")
    t1 = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        all_embs = dense_model.encode(
            [d.page_content for d in all_docs],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
        use_dense = True
        print(f"  Dense embeddings done. {time.time()-t1:.1f}s")
    except Exception as e:
        print(f"  Dense unavailable ({e}), skipping.")
        use_dense = False
        all_embs = None
        dense_model = None

    # ── Scaling loop ──────────────────────────────────────────────────────
    systems = ["bm25", "ner3"] + (["dense"] if use_dense else [])
    results = {s: {sz: [] for sz in SIZES} for s in systems}

    for sz in SIZES:
        n_trials = 1 if sz == len(all_docs) else N_TRIALS
        print(f"\n── Size {sz:>4} ({n_trials} trial{'s' if n_trials>1 else ''}) ──")

        for trial in range(n_trials):
            # Subsample: random non-gold indices, then add gold back
            non_gold_ids   = [i for i in range(len(all_docs))
                              if all_docs[i].metadata["chunk_id"] not in all_gold_ids]
            n_non_gold     = sz - len(gold_orig_indices)
            if n_non_gold <= 0:
                # Subset smaller than gold set — just use gold
                chosen = list(gold_orig_indices)
            elif n_non_gold >= len(non_gold_ids):
                chosen = list(range(len(all_docs)))
            else:
                distractor_idxs = RNG.choice(non_gold_ids,
                                              size=n_non_gold,
                                              replace=False).tolist()
                chosen = sorted(set(gold_orig_indices) | set(distractor_idxs))

            subset_docs = [all_docs[i] for i in chosen]
            actual_size = len(subset_docs)
            t_trial = time.time()

            # BM25
            bm25   = _SubsetSparseRetriever(subset_docs)
            hits_b = 0
            for q in multi_hop:
                top = bm25.query(q["query"], k=5)
                hits_b += hr_at_k(top, gold_map[q["query"]])
            hr_b = hits_b / len(multi_hop)
            results["bm25"][sz].append(hr_b)

            # NER3
            ner3   = _SubsetNER3Retriever(subset_docs, key_to_idx, key_to_text)
            hits_n = 0
            for q in multi_hop:
                top = ner3.query(q["query"], k=5)
                hits_n += hr_at_k(top, gold_map[q["query"]])
            hr_n = hits_n / len(multi_hop)
            results["ner3"][sz].append(hr_n)

            # Dense
            if use_dense:
                dr     = _SubsetDenseRetriever(chosen, all_embs,
                                               all_docs, dense_model)
                hits_d = 0
                for q in multi_hop:
                    top = dr.query(q["query"], k=5)
                    hits_d += hr_at_k(top, gold_map[q["query"]])
                hr_d = hits_d / len(multi_hop)
                results["dense"][sz].append(hr_d)
                print(f"  trial {trial+1}: actual_size={actual_size}  "
                      f"BM25={hr_b:.3f}  NER3={hr_n:.3f}  Dense={hr_d:.3f}  "
                      f"({time.time()-t_trial:.1f}s)")
            else:
                print(f"  trial {trial+1}: actual_size={actual_size}  "
                      f"BM25={hr_b:.3f}  NER3={hr_n:.3f}  "
                      f"({time.time()-t_trial:.1f}s)")

    # ── Summarise ─────────────────────────────────────────────────────────
    summary = {}
    for s in systems:
        summary[s] = {}
        for sz in SIZES:
            vals = results[s][sz]
            summary[s][sz] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)),  4),
                "n":    len(vals),
                "raw":  [round(v, 4) for v in vals],
            }

    print("\n\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    hdr = f"{'Size':>6} | " + " | ".join(f"{s:>12}" for s in systems)
    print(hdr)
    print("-" * len(hdr))
    for sz in SIZES:
        row = f"{sz:>6} | "
        row += " | ".join(
            f"{summary[s][sz]['mean']:.3f}±{summary[s][sz]['std']:.3f}"
            for s in systems
        )
        print(row)

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {OUT}")
    print("Paste the contents of that file back to generate the pgfplots figure.")


if __name__ == "__main__":
    main()
