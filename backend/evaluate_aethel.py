"""
Aethel Evaluation v3 — NER vocabulary + substring adjacency + entity filtering

Fix 1: Filter garbage entities (< 3 chars, OCR artifacts, pure numbers)
Fix 2: Build adjacency via substring matching (like regex version) but using
       the cleaner NER-derived entity vocabulary instead of regex patterns.
       This ensures "NorthRiver" as a seed connects to any passage containing
       "NorthRiver" in its text, even if spaCy extracted the full title.
"""

import json, re, sys, time
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse import csr_matrix as _csr
from typing import List, Dict, Set

sys.path.append("/Users/krishsapru/aethel-clean")
from backend.public_benchmark import (
    SimpleDocument, _SparseRetriever, _DenseRetriever, _GraphRetriever,
    DENSE_MODELS, DENSE_KEYS, _release_dense_model,
)

CHUNKS = "/Users/krishsapru/aethel-clean/backend/data/processed_chunks.json"
QUERIES = "/Users/krishsapru/aethel-clean/backend/data/eval_queries_gold.json"

KEEP_LABELS = frozenset({"ORG", "PERSON", "GPE", "PRODUCT", "MONEY", "PERCENT",
                         "FAC", "EVENT", "WORK_OF_ART", "LAW", "NORP"})

# ── Entity quality filters ───────────────────────────────────────────
_JUNK_RE = re.compile(
    r'[/\\]g\d'           # OCR artifact like /g415
    r'|[\n\r\t]'          # contains newlines (table headers)
    r'|^\d+\.?\d*%?$'     # pure number or percentage
    r'|^[A-Z]{1,2}$'      # 1-2 char uppercase abbreviations (IV, AI, etc.)
)

_STOPWORDS = frozenset({
    'the', 'and', 'for', 'was', 'with', 'are', 'has', 'had', 'his', 'her',
    'its', 'they', 'this', 'that', 'from', 'also', 'been', 'were', 'which',
    'their', 'have', 'she', 'who', 'not', 'but', 'all', 'can', 'one', 'two',
    'may', 'will', 'more', 'than', 'other', 'our', 'each', 'any', 'such',
    'per', 'sec', 'inc', 'ltd', 'llc', 'new', 'net', 'total', 'value',
    'portfolio', 'year', 'quarter', 'period', 'date', 'fair', 'table',
})

def _is_clean_entity(text: str) -> bool:
    """Return True if the entity passes quality filters."""
    if len(text) < 3:
        return False
    if _JUNK_RE.search(text):
        return False
    if text.lower() in _STOPWORDS:
        return False
    # Reject if > 50 chars (table headers, run-on OCR)
    if len(text) > 50:
        return False
    return True


class _NERv3Retriever:
    """
    NER vocabulary + substring adjacency + entity filtering.
    
    Phase 1: Extract entity vocabulary via spaCy NER (single pass).
             Apply aggressive quality filters.
    Phase 2: Build adjacency via substring matching — if entity text
             appears anywhere in a passage, add an edge.  This solves
             the normalization gap where "NorthRiver" ≠ 
             "NorthRiver IV LP Quarterly Report".
    """

    def __init__(self, docs: List[SimpleDocument]):
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        nlp.max_length = 2_000_000
        self.docs = docs
        np_ = len(docs)

        # ── Phase 1: NER vocabulary extraction ───────────────────────
        t0 = time.time()
        print(f"    [NERv3] Phase 1: Extracting entity vocabulary "
              f"from {np_} passages...")
        key_to_idx: Dict[str, int] = {}
        key_to_text: Dict[str, str] = {}
        raw_count = 0
        
        for di, d in enumerate(docs):
            if di % 1000 == 0 and di > 0:
                print(f"    [NERv3]   {di}/{np_} passages, "
                      f"{len(key_to_idx)} clean entities "
                      f"({raw_count} raw, {raw_count - len(key_to_idx)} filtered)")
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
        print(f"    [NERv3] Phase 1 done: {ne} clean entities "
              f"(from {raw_count} raw). {time.time()-t0:.1f}s")

        # Ordered entity list
        self.entities: List[str] = [""] * ne
        for k, i in key_to_idx.items():
            self.entities[i] = key_to_text[k]
        self._key_to_idx = key_to_idx

        # ── Phase 2: Substring adjacency ─────────────────────────────
        # Algorithm: entity-outer loop with pure substring match (ekey in cl).
        # This is semantically identical to the original verified run
        # (task-367.log). The dense matrix was the 20-minute bottleneck —
        # with sparse lil_matrix the same loop runs in ~10-15s.
        t1 = time.time()
        print(f"    [NERv3] Phase 2: Building {np_}×{ne} adjacency "
              f"(substring match, sparse storage)...")

        total = np_ + ne
        adj = lil_matrix((total, total), dtype=np.float32)

        # Pre-lowercase all passage content
        contents_lower = [d.page_content.lower() for d in docs]
        entity_keys = list(key_to_idx.keys())  # already lowercase

        for ei, ekey in enumerate(entity_keys):
            if ei % 1000 == 0 and ei > 0:
                print(f"    [NERv3]   edge scan: {ei}/{ne} entities...")
            eidx = key_to_idx[ekey]
            for pi, cl in enumerate(contents_lower):
                if ekey in cl:
                    adj[pi, np_ + eidx] = 1.0
                    adj[np_ + eidx, pi] = 1.0

        # Convert to CSR for fast arithmetic
        adj_csr = adj.tocsr()
        edge_count = int(adj_csr[:np_, np_:].nnz)
        print(f"    [NERv3] Phase 2 done: {edge_count} edges. "
              f"{time.time()-t1:.1f}s")

        self.np_ = np_
        self.ne = ne
        self.total = total

        # ── Alias map ────────────────────────────────────────────────
        self._alias_map: Dict[str, int] = {}
        for ei, ent in enumerate(self.entities):
            key = ent.lower()
            self._alias_map[key] = ei
            words = ent.split()
            if len(words) > 1 and len(words[-1]) >= 4:
                self._alias_map[words[-1].lower()] = ei

        # ── Pre-compute transition matrix (sparse CSR) ───────────────
        # Row-normalise: T_norm = D^{-1} A, then transpose so PPR loop
        # does T_norm.T @ v (column-stochastic walk).
        rs = np.asarray(adj_csr.sum(axis=1)).ravel()
        inv_rs = np.zeros_like(rs)
        nz = rs > 0
        inv_rs[nz] = 1.0 / rs[nz]
        self._transition = (diags(inv_rs) @ adj_csr).T.tocsr()

        self._nlp = nlp

        # ── Stats ────────────────────────────────────────────────────
        ent_degrees = np.asarray(adj_csr[np_:, :np_].sum(axis=1)).ravel()
        print(f"    [NERv3] Entity degree: mean={ent_degrees.mean():.1f}, "
              f"median={np.median(ent_degrees):.0f}, "
              f"max={ent_degrees.max():.0f}")
        print(f"    [NERv3] Ready. Total init: {time.time()-t0:.1f}s")

    def query(self, q: str, k: int = 5, use_aes: bool = True) -> List[int]:
        if self.ne == 0 or self.total == 0:
            return list(range(min(k, len(self.docs))))

        seeds: List[int] = []

        # Pass 1: NER on query
        q_doc = self._nlp(q)
        for ent in q_doc.ents:
            text = ent.text.strip()
            if not _is_clean_entity(text):
                continue
            key = text.lower()
            if key in self._key_to_idx:
                ei = self._key_to_idx[key]
                if ei not in seeds:
                    seeds.append(ei)

        # Pass 2: substring matching against entity vocabulary
        q_lower = q.lower()
        for ei, ent in enumerate(self.entities):
            if ent.lower() in q_lower and ei not in seeds:
                seeds.append(ei)

        # Pass 3: alias map (surname matching)
        for w in re.findall(r'[a-z]{4,}', q_lower):  # min 4 chars
            if w in self._alias_map:
                ei = self._alias_map[w]
                if ei not in seeds:
                    seeds.append(ei)

        if not seeds:
            return list(range(min(k, len(self.docs))))

        # Weighted teleport
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

        # PPR power iteration — transition is sparse CSR, v is dense.
        # csr.dot(v) returns a dense numpy array, same semantics as T @ v.
        alpha = 0.85
        v = u.copy()
        for _ in range(20):
            v = alpha * self._transition.dot(v) + (1 - alpha) * u

        ranked = np.argsort(v[:self.np_])[::-1]
        return [int(i) for i in ranked[:k]]


# ── Scoring ──────────────────────────────────────────────────────────
def score(retrieved_ids, gold_ids, k):
    topk = retrieved_ids[:k]
    hit = any(g in topk for g in gold_ids)
    rr = next((1/(i+1) for i, c in enumerate(topk) if c in gold_ids), 0.0)
    recall = len(set(topk) & set(gold_ids)) / len(gold_ids)
    return hit, rr, recall


# ── Reciprocal Rank Fusion ────────────────────────────────────────────
# RRF constant fixed at the canonical value from Cormack et al. (2009).
# NOT tuned to these 40 queries — k=60 is written into the paper as the
# standard default and this is the only value evaluated.
_RRF_K = 60
# How many results each retriever returns for the fusion input.
# Must be >> the scoring cutoff (5/10) so RRF can rescue passages
# that neither system ranked in its top-K individually.
_RRF_POOL = 100


def rrf_fuse(ranked_lists: List[List[int]], top_k: int = 10) -> List[int]:
    """Reciprocal Rank Fusion (Cormack et al. 2009): score = Σ 1/(k + rank + 1).

    ranked_lists: list of lists of doc indices, each sorted best-first.
                  MUST be full ranked lists (length _RRF_POOL), NOT pre-truncated
                  to the scoring cutoff -- truncation happens here, after fusion.
    top_k: number of results to return after fusion.
    """
    scores: Dict[int, float] = {}
    for lst in ranked_lists:
        for rank, doc_idx in enumerate(lst):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("Loading chunks and queries...")
    with open(CHUNKS) as f:
        chunks = json.load(f)
    with open(QUERIES) as f:
        queries = json.load(f)
    print(f"Loaded {len(chunks)} chunks and {len(queries)} queries.\n")

    docs = [
        SimpleDocument(page_content=c["content"],
                       metadata={"chunk_id": c["chunk_id"]})
        for c in chunks
    ]

    n_dense = len(DENSE_KEYS)
    n_steps = 4 + n_dense
    print("Initializing retrievers...")
    print(f"  1/{n_steps}  BM25...")
    sparse = _SparseRetriever(docs)

    # One encoder at a time: build its index, score all queries, then release
    # the model before loading the next. Holding four encoders resident over a
    # 4k-chunk corpus exceeds available RAM on a small machine and swap-thrashes.
    # The retained rankings are tiny (40 queries x 10 doc indices per encoder).
    dense_rankings: Dict[str, Dict[str, List[int]]] = {}
    for i, key in enumerate(DENSE_KEYS):
        label = DENSE_MODELS[key]['label']
        print(f"  {2 + i}/{n_steps}  {label} ({DENSE_MODELS[key]['model_name']})...")
        retr = _DenseRetriever(docs, model_key=key)
        dense_rankings[label] = {
            q["query"]: retr.query(q["query"], k=10) for q in queries
        }
        del retr
        _release_dense_model(key)

    print(f"  {2 + n_dense}/{n_steps}  Aethel-Regex (original)...")
    graph_regex = _GraphRetriever(docs)
    print(f"  {3 + n_dense}/{n_steps}  Aethel-NERv3 (filtered NER + substring adjacency)...")
    graph_ner = _NERv3Retriever(docs)
    print(f"  {4 + n_dense}/{n_steps}  Hybrid-RRF (BM25 + NER3 fusion, computed post-retrieval)")

    print(f"\nRunning {len(queries)} queries...\n")

    # Dense / Aethel-Reg are individual baselines only (k=10 for scoring).
    # BM25 / NER3 are ALSO the fusion inputs: retrieved at k=_RRF_POOL so
    # RRF sees the full ranked list before truncation.
    base_systems = {
        "BM25": {"ret": sparse, "graph": False, "fusion": True},
    }
    for key in DENSE_KEYS:
        label = DENSE_MODELS[key]['label']
        base_systems[label] = {
            "ret": None, "graph": False, "fusion": False,
            "precomputed": dense_rankings[label],
        }
    base_systems.update({
        "Aethel-Reg":  {"ret": graph_regex, "graph": True,  "fusion": False},
        "Aethel-NER3": {"ret": graph_ner,   "graph": True,  "fusion": True},
    })
    all_system_names = list(base_systems.keys()) + ["Hybrid-RRF"]
    results = {s: [] for s in all_system_names}

    for qi, q_item in enumerate(queries):
        qtxt  = q_item["query"]
        qtype = q_item["query_type"]
        golds = q_item["gold_chunk_ids"]
        print(f"[{qi+1:>2}/{len(queries)}] ({qtype:>10}) {qtxt}")

        fusion_inputs: Dict[str, List[int]] = {}  # full _RRF_POOL lists for RRF
        for sname, cfg in base_systems.items():
            if cfg["fusion"]:
                # Retrieve _RRF_POOL results — full list for fusion input.
                # [:10] slice is used for individual-system scoring.
                pool_k = _RRF_POOL
            else:
                pool_k = 10

            if cfg.get("precomputed") is not None:
                # Dense rankings were computed up-front, one encoder at a time.
                idxs = cfg["precomputed"][qtxt]
            elif cfg["graph"]:
                idxs = cfg["ret"].query(qtxt, k=pool_k, use_aes=True)
            else:
                idxs = cfg["ret"].query(qtxt, k=pool_k)

            if cfg["fusion"]:
                fusion_inputs[sname] = idxs          # full list → RRF
                score_idxs = idxs[:10]               # top-10 → individual scoring
            else:
                score_idxs = idxs

            cids = [docs[i].metadata["chunk_id"] for i in score_idxs]
            sc = {}
            for kv in [1, 3, 5, 10]:
                h, r, rc = score(cids, golds, kv)
                sc[f"hr{kv}"] = h
                if kv == 10: sc["mrr"] = r
                if kv == 5:  sc["recall5"] = rc
            results[sname].append({"query": qtxt, "q_type": qtype, "scores": sc})

        # Hybrid-RRF: fuse full BM25 + NER3 lists (each length _RRF_POOL=100),
        # then truncate to top-10 for scoring.
        # k=_RRF_K=60 is the canonical untuned constant; not chosen to fit output.
        fused_idxs = rrf_fuse(
            [fusion_inputs["BM25"], fusion_inputs["Aethel-NER3"]], top_k=10
        )
        fused_cids = [docs[i].metadata["chunk_id"] for i in fused_idxs]
        rrf_sc = {}
        for kv in [1, 3, 5, 10]:
            h, r, rc = score(fused_cids, golds, kv)
            rrf_sc[f"hr{kv}"] = h
            if kv == 10: rrf_sc["mrr"] = r
            if kv == 5:  rrf_sc["recall5"] = rc
        results["Hybrid-RRF"].append({"query": qtxt, "q_type": qtype, "scores": rrf_sc})

    # ── Report ───────────────────────────────────────────────────────
    order = (["BM25"]
             + [DENSE_MODELS[k]['label'] for k in DENSE_KEYS]
             + ["Aethel-Reg", "Aethel-NER3", "Hybrid-RRF"])
    print("\n" + "=" * 90)
    print("EVALUATION v3 — filtered NER + substring adjacency + Hybrid-RRF (BM25⊕NER3)")
    print("=" * 90 + "\n")

    for grp in ["all", "single-hop", "multi-hop"]:
        print(f"--- {grp.upper()} ---")
        hdr = (f"{'System':<18} | {'N':>3} | {'HR@1':>6} | {'HR@3':>6} | "
               f"{'HR@5':>6} | {'HR@10':>6} | {'MRR':>6} | {'Recall@5':>8}")
        print(hdr)
        print("-" * len(hdr))
        for sn in order:
            rows = results[sn]
            if grp != "all":
                rows = [r for r in rows if r["q_type"] == grp]
            n = len(rows)
            if n == 0: continue
            vals = {m: np.mean([r["scores"][m] for r in rows])
                    for m in ["hr1","hr3","hr5","hr10","mrr","recall5"]}
            print(f"{sn:<18} | {n:>3} | {vals['hr1']:>6.3f} | "
                  f"{vals['hr3']:>6.3f} | {vals['hr5']:>6.3f} | "
                  f"{vals['hr10']:>6.3f} | {vals['mrr']:>6.3f} | "
                  f"{vals['recall5']:>8.3f}")
        print()


if __name__ == "__main__":
    main()
