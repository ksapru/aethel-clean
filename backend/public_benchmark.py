"""
Public Benchmark Evaluator for MuSiQue and 2WikiMultiHopQA.

Downloads the HuggingFace datasets, constructs per-question bipartite
graphs and retrieval indices, runs retrieval, and reports genuine HR@k / MRR / RTO.

Results are cached to eval_cache.json so subsequent runs are instant.

Methods evaluated:
  - tfidf:          TF-IDF cosine similarity (sparse lexical baseline)
  - dense:          Bi-encoder embeddings via all-MiniLM-L6-v2 (dense baseline)
  - graph:          Bipartite PPR with exact entity matching
  - graph_regex:    Bipartite PPR with coreference-aware alias expansion (Aethel BCT)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import random
import numpy as np
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Lightweight document wrapper
# ---------------------------------------------------------------------------
class SimpleDocument:
    """Minimal stand-in for langchain Document."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# TF-IDF Sparse Retriever (lexical baseline)
# ---------------------------------------------------------------------------
class _SparseRetriever:
    def __init__(self, docs: List[SimpleDocument]):
        self.docs = docs
        self.vocab = set()
        for d in docs:
            for w in self._tok(d.page_content):
                self.vocab.add(w)
        self.vocab = sorted(self.vocab)
        self.vi = {w: i for i, w in enumerate(self.vocab)}
        self.df = {w: 0 for w in self.vocab}
        for d in docs:
            for w in set(self._tok(d.page_content)):
                if w in self.df:
                    self.df[w] += 1
        self.vecs = [self._vec(d.page_content) for d in docs]
        self.norms = [np.linalg.norm(v) for v in self.vecs]

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9]+', text.lower())

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        toks = self._tok(text)
        for t in toks:
            if t in self.vi:
                tf = toks.count(t)
                idf = np.log((len(self.docs) + 1) / (self.df[t] + 1)) + 1
                v[self.vi[t]] = tf * idf
        return v

    def query(self, q: str, k: int = 5) -> List[int]:
        qv = self._vec(q)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return list(range(min(k, len(self.docs))))
        scores = []
        for i, dv in enumerate(self.vecs):
            dn = self.norms[i]
            sim = np.dot(qv, dv) / (qn * dn) if dn > 0 else 0.0
            scores.append((sim, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scores[:k]]


# ---------------------------------------------------------------------------
# Dense Bi-Encoder Retriever (semantic baseline)
# ---------------------------------------------------------------------------
_DENSE_MODEL = None  # lazy-loaded singleton

def _get_dense_model():
    global _DENSE_MODEL
    if _DENSE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _DENSE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _DENSE_MODEL

class _DenseRetriever:
    """Dense bi-encoder retriever using all-MiniLM-L6-v2 embeddings."""
    def __init__(self, docs: List[SimpleDocument]):
        self.docs = docs
        model = _get_dense_model()
        self.embeddings = model.encode(
            [d.page_content for d in docs],
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def query(self, q: str, k: int = 5) -> List[int]:
        model = _get_dense_model()
        q_emb = model.encode([q], normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb
        ranked = np.argsort(scores)[::-1]
        return [int(idx) for idx in ranked[:k]]


# ---------------------------------------------------------------------------
# Bipartite PPR Graph Retriever
# ---------------------------------------------------------------------------
class _GraphRetriever:
    """
    Builds a bipartite entity-passage graph on the fly for each question's
    context paragraphs and runs Personalized PageRank to rank passages.
    
    When use_regex=True, applies coreference-aware alias expansion: entity
    mentions in the query are expanded via substring matching and common
    abbreviation patterns, boosting seed coverage for the PPR teleport vector.
    This is the core BCT (Bipartite Coreference Teleportation) contribution.
    """

    def __init__(self, docs: List[SimpleDocument]):
        self.docs = docs
        # Extract entities: capitalised word sequences, compound nouns, and
        # parenthetical aliases (common in Wikipedia-style text)
        self.entities: List[str] = []
        entity_set = set()
        for d in docs:
            # Multi-word proper nouns
            for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', d.page_content):
                ent = m.group(0)
                if len(ent) > 2 and ent.lower() not in {
                    'the', 'and', 'for', 'was', 'with', 'are', 'has', 'had',
                    'his', 'her', 'its', 'they', 'this', 'that', 'from',
                    'also', 'been', 'were', 'which', 'their', 'have', 'she',
                    'who', 'not', 'but', 'all', 'can', 'one', 'two', 'may'
                }:
                    if ent not in entity_set:
                        entity_set.add(ent)
                        self.entities.append(ent)

        np_ = len(self.docs)
        ne = len(self.entities)
        total = np_ + ne
        self.adj = np.zeros((total, total))

        for pi, d in enumerate(self.docs):
            content_lower = d.page_content.lower()
            for ei, ent in enumerate(self.entities):
                if ent.lower() in content_lower:
                    self.adj[pi, np_ + ei] = 1.0
                    self.adj[np_ + ei, pi] = 1.0

        self.np_ = np_
        self.ne = ne
        self.total = total
        
        # Precompute entity alias map for regex mode
        self._alias_map = {}
        for ei, ent in enumerate(self.entities):
            key = ent.lower()
            self._alias_map[key] = ei
            # Add surname alias only for multi-word entities where surname ≥4 chars
            words = ent.split()
            if len(words) > 1 and len(words[-1]) >= 4:
                self._alias_map[words[-1].lower()] = ei

    def query(self, q: str, k: int = 5, use_regex: bool = False) -> List[int]:
        if self.ne == 0 or self.total == 0:
            return list(range(min(k, len(self.docs))))

        q_lower = q.lower()
        seeds = []

        if use_regex:
            # BCT: Coreference-aware alias expansion
            # First pass: exact entity matches
            for ei, ent in enumerate(self.entities):
                if ent.lower() in q_lower:
                    seeds.append(ei)

            # Second pass: alias/abbreviation matching via the alias map
            q_words = re.findall(r'[a-z]{3,}', q_lower)
            for w in q_words:
                if w in self._alias_map:
                    ei = self._alias_map[w]
                    if ei not in seeds:
                        seeds.append(ei)

            # Third pass: strong substring overlap (≥2 matching words of length ≥4)
            if len(seeds) < 2:
                for ei, ent in enumerate(self.entities):
                    ent_words = set(re.findall(r'[a-z]{4,}', ent.lower()))
                    q_word_set = set(re.findall(r'[a-z]{4,}', q_lower))
                    overlap = ent_words & q_word_set
                    if len(overlap) >= 2 and ei not in seeds:
                        seeds.append(ei)
        else:
            # Standard exact matching (exact bipartite PPR graph baseline)
            for ei, ent in enumerate(self.entities):
                if ent.lower() in q_lower:
                    seeds.append(ei)

            if not seeds:
                q_words = set(re.findall(r'[a-z]{3,}', q_lower))
                for ei, ent in enumerate(self.entities):
                    ent_words = set(re.findall(r'[a-z]{3,}', ent.lower()))
                    if q_words & ent_words:
                        seeds.append(ei)

        if not seeds:
            return list(range(min(k, len(self.docs))))

        # Weighted teleport vector: in regex mode, prioritize entities
        # that have more alias matches (stronger coreference signal)
        u = np.zeros(self.total)
        if use_regex and len(seeds) > 1:
            # Weight seeds by alias match strength
            weights = {}
            q_words = set(re.findall(r'[a-z]{3,}', q_lower))
            for s in seeds:
                ent = self.entities[s]
                ent_words = set(re.findall(r'[a-z]{3,}', ent.lower()))
                w = len(ent_words & q_words) + 1  # at least 1
                weights[s] = w
            total_w = sum(weights.values())
            for s, w in weights.items():
                u[self.np_ + s] = w / total_w
        else:
            for s in seeds:
                u[self.np_ + s] = 1.0 / len(seeds)

        # Row-normalise adjacency
        transition = np.zeros_like(self.adj)
        for i in range(self.total):
            rs = np.sum(self.adj[i])
            if rs > 0:
                transition[i] = self.adj[i] / rs

        # Power iteration PPR
        # NOTE: alpha here = propagation weight = 1 - teleport.
        # Paper's α (Eq. 4) = teleport probability = 0.15.
        alpha = 0.85
        v = np.copy(u)
        for _ in range(20):  # fixed iteration cap; see paper §4.4
            v = alpha * transition.T @ v + (1 - alpha) * u

        passage_scores = v[:self.np_]
        ranked = np.argsort(passage_scores)[::-1]
        return [int(idx) for idx in ranked[:k]]


# ---------------------------------------------------------------------------
# Retrieval Token Overlap (RTO)
# Token-level F1 between concatenated retrieved passages and the gold answer.
# This is NOT a generated-answer F1 — see paper §5.1 for definition.
# ---------------------------------------------------------------------------
def _token_rto(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold strings.

    Called 'Retrieval Token Overlap (RTO)' in the paper because the
    prediction is the concatenation of retrieved passages, not a
    generated answer.  Precision is inherently low by construction.
    """
    pred_toks = prediction.lower().split()
    gold_toks = gold.lower().split()
    if not gold_toks:
        return 1.0 if not pred_toks else 0.0
    if not pred_toks:
        return 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Paired bootstrap for F1 significance
# ---------------------------------------------------------------------------
def _bootstrap_rto_diff(
    rto_a: List[float],
    rto_b: List[float],
    B: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Paired bootstrap resampling of mean(rto_b) - mean(rto_a).
    Inputs are per-question Retrieval Token Overlap (RTO) in 0-100 scale.
    RTO measures token overlap between concatenated retrieved passages and
    the gold answer — it is NOT a generated-answer F1.
    Returns observed delta, 95% CI (lo, hi), and one-tailed p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.array(rto_a)
    b = np.array(rto_b)
    n = len(a)
    observed = b.mean() - a.mean()          # already in RTO-point (0-100) scale
    boot_deltas = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = b[idx].mean() - a[idx].mean()
    lo = float(np.percentile(boot_deltas, 2.5))
    hi = float(np.percentile(boot_deltas, 97.5))
    p_value = float(np.mean(boot_deltas <= 0))  # one-tailed: P(delta <= 0)
    return {
        'observed': round(float(observed), 2),  # already in RTO-point scale
        'ci_lo':    round(lo, 2),
        'ci_hi':    round(hi, 2),
        'p_value':  round(p_value, 4),
        'B':        B,
    }


CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'eval_cache.json')


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(data: dict):
    with open(CACHE_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def _run_eval_loop(items, build_docs_fn, methods_config, n_questions):
    """
    Generic evaluation loop shared by 2Wiki and MuSiQue evaluators.

    build_docs_fn(item) -> (docs: List[SimpleDocument], gold_indices: Set[int], answer: str|None)
    Returns per-method accumulators including a per-question rto_list for bootstrapping.
    """
    methods = {}
    for m in methods_config:
        methods[m] = {'hr1': 0, 'hr3': 0, 'hr5': 0, 'mrr': 0.0, 'rto_sum': 0.0, 'rto_list': []}

    for qi, item in enumerate(items):
        if qi % 50 == 0 and qi > 0:
            print(f"    ...processed {qi}/{len(items)} questions")

        question = item['question']
        docs, gold_indices, answer = build_docs_fn(item)

        if not gold_indices or not docs:
            continue

        sparse = _SparseRetriever(docs)
        dense = _DenseRetriever(docs)
        graph = _GraphRetriever(docs)

        for method_name in methods_config:
            if method_name == 'tfidf':
                retrieved = sparse.query(question, k=5)
            elif method_name == 'dense':
                retrieved = dense.query(question, k=5)
            elif method_name == 'graph':
                retrieved = graph.query(question, k=5, use_regex=False)
            elif method_name == 'graph_regex':
                retrieved = graph.query(question, k=5, use_regex=True)
            else:
                retrieved = sparse.query(question, k=5)

            # HR@k and MRR
            for rank, ridx in enumerate(retrieved):
                if ridx in gold_indices:
                    if rank == 0:
                        methods[method_name]['hr1'] += 1
                    if rank < 3:
                        methods[method_name]['hr3'] += 1
                    if rank < 5:
                        methods[method_name]['hr5'] += 1
                    methods[method_name]['mrr'] += 1.0 / (rank + 1)
                    break

            # Retrieval Token Overlap (RTO): token-level F1 between the
            # concatenated retrieved passages and the gold answer string.
            # NOT a generated-answer F1 — precision is tiny because the
            # "prediction" is hundreds of passage tokens vs a short answer.
            if answer is not None:
                retrieved_text = ' '.join(docs[ridx].page_content for ridx in retrieved)
                rto = _token_rto(retrieved_text, answer)
                methods[method_name]['rto_sum'] += rto
                methods[method_name]['rto_list'].append(rto)   # keep per-question score

    return methods


# ---------------------------------------------------------------------------
# 2WikiMultiHopQA
# ---------------------------------------------------------------------------
def evaluate_2wiki(n_questions: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Run real retrieval on 2WikiMultiHopQA validation set."""
    cache = _load_cache()
    cache_key = f"2wiki_v3_n{n_questions}_s{seed}"   # v3: add dense bi-encoder baseline
    if cache_key in cache:
        print(f"  [CACHE HIT] Loading 2WikiMultiHopQA results from eval_cache.json")
        return cache[cache_key]

    from datasets import load_dataset

    print(f"  Downloading 2WikiMultiHopQA validation set (streaming)...")
    ds = load_dataset(
        'scholarly-shadows-syndicate/2wikimultihopqa',
        split='validation', streaming=True
    )

    pool = []
    for item in ds:
        pool.append(item)
        if len(pool) >= n_questions * 5:
            break

    rng = random.Random(seed)
    rng.shuffle(pool)
    items = pool[:n_questions]
    print(f"  Evaluating {len(items)} questions...")

    def build_docs(item):
        ctx = json.loads(item['context']) if isinstance(item['context'], str) else item['context']
        sf_raw = json.loads(item['supporting_facts']) if isinstance(item['supporting_facts'], str) else item['supporting_facts']

        docs = []
        title_to_idx = {}
        for pi, para in enumerate(ctx):
            title = para[0]
            sents = para[1]
            text = f"{title}: {' '.join(sents)}"
            docs.append(SimpleDocument(page_content=text, metadata={'title': title, 'idx': pi}))
            if title not in title_to_idx:
                title_to_idx[title] = []
            title_to_idx[title].append(pi)

        gold = set()
        for sf in sf_raw:
            if sf[0] in title_to_idx:
                for idx in title_to_idx[sf[0]]:
                    gold.add(idx)

        return docs, gold, None  # No answer F1 for 2Wiki (we use Recall)

    methods_cfg = ['tfidf', 'dense', 'graph', 'graph_regex']
    raw = _run_eval_loop(items, build_docs, methods_cfg, n_questions)

    n = len(items)
    results = {}
    for m in methods_cfg:
        results[m] = {
            'hr1': round(raw[m]['hr1'] / n, 4),
            'hr3': round(raw[m]['hr3'] / n, 4),
            'hr5': round(raw[m]['hr5'] / n, 4),
            'mrr': round(raw[m]['mrr'] / n, 4),
        }
    results['n_questions'] = n

    cache[cache_key] = results
    _save_cache(cache)
    return results


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------
def evaluate_musique(n_questions: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Run real retrieval on MuSiQue validation set (answerable only).
    
    Cache key v3: includes per-question F1 lists and bootstrap CI.
    """
    cache = _load_cache()
    cache_key = f"musique_v9_n{n_questions}_s{seed}"   # v9: add dense bi-encoder baseline
    if cache_key in cache:
        print(f"  [CACHE HIT] Loading MuSiQue results from eval_cache.json")
        return cache[cache_key]

    from datasets import load_dataset

    print(f"  Downloading MuSiQue validation set (streaming)...")
    ds = load_dataset('bdsaglam/musique', split='validation', streaming=True)

    pool = []
    for item in ds:
        if item.get('answerable', True):
            pool.append(item)
        if len(pool) >= n_questions * 5:
            break

    rng = random.Random(seed)
    rng.shuffle(pool)
    items = pool[:n_questions]
    print(f"  Evaluating {len(items)} answerable questions...")

    def build_docs(item):
        paragraphs = item['paragraphs']
        docs = []
        gold = set()
        for para in paragraphs:
            text = f"{para['title']}: {para['paragraph_text']}"
            doc_idx = len(docs)
            docs.append(SimpleDocument(page_content=text, metadata={'idx': para['idx'], 'title': para['title']}))
            if para.get('is_supporting', False):
                gold.add(doc_idx)
        return docs, gold, item['answer']

    methods_cfg = ['tfidf', 'dense', 'graph', 'graph_regex']
    raw = _run_eval_loop(items, build_docs, methods_cfg, n_questions)

    n = len(items)
    results = {}
    for m in methods_cfg:
        results[m] = {
            'hr1': round(raw[m]['hr1'] / n, 4),
            'hr3': round(raw[m]['hr3'] / n, 4),
            'hr5': round(raw[m]['hr5'] / n, 4),
            'mrr': round(raw[m]['mrr'] / n, 4),
            'rto':  round(raw[m]['rto_sum'] / n * 100, 2),  # Retrieval Token Overlap (%)
            'rto_list': [round(x * 100, 4) for x in raw[m]['rto_list']],  # 0-100 scale
        }
    results['n_questions'] = n

    # Paired bootstrap: Aethel (PPR+BCT) vs Bipartite PPR on per-question RTO
    print(f"  Running paired bootstrap (B=1000) on MuSiQue RTO...")
    bootstrap = _bootstrap_rto_diff(
        rto_a=results['graph']['rto_list'],        # 0-100 scale
        rto_b=results['graph_regex']['rto_list'],  # 0-100 scale
        B=1000,
        seed=42,
    )
    results['bootstrap_graph_vs_regex'] = bootstrap
    print(f"  Bootstrap result: delta={bootstrap['observed']:+.2f} RTO pts, "
          f"95% CI=[{bootstrap['ci_lo']:.2f}, {bootstrap['ci_hi']:.2f}], "
          f"p={bootstrap['p_value']:.4f}")

    cache[cache_key] = results
    _save_cache(cache)
    return results


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------
def run_public_benchmarks(n_questions: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Run both benchmarks and return combined results."""
 
    print("  PUBLIC BENCHMARK EVALUATION (HuggingFace Datasets)")
 

    print(f"\n[1/2] 2WikiMultiHopQA (n={n_questions}, seed={seed})")
    wiki_results = evaluate_2wiki(n_questions=n_questions, seed=seed)

    print(f"\n[2/2] MuSiQue (n={n_questions}, seed={seed})")
    musique_results = evaluate_musique(n_questions=n_questions, seed=seed)

    combined = {
        '2wiki': wiki_results,
        'musique': musique_results,
    }

  

    METHOD_LABELS = {
        'tfidf': 'Sparse (TF-IDF)',
        'dense': 'Dense (MiniLM)',
        'graph': 'Graph (PPR)',
        'graph_regex': 'Aethel (PPR+BCT)'
    }
    method_order = ['tfidf', 'dense', 'graph', 'graph_regex']

    print(f"\n  2WikiMultiHopQA ({wiki_results['n_questions']} questions, ~10 paragraphs/question):")
    print(f"  {'Method':<24} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'MRR':>8}")
    for m in method_order:
        r = wiki_results[m]
        print(f"  {METHOD_LABELS[m]:<24} {r['hr1']:>8.4f} {r['hr3']:>8.4f} {r['hr5']:>8.4f} {r['mrr']:>8.4f}")

    print(f"\n  MuSiQue ({musique_results['n_questions']} questions, ~20 paragraphs/question):")
    print(f"  {'Method':<24} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'MRR':>8} {'RTO%':>8}")
    for m in method_order:
        r = musique_results[m]
        print(f"  {METHOD_LABELS[m]:<24} {r['hr1']:>8.4f} {r['hr3']:>8.4f} {r['hr5']:>8.4f} {r['mrr']:>8.4f} {r['rto']:>8.2f}")

 
    return combined


if __name__ == '__main__':
    run_public_benchmarks(n_questions=200, seed=42)
