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
# Dense Bi-Encoder Retrievers (semantic baselines)
#
# Registry of bi-encoders. `all-MiniLM-L6-v2` (22M params, 2021) is retained as
# a legacy reference point; the three base-size encoders (109M params) are the
# contemporary baselines.
#
# The query/document prefixes are NOT cosmetic. BGE and E5 are trained with
# asymmetric instruction prefixes, and omitting them materially understates
# those models. GTE and MiniLM are trained without prefixes.
# ---------------------------------------------------------------------------
DENSE_MODELS: Dict[str, Dict[str, str]] = {
    'dense': {
        'model_name': 'all-MiniLM-L6-v2',
        'label':      'Dense (MiniLM)',
        'params':     '22M',
        'query_prefix': '',
        'doc_prefix':   '',
    },
    'dense_bge': {
        'model_name': 'BAAI/bge-base-en-v1.5',
        'label':      'Dense (BGE-base)',
        'params':     '109M',
        'query_prefix': 'Represent this sentence for searching relevant passages: ',
        'doc_prefix':   '',
    },
    'dense_e5': {
        'model_name': 'intfloat/e5-base-v2',
        'label':      'Dense (E5-base)',
        'params':     '109M',
        'query_prefix': 'query: ',
        'doc_prefix':   'passage: ',
    },
    'dense_gte': {
        'model_name': 'thenlper/gte-base',
        'label':      'Dense (GTE-base)',
        'params':     '109M',
        'query_prefix': '',
        'doc_prefix':   '',
    },
}

DENSE_KEYS = list(DENSE_MODELS.keys())

_DENSE_MODEL_CACHE: Dict[str, Any] = {}  # model_key -> SentenceTransformer


def _get_dense_model(model_key: str = 'dense'):
    """Lazy-load and cache a SentenceTransformer by registry key."""
    if model_key not in DENSE_MODELS:
        raise KeyError(
            f"Unknown dense model key {model_key!r}. "
            f"Known keys: {sorted(DENSE_MODELS)}"
        )
    if model_key not in _DENSE_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        # AETHEL_DENSE_DEVICE lets a memory-constrained host force 'cpu'.
        # On small unified-memory Macs, MPS allocations compete with system
        # RAM and can push the process into swap thrash.
        device = os.environ.get('AETHEL_DENSE_DEVICE') or None
        _DENSE_MODEL_CACHE[model_key] = SentenceTransformer(
            DENSE_MODELS[model_key]['model_name'], device=device
        )
    return _DENSE_MODEL_CACHE[model_key]


# Encoding batch size. Lower it via AETHEL_DENSE_BATCH on low-RAM machines:
# activation memory scales with batch size and dominates weight memory.
DENSE_BATCH_SIZE = int(os.environ.get('AETHEL_DENSE_BATCH', '32'))


def _release_dense_model(model_key: str) -> None:
    """Drop a cached encoder and reclaim its memory.

    Holding all four encoders resident at once costs well over a gigabyte,
    which is enough to push a memory-constrained machine into swap thrash.
    The benchmark evaluates one encoder at a time and releases it here.
    """
    _DENSE_MODEL_CACHE.pop(model_key, None)
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _DenseRetriever:
    """Dense bi-encoder retriever.

    Defaults to `all-MiniLM-L6-v2` so existing callers (e.g. evaluate_aethel.py)
    keep their previous behaviour unchanged.
    """

    def __init__(self, docs: List[SimpleDocument], model_key: str = 'dense'):
        self.docs = docs
        self.model_key = model_key
        spec = DENSE_MODELS[model_key]
        self._query_prefix = spec['query_prefix']
        model = _get_dense_model(model_key)
        doc_prefix = spec['doc_prefix']
        self.embeddings = model.encode(
            [doc_prefix + d.page_content for d in docs],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=DENSE_BATCH_SIZE,
        )

    def query(self, q: str, k: int = 5) -> List[int]:
        model = _get_dense_model(self.model_key)
        q_emb = model.encode(
            [self._query_prefix + q], normalize_embeddings=True
        )[0]
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

    def query(
        self,
        q: str,
        k: int = 5,
        use_regex: bool = False,
        alias: bool = None,
        substring: bool = None,
        weighted: bool = None,
        exact_fallback: bool = None,
        return_diagnostics: bool = False,
    ):
        """Rank passages by Personalized PageRank over the bipartite graph.

        BCT decomposes into three independent mechanisms plus one historical
        asymmetry, each exposed as its own flag so they can be ablated:

          alias          — expand seeds via the surname/abbreviation alias map
          substring      — seed on strong substring overlap when <2 seeds found
          weighted       — weight the teleport vector by query-word overlap
                           instead of distributing mass uniformly
          exact_fallback — when NO seeds are found, fall back to loose (>=3 char)
                           word-overlap seeding rather than returning the first
                           k passages in document order

        `use_regex` is retained as the historical entry point. When the four
        flags are left unset they are derived from it so that the two shipped
        configurations reproduce exactly:

          use_regex=False -> alias=F, substring=F, weighted=F, exact_fallback=T
          use_regex=True  -> alias=T, substring=T, weighted=T, exact_fallback=F

        Note the asymmetry in the last column: the original BCT path never had
        the no-seed fallback that the exact path had. That difference was
        incidental rather than designed, so it is ablated separately.
        """
        if alias is None:
            alias = use_regex
        if substring is None:
            substring = use_regex
        if weighted is None:
            weighted = use_regex
        if exact_fallback is None:
            exact_fallback = not use_regex

        diag = {'n_seeds': 0, 'zero_seed': False, 'fallback_used': False}

        def _finish(result, seeds_count=0, zero_seed=False, fallback_used=False):
            diag['n_seeds'] = seeds_count
            diag['zero_seed'] = zero_seed
            diag['fallback_used'] = fallback_used
            return (result, diag) if return_diagnostics else result

        if self.ne == 0 or self.total == 0:
            return _finish(list(range(min(k, len(self.docs)))), 0, True, False)

        q_lower = q.lower()
        seeds = []

        # Pass 1 (always): exact entity surface-form match
        for ei, ent in enumerate(self.entities):
            if ent.lower() in q_lower:
                seeds.append(ei)

        # Pass 2 (alias): alias/abbreviation matching via the alias map
        if alias:
            q_words = re.findall(r'[a-z]{3,}', q_lower)
            for w in q_words:
                if w in self._alias_map:
                    ei = self._alias_map[w]
                    if ei not in seeds:
                        seeds.append(ei)

        # Pass 3 (substring): strong overlap (>=2 matching words of length >=4)
        if substring and len(seeds) < 2:
            q_word_set = set(re.findall(r'[a-z]{4,}', q_lower))
            for ei, ent in enumerate(self.entities):
                ent_words = set(re.findall(r'[a-z]{4,}', ent.lower()))
                if len(ent_words & q_word_set) >= 2 and ei not in seeds:
                    seeds.append(ei)

        # Pass 4 (exact_fallback): loose word-overlap rescue when nothing hit
        fallback_used = False
        if exact_fallback and not seeds:
            q_words = set(re.findall(r'[a-z]{3,}', q_lower))
            for ei, ent in enumerate(self.entities):
                ent_words = set(re.findall(r'[a-z]{3,}', ent.lower()))
                if q_words & ent_words:
                    seeds.append(ei)
            fallback_used = bool(seeds)

        if not seeds:
            return _finish(
                list(range(min(k, len(self.docs)))), 0, True, fallback_used
            )

        # Weighted teleport vector: prioritize entities with more query-word
        # overlap (stronger coreference signal)
        u = np.zeros(self.total)
        if weighted and len(seeds) > 1:
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
        return _finish(
            [int(idx) for idx in ranked[:k]],
            seeds_count=len(seeds),
            zero_seed=False,
            fallback_used=fallback_used,
        )


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


# ---------------------------------------------------------------------------
# Method ordering and display labels
# ---------------------------------------------------------------------------
BASELINE_METHODS = ['tfidf'] + DENSE_KEYS + ['graph', 'graph_regex']
ABLATION_METHODS = ['bct_none', 'bct_alias', 'bct_alias_sub', 'bct_full']
ALL_METHODS = BASELINE_METHODS + ABLATION_METHODS

METHOD_LABELS = {
    'tfidf':         'Sparse (TF-IDF)',
    'graph':         'Graph (PPR)',
    'graph_regex':   'Aethel (PPR+BCT)',
    'bct_none':      'BCT-0: exact only',
    'bct_alias':     'BCT-1: +alias',
    'bct_alias_sub': 'BCT-2: +substring',
    'bct_full':      'BCT-3: +weighted',
}
METHOD_LABELS.update({k: v['label'] for k, v in DENSE_MODELS.items()})


def _attach_seed_diagnostics(out: dict, raw_m: dict) -> None:
    """Copy graph seeding diagnostics onto a results dict, if any were recorded."""
    gq = raw_m.get('graph_questions', 0)
    if not gq:
        return
    out['mean_seeds'] = round(raw_m['seed_sum'] / gq, 2)
    out['zero_seed_questions'] = raw_m['zero_seed']
    out['fallback_questions'] = raw_m['fallback_used']


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


# ---------------------------------------------------------------------------
# Graph method registry — the BCT ablation ladder
#
# `graph` and `graph_regex` are the two historically shipped configurations and
# are preserved bit-for-bit. The `bct_*` rungs add exactly one mechanism each,
# holding exact_fallback=True throughout so that the ladder isolates the three
# designed BCT mechanisms. Comparing `bct_full` against `graph_regex` then
# isolates the incidental no-seed-fallback asymmetry on its own.
# ---------------------------------------------------------------------------
GRAPH_METHODS: Dict[str, Dict[str, bool]] = {
    'graph':         dict(use_regex=False),
    'graph_regex':   dict(use_regex=True),
    'bct_none':      dict(alias=False, substring=False, weighted=False, exact_fallback=True),
    'bct_alias':     dict(alias=True,  substring=False, weighted=False, exact_fallback=True),
    'bct_alias_sub': dict(alias=True,  substring=True,  weighted=False, exact_fallback=True),
    'bct_full':      dict(alias=True,  substring=True,  weighted=True,  exact_fallback=True),
}


def _run_eval_loop(items, build_docs_fn, methods_config, n_questions):
    """
    Generic evaluation loop shared by 2Wiki and MuSiQue evaluators.

    build_docs_fn(item) -> (docs: List[SimpleDocument], gold_indices: Set[int], answer: str|None)
    Returns per-method accumulators including a per-question rto_list for bootstrapping.

    Evaluation is split into passes so that only ONE dense encoder is resident
    at a time: the lexical/graph methods share a pass (they can reuse the same
    per-question sparse and graph indices), then each encoder gets its own pass
    and is released afterwards. Holding all four encoders simultaneously costs
    over a gigabyte and can push a constrained machine into swap thrash.
    """
    methods = {}
    for m in methods_config:
        methods[m] = {
            'hr1': 0, 'hr3': 0, 'hr5': 0, 'mrr': 0.0,
            'rto_sum': 0.0, 'rto_list': [],
            'seed_sum': 0, 'zero_seed': 0, 'fallback_used': 0, 'graph_questions': 0,
        }

    lexical_graph = [m for m in methods_config
                     if m == 'tfidf' or m in GRAPH_METHODS]
    dense_methods = [m for m in methods_config if m in DENSE_MODELS]
    other = [m for m in methods_config
             if m not in lexical_graph and m not in dense_methods]

    passes = []
    if lexical_graph or other:
        passes.append(('lexical+graph', lexical_graph + other))
    for m in dense_methods:
        passes.append((DENSE_MODELS[m]['label'], [m]))

    for pass_label, pass_methods in passes:
        print(f"    [pass] {pass_label}: {', '.join(pass_methods)}")
        _eval_pass(items, build_docs_fn, pass_methods, methods)
        for m in pass_methods:
            if m in DENSE_MODELS:
                _release_dense_model(m)

    return methods


def _eval_pass(items, build_docs_fn, methods_config, methods):
    """Score one group of methods over every question, accumulating into `methods`."""
    for qi, item in enumerate(items):
        if qi % 50 == 0 and qi > 0:
            print(f"    ...processed {qi}/{len(items)} questions")

        question = item['question']
        docs, gold_indices, answer = build_docs_fn(item)

        if not gold_indices or not docs:
            continue

        sparse = None
        graph = None
        dense_by_key = {}

        for method_name in methods_config:
            if method_name == 'tfidf':
                if sparse is None:
                    sparse = _SparseRetriever(docs)
                retrieved = sparse.query(question, k=5)
            elif method_name in DENSE_MODELS:
                if method_name not in dense_by_key:
                    dense_by_key[method_name] = _DenseRetriever(docs, model_key=method_name)
                retrieved = dense_by_key[method_name].query(question, k=5)
            elif method_name in GRAPH_METHODS:
                if graph is None:
                    graph = _GraphRetriever(docs)
                retrieved, diag = graph.query(
                    question, k=5, return_diagnostics=True,
                    **GRAPH_METHODS[method_name]
                )
                acc = methods[method_name]
                acc['seed_sum'] += diag['n_seeds']
                acc['zero_seed'] += int(diag['zero_seed'])
                acc['fallback_used'] += int(diag['fallback_used'])
                acc['graph_questions'] += 1
            else:
                if sparse is None:
                    sparse = _SparseRetriever(docs)
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


# ---------------------------------------------------------------------------
# 2WikiMultiHopQA
# ---------------------------------------------------------------------------
def evaluate_2wiki(n_questions: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Run real retrieval on 2WikiMultiHopQA validation set."""
    cache = _load_cache()
    # v4: add BGE/E5/GTE dense baselines and the BCT ablation ladder
    cache_key = f"2wiki_v4_n{n_questions}_s{seed}"
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

        return docs, gold, None  # No answer F1 for 2Wiki (we use Hit Rate)

    methods_cfg = ALL_METHODS
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
        _attach_seed_diagnostics(results[m], raw[m])
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
    # v10: add BGE/E5/GTE dense baselines and the BCT ablation ladder
    cache_key = f"musique_v10_n{n_questions}_s{seed}"
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

    methods_cfg = ALL_METHODS
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
        _attach_seed_diagnostics(results[m], raw[m])
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

  

    def _table(title, res, methods, show_rto):
        print(f"\n  {title}")
        hdr = f"  {'Method':<24} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'MRR':>8}"
        if show_rto:
            hdr += f" {'RTO%':>8}"
        print(hdr)
        for m in methods:
            r = res[m]
            row = (f"  {METHOD_LABELS[m]:<24} {r['hr1']:>8.4f} {r['hr3']:>8.4f} "
                   f"{r['hr5']:>8.4f} {r['mrr']:>8.4f}")
            if show_rto:
                row += f" {r['rto']:>8.2f}"
            print(row)

    def _diag_table(title, res):
        print(f"\n  {title}")
        print(f"  {'Config':<24} {'mean seeds':>12} {'zero-seed q':>13} {'fallback q':>12}")
        for m in ABLATION_METHODS + ['graph_regex']:
            r = res[m]
            if 'mean_seeds' not in r:
                continue
            print(f"  {METHOD_LABELS[m]:<24} {r['mean_seeds']:>12.2f} "
                  f"{r['zero_seed_questions']:>13d} {r['fallback_questions']:>12d}")

    _table(
        f"2WikiMultiHopQA ({wiki_results['n_questions']} questions, ~10 paragraphs/question):",
        wiki_results, BASELINE_METHODS, show_rto=False)
    _table(
        f"MuSiQue ({musique_results['n_questions']} questions, ~20 paragraphs/question):",
        musique_results, BASELINE_METHODS, show_rto=True)

    print("\n" + "=" * 78)
    print("  BCT ABLATION (each rung adds exactly one mechanism)")
    print("=" * 78)
    _table("2WikiMultiHopQA:", wiki_results, ABLATION_METHODS + ['graph_regex'], show_rto=False)
    _table("MuSiQue:", musique_results, ABLATION_METHODS + ['graph_regex'], show_rto=True)

    _diag_table("2Wiki seeding diagnostics:", wiki_results)
    _diag_table("MuSiQue seeding diagnostics:", musique_results)

    return combined


if __name__ == '__main__':
    run_public_benchmarks(n_questions=200, seed=42)
