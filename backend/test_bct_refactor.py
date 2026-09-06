"""
Differential test: the flag-based _GraphRetriever.query refactor must be
behaviour-preserving for the two historically shipped configurations.

Compares the refactored implementation against the pre-refactor version
(extracted from git) on real 2WikiMultiHopQA and MuSiQue items:

  new(use_regex=False) == original(use_regex=False)   [Graph (PPR)]
  new(use_regex=True)  == original(use_regex=True)     [Aethel (PPR+BCT)]
  new(bct_none rung)   == new(use_regex=False)         [ablation rung 0 sanity]

Run:  PYTHONPATH=. python3 backend/test_bct_refactor.py [--orig /tmp/orig_pb.py]
"""

import argparse
import importlib.util
import os
import json
import random
import sys

from backend.public_benchmark import (
    SimpleDocument, _GraphRetriever, GRAPH_METHODS,
)


def _load_original(path):
    if not os.path.exists(path):
        sys.exit(
            f"Reference implementation not found at {path}.\n"
            "Regenerate it from the commit before the flag refactor, e.g.:\n"
            "  git show <pre-refactor-rev>:backend/public_benchmark.py > /tmp/orig_pb.py\n"
            "then re-run with --orig /tmp/orig_pb.py"
        )
    spec = importlib.util.spec_from_file_location("orig_pb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_2wiki_docs(item):
    ctx = json.loads(item['context']) if isinstance(item['context'], str) else item['context']
    docs = []
    for pi, para in enumerate(ctx):
        text = f"{para[0]}: {' '.join(para[1])}"
        docs.append(SimpleDocument(page_content=text, metadata={'idx': pi}))
    return docs


def _build_musique_docs(item):
    return [
        SimpleDocument(page_content=f"{p['title']}: {p['paragraph_text']}",
                       metadata={'idx': p['idx']})
        for p in item['paragraphs']
    ]


def _sample(dataset_name, split_kwargs, builder, n, seed=42):
    from datasets import load_dataset
    ds = load_dataset(**split_kwargs, streaming=True)
    pool = []
    for item in ds:
        pool.append(item)
        if len(pool) >= n * 5:
            break
    rng = random.Random(seed)
    rng.shuffle(pool)
    return [(item['question'], builder(item)) for item in pool[:n]]


def run_dataset(label, cases, orig_mod):
    failures = []
    for qi, (question, docs) in enumerate(cases):
        new_g = _GraphRetriever(docs)
        old_g = orig_mod._GraphRetriever(docs)

        # Guard: identical graph construction is a precondition for the
        # query comparison to mean anything.
        if new_g.entities != old_g.entities:
            failures.append((qi, 'entities', new_g.entities[:5], old_g.entities[:5]))
            continue

        for use_regex in (False, True):
            new_r = new_g.query(question, k=5, use_regex=use_regex)
            old_r = old_g.query(question, k=5, use_regex=use_regex)
            if new_r != old_r:
                failures.append((qi, f'use_regex={use_regex}', new_r, old_r))

        # Ablation rung 0 must reproduce the exact-match baseline.
        rung0 = new_g.query(question, k=5, **GRAPH_METHODS['bct_none'])
        base = new_g.query(question, k=5, use_regex=False)
        if rung0 != base:
            failures.append((qi, 'bct_none vs graph', rung0, base))

    status = "PASS" if not failures else "FAIL"
    print(f"  [{status}] {label}: {len(cases)} questions, {len(failures)} mismatches")
    for qi, what, a, b in failures[:5]:
        print(f"      q{qi} {what}: new={a} old={b}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', default='/tmp/orig_pb.py',
                    help='pre-refactor public_benchmark.py (git show HEAD:...)')
    ap.add_argument('-n', type=int, default=60, help='questions per dataset')
    args = ap.parse_args()

    orig_mod = _load_original(args.orig)
    print(f"Differential test vs {args.orig}\n")

    all_failures = []
    print("Sampling 2WikiMultiHopQA...")
    cases = _sample(
        '2wiki',
        dict(path='scholarly-shadows-syndicate/2wikimultihopqa', split='validation'),
        _build_2wiki_docs, args.n)
    all_failures += run_dataset('2WikiMultiHopQA', cases, orig_mod)

    print("Sampling MuSiQue...")
    cases = _sample(
        'musique',
        dict(path='bdsaglam/musique', split='validation'),
        _build_musique_docs, args.n)
    all_failures += run_dataset('MuSiQue', cases, orig_mod)

    if all_failures:
        print(f"\nREFACTOR IS NOT BEHAVIOUR-PRESERVING — {len(all_failures)} mismatches")
        sys.exit(1)
    print("\nAll configurations reproduce the pre-refactor implementation exactly.")


if __name__ == '__main__':
    main()
