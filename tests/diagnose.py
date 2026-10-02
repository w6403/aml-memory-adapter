# -*- coding: utf-8 -*-
"""失败诊断：把 single-hop/multi-hop 的 miss 分类

对每个 miss 的 QA，计算：
  - overlap: 问题关键词与任一证据记忆关键词是否有重合（区分 paraphrase vs 排序）
  - best_rank: 证据在完整打分榜里的最好名次（判断是 top-k 截断还是打分失效）
输出分类统计 + 每类 3 个样例。

用法：python tests/diagnose.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core import MemoryEngine, EmbeddingClient
from memory_core.engine import extract_keywords, _cosine
from memory_core.models import MemoryRecord
from sqlalchemy import select
from sqlalchemy.orm import Session
from local_eval import load_conversations, ingest, CATEGORY_NAMES

TOP_K = 20


def full_rank_search(engine: MemoryEngine, user_id: str, query: str):
    """用 engine.search 全量返回（大 top_k），避免复制打分逻辑。"""
    hits = engine.search(user_id, query, top_k=100000)
    q_kws = set(w.lower() for w in extract_keywords(query))
    scored = []  # (rank, content, overlap)
    for i, h in enumerate(hits, 1):
        r_kws = set(w.lower() for w in (h.get('keywords') or []))
        if h.get('speaker'):
            r_kws.add(h['speaker'].lower())
        scored.append((i, h['content'], len(q_kws & r_kws)))
    return scored, q_kws


def main():
    db = os.path.join(tempfile.gettempdir(), 'aml_locomo_diag.db')
    if os.path.exists(db):
        os.remove(db)
    engine = MemoryEngine(f'sqlite:///{db}', embedding_client=EmbeddingClient(),
                          recency_weight=0.0)

    data = os.path.join(os.path.dirname(__file__), 'locomo10.json')
    convs = load_conversations(data, 10)

    # bucket: (category, kind) -> [count, samples]
    buckets = {}
    for conv in convs:
        user_id = str(conv.get('sample_id', 'unknown'))
        dia_map = ingest(engine, conv)
        for qa in conv['qa']:
            cat = qa['category']
            if cat not in (1, 3):
                continue
            evidence = qa.get('evidence') or []
            evidence_texts = {dia_map[d] for d in evidence if d in dia_map}
            if not evidence_texts:
                continue
            scored, q_kws = full_rank_search(engine, user_id, qa['question'])
            top_contents = {c for _, c, _ in scored[:TOP_K]}
            if top_contents & evidence_texts:
                continue  # hit
            # miss：找证据的最好名次与重合数
            best_rank, best_overlap = None, 0
            for rank, content, ov in scored:
                if content in evidence_texts:
                    if best_rank is None:
                        best_rank = rank
                    best_overlap = max(best_overlap, ov)
            if best_overlap == 0:
                kind = 'zero_overlap'      # paraphrase：关键词完全沾不上
            elif best_rank and best_rank <= TOP_K * 2:
                kind = 'near_miss'          # 有重合但差一点进 top_k
            else:
                kind = 'buried'             # 有重合但排得很靠后
            key = (CATEGORY_NAMES[cat], kind)
            cnt, samples = buckets.setdefault(key, [0, []])
            cnt += 1
            buckets[key][0] += 1
            if len(samples) < 3:
                ev_sample = next(iter(evidence_texts))
                samples.append({
                    'q': qa['question'],
                    'q_kws': sorted(q_kws),
                    'evidence': ev_sample[:160],
                    'ev_kws': extract_keywords(ev_sample),
                    'best_rank': best_rank,
                    'overlap': best_overlap,
                })

    print(f'{"category":<12}{"kind":<14}{"count":>6}')
    print('-' * 32)
    for (cat, kind), (cnt, _) in sorted(buckets.items()):
        print(f'{cat:<12}{kind:<14}{cnt:>6}')
    print()
    for (cat, kind), (cnt, samples) in sorted(buckets.items()):
        print(f'=== {cat} / {kind} ({cnt}) ===')
        for s in samples:
            print(f"  Q: {s['q']}")
            print(f"  q_kws={s['q_kws']}")
            print(f"  EV: {s['evidence']}")
            print(f"  ev_kws={s['ev_kws']}")
            print(f"  best_rank={s['best_rank']} overlap={s['overlap']}")
            print()


if __name__ == '__main__':
    main()
