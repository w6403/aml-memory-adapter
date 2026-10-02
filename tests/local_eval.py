# -*- coding: utf-8 -*-
"""LoCoMo 本地评测脚本

目的：不依赖 AML 平台的 2 次 Full 机会，在本地用公开基准量化检索质量。
指标：Evidence Recall@K —— 对每个带证据的 QA，检索 top_k 条记忆，
      若命中任一证据对话轮次（按原文精确匹配）则记为命中。

数据：tests/locomo10.json（snap-research/locomo，10 段多会话长对话）
用法：
  python tests/local_eval.py                     # 关键词降级模式（离线）
  DASHSCOPE_API_KEY=sk-xxx python tests/local_eval.py   # 向量混合模式
  python tests/local_eval.py --top-k 10 --max-conv 3 --out results.json
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
            override=True)

from memory_core import MemoryEngine, EmbeddingClient

# LoCoMo 论文的 QA 类目（类别 4/5 多数无证据，仅供对照）
CATEGORY_NAMES = {
    1: 'single-hop',
    2: 'temporal',
    3: 'multi-hop',
    4: 'open-domain',
    5: 'adversarial',
}


def parse_session_time(s: str):
    """LoCoMo 会话时间格式：'1:56 pm on 8 May, 2023'"""
    try:
        return datetime.strptime(s, '%I:%M %p on %d %B, %Y')
    except (ValueError, TypeError):
        return None


def load_conversations(path: str, max_conv: int):
    with open(path, encoding='utf-8') as f:
        return json.load(f)[:max_conv]


def ingest(engine: MemoryEngine, conv: dict) -> dict:
    """把一段多会话对话写入引擎，返回 dia_id -> content 映射。"""
    user_id = str(conv.get('sample_id', 'unknown'))
    c = conv['conversation']
    dia_map = {}
    n_sessions = 0
    while f'session_{n_sessions + 1}' in c:
        n_sessions += 1
    for i in range(1, n_sessions + 1):
        turns = c[f'session_{i}']
        event_time = parse_session_time(c.get(f'session_{i}_date_time'))
        messages = []
        for t in turns:
            dia_map[t['dia_id']] = t['text']
            msg = {'speaker': t['speaker'], 'content': t['text']}
            if event_time:
                msg['event_time'] = event_time.isoformat()
            messages.append(msg)
        engine.add(user_id, task_id='', session_id=f'session_{i}', messages=messages)
    return dia_map


def evaluate(engine: MemoryEngine, convs: list, top_k: int):
    stats = {}  # category -> [hits, total]
    latencies = []
    skipped = 0
    for conv in convs:
        user_id = str(conv.get('sample_id', 'unknown'))
        dia_map = ingest(engine, conv)
        for qa in conv['qa']:
            evidence = qa.get('evidence') or []
            if not evidence:
                skipped += 1
                continue
            evidence_texts = {dia_map[d] for d in evidence if d in dia_map}
            if not evidence_texts:
                skipped += 1
                continue
            cat = qa['category']
            t0 = time.perf_counter()
            hits = engine.search(user_id, qa['question'], top_k=top_k)
            latencies.append(time.perf_counter() - t0)
            hit = any(h['content'] in evidence_texts for h in hits)
            s = stats.setdefault(cat, [0, 0])
            s[0] += int(hit)
            s[1] += 1
    return stats, latencies, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), 'locomo10.json'))
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--max-conv', type=int, default=10)
    ap.add_argument('--vector-weight', type=float, default=0.6, help='向量相似度权重（向量模式）')
    ap.add_argument('--kw-weight', type=float, default=0.3, help='关键词权重（降级模式按 kw:recency 归一化）')
    ap.add_argument('--recency-weight', type=float, default=0.1, help='时间近因权重')
    ap.add_argument('--out', default='', help='可选：把结果写成 JSON')
    args = ap.parse_args()

    db = os.path.join(tempfile.gettempdir(), 'aml_locomo_eval.db')
    if os.path.exists(db):
        os.remove(db)
    embedder = EmbeddingClient()  # 无 DASHSCOPE_API_KEY 时自动降级关键词模式
    engine = MemoryEngine(f'sqlite:///{db}', embedding_client=embedder,
                          vector_weight=args.vector_weight,
                          keyword_weight=args.kw_weight, recency_weight=args.recency_weight)
    mode = 'vector+keyword' if embedder.available else 'keyword-only (degraded)'
    print(f'检索模式: {mode} | top_k={args.top_k}')

    convs = load_conversations(args.data, args.max_conv)
    print(f'载入 {len(convs)} 段对话，开始写入与评测...\n')
    t0 = time.perf_counter()
    stats, latencies, skipped = evaluate(engine, convs, args.top_k)
    total_s = time.perf_counter() - t0

    total_hits = sum(h for h, _ in stats.values())
    total_qa = sum(t for _, t in stats.values())
    print(f'{"category":<14}{"recall@" + str(args.top_k):>10}{"hits":>7}{"total":>7}')
    print('-' * 40)
    for cat in sorted(stats):
        h, t = stats[cat]
        name = CATEGORY_NAMES.get(cat, f'cat{cat}')
        print(f'{name:<14}{h / t:>10.3f}{h:>7}{t:>7}')
    print('-' * 40)
    print(f'{"OVERALL":<14}{total_hits / total_qa:>10.3f}{total_hits:>7}{total_qa:>7}')
    avg_ms = sum(latencies) / len(latencies) * 1000
    print(f'\n平均检索延迟: {avg_ms:.1f} ms | 无证据跳过: {skipped} | 总耗时: {total_s:.1f}s')

    if args.out:
        result = {
            'mode': mode, 'top_k': args.top_k,
            'overall_recall': round(total_hits / total_qa, 4),
            'per_category': {
                CATEGORY_NAMES.get(c, str(c)): {'recall': round(h / t, 4), 'hits': h, 'total': t}
                for c, (h, t) in sorted(stats.items())
            },
            'avg_search_ms': round(avg_ms, 2),
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'结果已写入 {args.out}')


if __name__ == '__main__':
    main()
