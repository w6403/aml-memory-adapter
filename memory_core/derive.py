# -*- coding: utf-8 -*-
"""派生记忆：会话摘要记录（v0.3.0 新增，F1 摘要压缩对策）

零生成式 LLM 的抽取式摘要：复用入库时已算好的 embedding 求质心，
取与质心余弦最高的代表句，按事件时间排序拼接。摘要是"记忆组织"产物
（平台契约允许参赛方组织/索引/更新记忆），content 全部由原文句子组成，
不是生成式答案。

摘要记录以普通 MemoryRecord 入库，speaker 用哨兵 '__summary__' 标记，
embedding 直接用质心向量（零额外 API 调用）。
"""
from collections import Counter

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

SUMMARY_SPEAKER = '__summary__'
SUMMARY_PREFIX = 'Session summary: '


def _mean_vec(vecs: list) -> list:
    if _HAS_NP:
        return np.mean(np.asarray(vecs, dtype=np.float32), axis=0).tolist()
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]


def _cos(a: list, b: list) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_summary(records: list, max_sents: int = 5,
                  top_keywords: int = 10):
    """从同一 session 的记录构造摘要。

    records: [{'content','vec','sort_time','keywords'}]，vec 可为 None。
    返回 {'content','vec','keywords'} 或 None（记录不足/向量全缺时）。
    """
    usable = [r for r in records if r.get('content')]
    if len(usable) < 2:
        return None
    with_vec = [r for r in usable if r.get('vec')]
    if not with_vec:
        return None
    centroid = _mean_vec([r['vec'] for r in with_vec])
    scored = sorted(with_vec, key=lambda r: _cos(r['vec'], centroid),
                    reverse=True)
    picked = scored[:max_sents]
    # 代表句按原事件顺序拼接，读起来是时间线而非相似度序
    picked.sort(key=lambda r: r.get('sort_time') or '')
    content = SUMMARY_PREFIX + ' | '.join(r['content'] for r in picked)

    df = Counter()
    for r in usable:
        df.update(w.lower() for w in (r.get('keywords') or []))
    keywords = [w for w, _ in df.most_common(top_keywords)]
    return {'content': content, 'vec': centroid, 'keywords': keywords}
