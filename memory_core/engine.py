# -*- coding: utf-8 -*-
"""记忆引擎：Add 写入 + Search 混合检索

这是参赛的核心模块（memory-core），不含任何智防校园业务字段。
检索策略（混合打分）：
  score = VECTOR_WEIGHT * 余弦相似度
        + KEYWORD_WEIGHT * 关键词重合率
        + RECENCY_WEIGHT * 时间近因
向量不可用时自动退化为关键词 + 近因模式。
"""
import math
import re
from datetime import datetime
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .models import Base, MemoryRecord
from .embedding_client import EmbeddingClient

try:
    import jieba  # 中文分词用于关键词抽取
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_STOPWORDS = set('的了和是就都而及与着或一个没有我们你们他们这那也再说吧吗呢啊'.replace(' ', ''))
_TOKEN_RE = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]+')


def extract_keywords(text: str, limit: int = 12) -> list:
    """轻量关键词抽取：jieba 分词优先，退化到正则切词。不依赖 LLM。"""
    if not text:
        return []
    if _HAS_JIEBA:
        words = [w.strip() for w in jieba.cut_for_search(text)]
    else:
        words = _TOKEN_RE.findall(text)
    seen, out = set(), []
    for w in words:
        if len(w) < 2 or w.lower() in _STOPWORDS or w.isdigit():
            continue
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
        if len(out) >= limit:
            break
    return out


def _cosine(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryEngine:
    def __init__(self, database_uri: str, embedding_client: EmbeddingClient = None,
                 vector_weight=0.6, keyword_weight=0.3, recency_weight=0.1):
        self.engine = create_engine(database_uri, pool_pre_ping=True)
        self.embedder = embedding_client or EmbeddingClient()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.recency_weight = recency_weight
        Base.metadata.create_all(self.engine)

    # ---------- Add ----------

    def add(self, user_id: str, task_id: str, session_id: str,
            messages: list) -> dict:
        """写入一批对话/事件记忆。messages: [{speaker, content, event_time?}]"""
        messages = [m for m in messages if m.get('content')]
        if not messages:
            return {'added': 0}

        contents = [m['content'] for m in messages]
        keywords_list = [extract_keywords(c) for c in contents]
        vectors = self.embedder.embed(contents) if self.vector_weight > 0 else [None] * len(contents)

        added = 0
        with Session(self.engine) as session:
            for m, kws, vec in zip(messages, keywords_list, vectors):
                evt = m.get('event_time')
                if isinstance(evt, str):
                    try:
                        evt = datetime.fromisoformat(evt.replace('Z', '+00:00')).replace(tzinfo=None)
                    except ValueError:
                        evt = None
                rec = MemoryRecord(
                    user_id=str(user_id), task_id=str(task_id or ''),
                    session_id=str(session_id or ''),
                    speaker=str(m.get('speaker', '')),
                    content=m['content'], keywords=kws,
                    embedding=vec, event_time=evt,
                )
                session.add(rec)
                added += 1
            session.commit()
        return {'added': added, 'embedded': sum(1 for v in vectors if v)}

    # ---------- Search ----------

    def search(self, user_id: str, query: str, task_id: str = None,
               top_k: int = 20) -> list:
        """按混合得分返回相关记忆证据。严格限定 user_id 作用域。"""
        if not query:
            return []
        q_vec = (self.embedder.embed([query])[0]
                 if self.vector_weight > 0 and self.embedder.available else None)
        q_kws = set(w.lower() for w in extract_keywords(query))

        with Session(self.engine) as session:
            stmt = (select(MemoryRecord)
                    .where(MemoryRecord.user_id == str(user_id),
                           MemoryRecord.is_active.is_(True)))
            if task_id:
                stmt = stmt.where(MemoryRecord.task_id == str(task_id))
            records = session.scalars(stmt).all()
            snapshot = [(r, (r.embedding or None)) for r in records]

        now = datetime.now()
        scored = []
        for rec, vec in snapshot:
            v_score = _cosine(q_vec, vec) if q_vec and vec else 0.0
            r_kws = set(w.lower() for w in (rec.keywords or []))
            k_score = (len(q_kws & r_kws) / len(q_kws)) if q_kws else 0.0
            ref_time = rec.event_time or rec.ingested_at
            age_days = max((now - ref_time).total_seconds() / 86400.0, 0.0) if ref_time else 0.0
            t_score = 1.0 / (1.0 + age_days / 30.0)  # 30 天半衰

            if q_vec and self.vector_weight > 0:
                score = (self.vector_weight * v_score
                         + self.keyword_weight * k_score
                         + self.recency_weight * t_score)
            else:  # 降级：关键词 + 近因
                score = 0.75 * k_score + 0.25 * t_score
            if score > 0:
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**rec.to_dict(), 'score': round(score, 6)}
            for score, rec in scored[:top_k]
        ]

    # ---------- 记忆治理（v0.2 计划） ----------

    def supersede(self, old_id: int, new_id: int) -> bool:
        """冲突消解：旧记忆失效并指向新记忆。"""
        with Session(self.engine) as session:
            rec = session.get(MemoryRecord, old_id)
            if not rec:
                return False
            rec.is_active = False
            rec.superseded_by = new_id
            session.commit()
        return True
