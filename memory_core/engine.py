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
from collections import Counter, OrderedDict
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

try:
    import numpy as np  # 批量余弦计算（纯 Python 逐条算在 Full 规模下过慢）
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

_STOPWORDS = set('的了和是就都而及与着或一个没有我们你们他们这那也再说吧吗呢啊'.replace(' ', ''))
_EN_STOPWORDS = set(
    'a an the and or but if then else when while at by for with about into through '
    'during before after above below to from up down in out on off over under again '
    'further once here there all any both each few more most other some such no nor '
    'not only own same so than too very can will just should now of do does did doing '
    'have has had having am is are was were be been being would could ought shall '
    'i me my we our us you your he him his she her it its they them their this that '
    'these those what which who whom how as '
    # 疑问词/高频虚词：出现在问题里只制造噪声重合
    'where why whether many much like really very quite going go goes went gone '
    'get got getting make makes made take takes took thing things something anything '
    'everything nothing someone anyone everyone say says said tell tells told '
    'know knows knew think thinks thought want wants wanted'.split()
)
_TOKEN_RE = re.compile(r'[一-龥a-zA-Z0-9]+')
_CJK_RE = re.compile(r'[一-龥]+')
_LATIN_RE = re.compile(r'[a-zA-Z0-9]+')


def _stem(word: str) -> str:
    """轻量英文词干化（规则法）：只要求 query 侧和记忆侧一致，不追求语言学正确。
    解决 research/researching、books/book、stories/story 类形态差异。"""
    w = word.lower()
    if len(w) < 5 or not w.isalpha() or w.isdigit():
        return w
    if w.endswith('ies'):
        return w[:-3] + 'y'
    for suf in ('ing', 'ied', 'es', 'ed', 's'):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[:-len(suf)]
            # running -> run，stopped -> stop（双写辅音还原）
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
                stem = stem[:-1]
            return stem
    return w


def extract_keywords(text: str, limit: int = 40) -> list:
    """轻量关键词抽取（中英双语）：中文段走 jieba，英文/数字段按整词。
    保留年份等数字 token（时序推理依赖）；不依赖 LLM。"""
    if not text:
        return []
    text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', '', text)  # de-stress -> destress，连字符分词修复
    if _HAS_JIEBA:
        words = []
        pos = 0
        for m in _CJK_RE.finditer(text):
            words.extend(_LATIN_RE.findall(text[pos:m.start()]))
            words.extend(w.strip() for w in jieba.cut_for_search(m.group()))
            pos = m.end()
        words.extend(_LATIN_RE.findall(text[pos:]))
    else:
        words = _TOKEN_RE.findall(text)
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if len(w) < 2 or lw in _STOPWORDS or lw in _EN_STOPWORDS:
            continue
        if w.isdigit() and len(w) < 3:  # 丢弃短数字噪声，保留年份/日期
            continue
        key = _stem(w) if w.isascii() else lw  # 英文存词干，中文原样
        if key not in seen:
            seen.add(key)
            out.append(key)
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
        # 检索快照缓存：embedding 以 JSON 列存储，每次检索全量反序列化太慢。
        # 按 (user_id, task_id) 缓存纯 dict 快照，add/supersede 时按 user 失效。
        self._snap_cache = OrderedDict()  # (user_id, task_id) -> dict 列表
        self._snap_cap = 32

    def _invalidate(self, user_id: str):
        for key in [k for k in self._snap_cache if k[0] == str(user_id)]:
            del self._snap_cache[key]

    def _load_snapshot(self, user_id: str, task_id: str = None) -> list:
        key = (str(user_id), str(task_id or ''))
        cached = self._snap_cache.get(key)
        if cached is not None:
            self._snap_cache.move_to_end(key)
            return cached
        with Session(self.engine) as session:
            stmt = (select(MemoryRecord)
                    .where(MemoryRecord.user_id == str(user_id),
                           MemoryRecord.is_active.is_(True)))
            if task_id:
                stmt = stmt.where(MemoryRecord.task_id == str(task_id))
            records = session.scalars(stmt).all()
            snapshot = [{
                'memory_id': r.id, 'user_id': r.user_id, 'task_id': r.task_id,
                'session_id': r.session_id, 'speaker': r.speaker or '',
                'content': r.content, 'keywords': r.keywords or [],
                'event_time': r.event_time, 'ingested_at': r.ingested_at,
                'is_active': True, 'vec': r.embedding or None,
            } for r in records]
        self._snap_cache[key] = snapshot
        if len(self._snap_cache) > self._snap_cap:
            self._snap_cache.popitem(last=False)
        return snapshot

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
        self._invalidate(user_id)
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

        snapshot = self._load_snapshot(user_id, task_id)

        # 每个记忆的匹配词集 = 内容关键词 + 说话人名（问答常问"X 做了什么"，
        # 而证据往往是 X 本人的发言，正文里并不出现 X 的名字）
        kw_sets = []
        df = Counter()  # 用户语料内的文档频率，用于 IDF 加权（压低人名等泛词）
        for rec in snapshot:
            kws = set(w.lower() for w in rec['keywords'])
            if rec['speaker']:
                kws.add(rec['speaker'].lower())
            kw_sets.append(kws)
            df.update(kws)
        n_docs = max(len(snapshot), 1)

        def _idf(kw: str) -> float:
            return math.log(1.0 + n_docs / (1.0 + df.get(kw, 0)))

        q_idf_total = sum(_idf(kw) for kw in q_kws)

        # 批量余弦：numpy 可用且有 query 向量时一次性算完，否则逐条退化
        v_scores = {}
        if q_vec and _HAS_NP:
            try:
                q_arr = np.asarray(q_vec, dtype=np.float32)
                q_norm = float(np.linalg.norm(q_arr))
                idx = [i for i, rec in enumerate(snapshot)
                       if rec['vec'] and len(rec['vec']) == len(q_vec)]
                if q_norm > 0 and idx:
                    mat = np.asarray([snapshot[i]['vec'] for i in idx], dtype=np.float32)
                    norms = np.linalg.norm(mat, axis=1)
                    cos = (mat @ q_arr) / np.maximum(norms * q_norm, 1e-12)
                    cos[norms == 0] = 0.0
                    v_scores = dict(zip(idx, (float(c) for c in cos)))
            except (ValueError, TypeError):
                v_scores = {}  # 维度不齐等异常，退回逐条计算

        now = datetime.now()
        scored = []
        for i, (rec, r_kws) in enumerate(zip(snapshot, kw_sets)):
            if q_vec and self.vector_weight > 0:
                v_score = (v_scores[i] if i in v_scores
                           else _cosine(q_vec, rec['vec']) if rec['vec'] else 0.0)
            else:
                v_score = 0.0
            k_score = (sum(_idf(kw) for kw in (q_kws & r_kws)) / q_idf_total
                       if q_idf_total else 0.0)
            ref_time = rec['event_time'] or rec['ingested_at']
            age_days = max((now - ref_time).total_seconds() / 86400.0, 0.0) if ref_time else 0.0
            t_score = 1.0 / (1.0 + age_days / 30.0)  # 30 天半衰

            if q_vec and self.vector_weight > 0:
                score = (self.vector_weight * v_score
                         + self.keyword_weight * k_score
                         + self.recency_weight * t_score)
            else:  # 降级：关键词 + 近因（按配置权重归一化）
                w_sum = self.keyword_weight + self.recency_weight
                score = (self.keyword_weight * k_score
                         + self.recency_weight * t_score) / w_sum if w_sum else k_score
            if score > 0:
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, rec in scored[:top_k]:
            out.append({
                'memory_id': rec['memory_id'], 'user_id': rec['user_id'],
                'task_id': rec['task_id'], 'session_id': rec['session_id'],
                'speaker': rec['speaker'], 'content': rec['content'],
                'keywords': rec['keywords'],
                'event_time': rec['event_time'].isoformat() if rec['event_time'] else None,
                'ingested_at': rec['ingested_at'].isoformat() if rec['ingested_at'] else None,
                'is_active': rec['is_active'],
                'score': round(score, 6),
            })
        return out

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
            user_id = rec.user_id
        self._invalidate(user_id)
        return True
