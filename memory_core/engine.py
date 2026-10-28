# -*- coding: utf-8 -*-
"""记忆引擎：Add 写入 + Search 混合检索

这是参赛的核心模块（memory-core），不含任何智防校园业务字段。
检索策略（混合打分）：
  score = VECTOR_WEIGHT * 余弦相似度
        + KEYWORD_WEIGHT * 关键词重合率
        + RECENCY_WEIGHT * 时间近因
向量不可用时自动退化为关键词 + 近因模式。
"""
import logging
import math
import re
from collections import Counter, OrderedDict
from datetime import datetime
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from .models import Base, MemoryRecord
from .embedding_client import EmbeddingClient
from .governance import (
    FORGET_SENTINEL, classify_forget, has_update_trigger,
    is_update_candidate, shared_keywords)
from .query_rules import classify_query
from .derive import SUMMARY_SPEAKER, build_summary

logger = logging.getLogger(__name__)

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
                 vector_weight=0.6, keyword_weight=0.3, recency_weight=0.1,
                 governance_enabled=True, summary_enabled=True,
                 query_boost_enabled=True,
                 forget_sim_threshold=0.45, forget_strong_sim=0.85,
                 forget_max_targets=3,
                 update_sim_threshold=0.55, update_min_shared_kws=2,
                 dedup_sim_threshold=0.97,
                 summary_min_messages=20, summary_max_sents=5,
                 summary_boost=0.5, recency_query_boost=0.15,
                 forget_query_boost=0.15):
        self.engine = create_engine(database_uri, pool_pre_ping=True)
        self.embedder = embedding_client or EmbeddingClient()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.recency_weight = recency_weight
        # v0.3.0：治理 / 摘要 / query 条件注入（默认值与 config.py 一致，
        # local_eval 不经 config 直接用默认即"全开"，回归口径与线上一致）
        self.governance_enabled = governance_enabled
        self.summary_enabled = summary_enabled
        self.query_boost_enabled = query_boost_enabled
        self.forget_sim_threshold = forget_sim_threshold
        self.forget_strong_sim = forget_strong_sim
        self.forget_max_targets = forget_max_targets
        self.update_sim_threshold = update_sim_threshold
        self.update_min_shared_kws = update_min_shared_kws
        self.dedup_sim_threshold = dedup_sim_threshold
        self.summary_min_messages = summary_min_messages
        self.summary_max_sents = summary_max_sents
        self.summary_boost = summary_boost
        self.recency_query_boost = recency_query_boost
        self.forget_query_boost = forget_query_boost
        # 治理动作计数（回归评测观察误杀率：LoCoMo 全程应近零）
        self.stats = {'dedup': 0, 'forget': 0, 'update': 0, 'summary': 0}
        Base.metadata.create_all(self.engine)
        # 检索快照缓存：embedding 以 JSON 列存储，每次检索全量反序列化太慢。
        # 按 (user_id, task_id) 缓存纯 dict 快照，add/supersede 时按 user 失效。
        self._snap_cache = OrderedDict()  # (user_id, task_id) -> dict 列表
        self._snap_cap = 32
        self._summary_done = {}  # (user_id, session_id) -> 上次生成摘要时的消息数

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

    def _load_superseded(self, user_id: str, task_id: str = None) -> list:
        """已被覆盖/遗忘的旧记录（is_active=False 且 superseded_by 非空）。
        仅供 conflict 类 query 合并作证（D2 矛盾检测需要新旧两值同时在场）。"""
        key = (str(user_id), str(task_id or ''), '__superseded__')
        cached = self._snap_cache.get(key)
        if cached is not None:
            self._snap_cache.move_to_end(key)
            return cached
        with Session(self.engine) as session:
            stmt = (select(MemoryRecord)
                    .where(MemoryRecord.user_id == str(user_id),
                           MemoryRecord.is_active.is_(False),
                           MemoryRecord.superseded_by.isnot(None)))
            if task_id:
                stmt = stmt.where(MemoryRecord.task_id == str(task_id))
            records = session.scalars(stmt).all()
            snapshot = [{
                'memory_id': r.id, 'user_id': r.user_id, 'task_id': r.task_id,
                'session_id': r.session_id, 'speaker': r.speaker or '',
                'content': r.content, 'keywords': r.keywords or [],
                'event_time': r.event_time, 'ingested_at': r.ingested_at,
                'is_active': False, 'vec': r.embedding or None,
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

        # v0.3.0 治理：写入前快照作为覆盖/遗忘比对基线（治理关闭时跳过）
        old_snapshot = (self._load_snapshot(user_id, task_id)
                        if self.governance_enabled else [])

        added = 0
        new_records = []  # (id, speaker, content, kws, vec, forget_level)
        with Session(self.engine) as session:
            for m, kws, vec in zip(messages, keywords_list, vectors):
                evt = m.get('event_time')
                if isinstance(evt, str):
                    try:
                        evt = datetime.fromisoformat(evt.replace('Z', '+00:00')).replace(tzinfo=None)
                    except ValueError:
                        evt = None
                speaker = str(m.get('speaker', ''))
                # 遗忘指令分级：weak/strong 都在 keywords 里追加哨兵（内部字段，
                # 不影响 Search 返回的 content），供 forget 类 query 条件提升
                forget_level = (classify_forget(m['content'], speaker)
                                if self.governance_enabled else '')
                if forget_level:
                    kws = list(kws) + [FORGET_SENTINEL]
                rec = MemoryRecord(
                    user_id=str(user_id), task_id=str(task_id or ''),
                    session_id=str(session_id or ''),
                    speaker=speaker,
                    content=m['content'], keywords=kws,
                    embedding=vec, event_time=evt,
                )
                session.add(rec)
                added += 1
                new_records.append([rec, speaker, m['content'], kws, vec,
                                    forget_level])
            session.flush()  # 先拿自增 id 再提交
            for item in new_records:
                item[0] = item[0].id
            session.commit()

        if self.governance_enabled:
            self._govern(user_id, task_id, old_snapshot, new_records)
        if self.summary_enabled:
            self._refresh_summary(user_id, task_id, str(session_id or ''))
        self._invalidate(user_id)
        return {'added': added, 'embedded': sum(1 for v in vectors if v)}

    # ---------- v0.3.0 记忆治理（D1 覆盖 / D3 遗忘 / 重放去重） ----------

    def _govern(self, user_id: str, task_id: str,
                old_snapshot: list, new_records: list):
        """对每条新记录执行遗忘/覆盖/去重判定并 supersede 旧记录。

        防误杀四道闸：高相似起阈 + 同说话人 + 关键词交集 + 每条上限；
        摘要记录（'__summary__'）不参与治理（既不作目标也不作触发源）。
        """
        olds = [r for r in old_snapshot if r['speaker'] != SUMMARY_SPEAKER]
        if not olds:
            return
        used = set()  # 本轮已被 supersede 的旧记录 id
        # 批量余弦矩阵（向量缺失的旧记录不参与相似度判定）
        mat = None
        mat_rows = []
        if _HAS_NP:
            mat_rows = [i for i, r in enumerate(olds) if r['vec']]
            if mat_rows:
                try:
                    mat = np.asarray([olds[i]['vec'] for i in mat_rows],
                                     dtype=np.float32)
                except (ValueError, TypeError):
                    mat, mat_rows = None, []

        for nid, speaker, content, kws, vec, forget_level in new_records:
            if speaker == SUMMARY_SPEAKER:
                continue
            # 1) 同文去重（无需向量，降级模式也生效）：
            #    同说话人 + 规范化同文 → 平台重放重复写入，supersede 旧留新
            norm = content.strip().lower()
            for r in olds:
                if (r['memory_id'] not in used and r['speaker'] == speaker
                        and r['content'].strip().lower() == norm):
                    if self.supersede(r['memory_id'], nid):
                        used.add(r['memory_id'])
                        self.stats['dedup'] += 1
                    break
            else:
                # 2) 强遗忘指令：定位目标旧记忆并 supersede（每指令上限 N 条）
                #    命中条件（同说话人）：cos >= 强阈值 直接命中；
                #    或 cos >= 下限 且共享 >=2 内容关键词（"forget what I said
                #    about X" 的措辞与目标差异大，cosine 不可靠，关键词交集才是主闸）
                if forget_level == 'strong' and vec is not None and mat is not None:
                    q = np.asarray(vec, dtype=np.float32)
                    qn = float(np.linalg.norm(q))
                    if qn > 0:
                        norms = np.linalg.norm(mat, axis=1)
                        cos = (mat @ q) / np.maximum(norms * qn, 1e-12)
                        cands = []
                        excl = {FORGET_SENTINEL, speaker.lower()}
                        for row_i, c in zip(mat_rows, cos):
                            r = olds[row_i]
                            if r['memory_id'] in used:
                                continue
                            if r['speaker'] != speaker:
                                continue
                            c = float(c)
                            if c >= self.forget_strong_sim:
                                cands.append((c, r))
                            elif (c >= self.forget_sim_threshold
                                  and len(shared_keywords(kws, r['keywords'],
                                                          exclude=excl)) >= 2):
                                cands.append((c, r))
                        cands.sort(key=lambda x: x[0], reverse=True)
                        for c, r in cands[:self.forget_max_targets]:
                            if self.supersede(r['memory_id'], nid):
                                used.add(r['memory_id'])
                                self.stats['forget'] += 1
                                logger.info('governance: forget supersede %s -> %s '
                                            '(cos=%.3f, user=%s)',
                                            r['memory_id'], nid, c, user_id)
                # 3) 新值覆盖/近同文去重（向量）：每条新记录最多 supersede 1 条
                #    前置闸：新记录必须自带变更措辞（moved/now/不再是…），
                #    否则同话题回声会被误杀（LoCoMo 回归实测教训）
                elif (vec is not None and mat is not None
                      and has_update_trigger(content)):
                    q = np.asarray(vec, dtype=np.float32)
                    qn = float(np.linalg.norm(q))
                    if qn > 0:
                        norms = np.linalg.norm(mat, axis=1)
                        cos = (mat @ q) / np.maximum(norms * qn, 1e-12)
                        order = np.argsort(-cos)
                        for oi in order:
                            r = olds[mat_rows[oi]]
                            if r['memory_id'] in used:
                                continue
                            c = float(cos[oi])
                            if c < self.update_sim_threshold:
                                break
                            verdict = is_update_candidate(
                                c, kws, r['keywords'],
                                same_speaker=(r['speaker'] == speaker),
                                sim_threshold=self.update_sim_threshold,
                                dedup_threshold=self.dedup_sim_threshold,
                                min_shared_kws=self.update_min_shared_kws)
                            if verdict:
                                if self.supersede(r['memory_id'], nid):
                                    used.add(r['memory_id'])
                                    self.stats[verdict] += 1
                                    logger.info(
                                        'governance: %s supersede %s -> %s '
                                        '(cos=%.3f, user=%s)',
                                        verdict, r['memory_id'], nid, c,
                                        user_id)
                            break  # 无论是否命中都只试最相似的一条

    # ---------- v0.3.0 会话摘要（F1 摘要压缩对策，抽取式，零 LLM） ----------

    def _refresh_summary(self, user_id: str, task_id: str, session_id: str):
        """session 消息数每新增 SUMMARY_MIN_MESSAGES 条刷新一次抽取式摘要。"""
        if not session_id:
            return
        skey = (str(user_id), session_id)
        with Session(self.engine) as session:
            base = (select(MemoryRecord)
                    .where(MemoryRecord.user_id == str(user_id),
                           MemoryRecord.session_id == session_id,
                           MemoryRecord.is_active.is_(True)))
            rows = session.scalars(base.where(
                MemoryRecord.speaker != SUMMARY_SPEAKER)).all()
            count = len(rows)
            if count < self.summary_min_messages:
                return
            last = self._summary_done.get(skey, 0)
            if count - last < self.summary_min_messages:
                return
            records = [{
                'content': r.content,
                'vec': r.embedding or None,
                'sort_time': (r.event_time or r.ingested_at).isoformat()
                             if (r.event_time or r.ingested_at) else '',
                'keywords': r.keywords or [],
            } for r in rows]
            old_sum = session.scalars(base.where(
                MemoryRecord.speaker == SUMMARY_SPEAKER)).first()
            s = build_summary(records, self.summary_max_sents)
            if not s:
                return
            rec = MemoryRecord(
                user_id=str(user_id), task_id=str(task_id or ''),
                session_id=session_id, speaker=SUMMARY_SPEAKER,
                content=s['content'], keywords=s['keywords'],
                embedding=s['vec'],
            )
            session.add(rec)
            session.flush()
            new_id = rec.id
            session.commit()
        self._summary_done[skey] = count
        self.stats['summary'] += 1
        if old_sum is not None:
            self.supersede(old_sum.id, new_id)

    # ---------- Search ----------

    def search(self, user_id: str, query: str, task_id: str = None,
               top_k: int = 20) -> list:
        """按混合得分返回相关记忆证据。严格限定 user_id 作用域。"""
        if not query:
            return []
        q_vec = (self.embedder.embed([query])[0]
                 if self.vector_weight > 0 and self.embedder.available else None)
        q_kws = set(w.lower() for w in extract_keywords(query))

        # v0.3.0 query 条件注入：仅命中特征的 query 改变行为，
        # 常规 query 与 v0.2.0 完全一致（LoCoMo 回归保护）
        flags = classify_query(query) if self.query_boost_enabled else set()
        snapshot = self._load_snapshot(user_id, task_id)
        if 'conflict' in flags:
            # D2 矛盾检测：把已被覆盖/遗忘的旧值并入候选（新旧同时在场才能比对）
            snapshot = snapshot + self._load_superseded(user_id, task_id)

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

            # v0.3.0 条件注入（默认全不触发时 score 不变）：
            # 摘要记录：仅 aggregate query 可见（乘法提升）；
            # 常规 query 直接排除，零挤占证据位（回归保护）
            if rec['speaker'] == SUMMARY_SPEAKER:
                if 'aggregate' not in flags:
                    continue
                score *= 1.0 + self.summary_boost
            # recency query：仅该 query 叠加近因分（全局 RECENCY_WEIGHT 仍为 0）
            if 'recency' in flags:
                score += self.recency_query_boost * t_score
            # forget query：提升遗忘指令记录（带 '__forget__' 哨兵）
            if 'forget' in flags and FORGET_SENTINEL in r_kws:
                score += self.forget_query_boost

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
