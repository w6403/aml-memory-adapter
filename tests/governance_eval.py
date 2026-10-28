# -*- coding: utf-8 -*-
"""治理与条件注入评测（v0.3.0 新增）

LoCoMo 只测检索召回，测不了治理/摘要类行为，本脚本用自造迷你场景补齐。
覆盖：D1 新值覆盖、D3 删除遗忘、误杀防护（never forget / 跨说话人 / 异主题）、
重放去重、摘要生成与 aggregate 条件提升、开关回退。

用法：
  python tests/governance_eval.py            # 有 DASHSCOPE_API_KEY 时全量跑
  无 Key 时向量场景自动跳过（规则层已被离线单测覆盖），去重场景仍可跑
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
            override=True)

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import Config
from memory_core import MemoryEngine, EmbeddingClient
from memory_core.models import MemoryRecord
from memory_core.governance import FORGET_SENTINEL
from memory_core.derive import SUMMARY_SPEAKER

PASS, FAIL, SKIP = 'PASS', 'FAIL', 'SKIP'
results = []


def report(name, status, detail=''):
    results.append((name, status, detail))
    print(f'[{status}] {name}' + (f' — {detail}' if detail else ''))


def fresh_engine(governance=True):
    fd, db = tempfile.mkstemp(suffix='.db', prefix='aml_gov_')
    os.close(fd)
    os.remove(db)
    embedder = EmbeddingClient(model=Config.EMBEDDING_MODEL,
                               dimensions=Config.EMBEDDING_DIMENSIONS)
    engine = MemoryEngine(f'sqlite:///{db}', embedding_client=embedder,
                          vector_weight=0.5, keyword_weight=0.5,
                          recency_weight=0.0, governance_enabled=governance)
    return engine, db


def active_contents(engine, user_id):
    """返回记录列表 [(content, is_active, superseded_by, keywords)]——
    不能用 dict：同文重放场景 content 相同会被折叠。"""
    with Session(engine.engine) as s:
        rows = s.scalars(select(MemoryRecord).where(
            MemoryRecord.user_id == user_id)).all()
        return [(r.content, r.is_active, r.superseded_by, r.keywords)
                for r in rows]


def find(records, needle):
    for r in records:
        if needle in r[0]:
            return r
    return None


def main():
    has_vec = EmbeddingClient(model=Config.EMBEDDING_MODEL).available
    if not has_vec:
        print('无 DASHSCOPE_API_KEY：向量场景跳过\n')

    # ---- 场景 1：新值覆盖（D1）----
    if has_vec:
        eng, _ = fresh_engine()
        eng.add('u1', '', 's1', [
            {'speaker': 'user', 'content': 'My home city is Beijing and I live downtown.'}])
        eng.add('u1', '', 's2', [
            {'speaker': 'user', 'content': 'My home city is Shanghai now, I just moved last month.'}])
        recs = active_contents(eng, 'u1')
        old = find(recs, 'My home city is Beijing')
        new_id = [r['memory_id'] for r in eng._load_snapshot('u1')
                  if 'Shanghai' in r['content']]
        ok = (old and old[1] is False and new_id and old[2] == new_id[0])
        report('S1 新值覆盖：旧值 supersede 并指向新值',
               PASS if ok else FAIL, f'旧记录状态={old}')
        hits = eng.search('u1', 'Where does the user live currently?', top_k=10)
        contents = [h['content'] for h in hits]
        ok = any('Shanghai' in c for c in contents) and not any(
            'Beijing' in c for c in contents)
        report('S1b 覆盖后检索：query 只见到新值', PASS if ok else FAIL)
    else:
        report('S1 新值覆盖', SKIP, '无 key')

    # ---- 场景 2：遗忘指令（D3）----
    if has_vec:
        eng, _ = fresh_engine()
        eng.add('u2', '', 's1', [
            {'speaker': 'user', 'content': "My favorite restaurant is Luigi's on 5th Avenue."}])
        eng.add('u2', '', 's2', [
            {'speaker': 'user', 'content': 'Please forget what I said about my favorite restaurant.'}])
        recs = active_contents(eng, 'u2')
        old = find(recs, "Luigi's")
        instr = find(recs, 'Please forget')
        ok = (old and old[1] is False
              and instr and instr[1] is True and FORGET_SENTINEL in instr[3])
        report('S2 遗忘指令：目标 supersede + 指令留痕带哨兵',
               PASS if ok else FAIL, f'目标={old} 指令={instr}')
        hits = eng.search('u2', 'What is my favorite restaurant?', top_k=10)
        ok = not any("Luigi's" in h['content'] for h in hits)
        report('S2b 遗忘后检索：目标不再返回', PASS if ok else FAIL)
        hits = eng.search('u2', 'What did I ask you to forget?', top_k=10)
        ok = any('forget' in h['content'].lower() for h in hits)
        report('S2c 遗忘回忆类 query：指令记录可检索', PASS if ok else FAIL)
    else:
        report('S2 遗忘指令', SKIP, '无 key')

    # ---- 场景 3：never forget 误杀防护 ----
    if has_vec:
        eng, _ = fresh_engine()
        eng.add('u3', '', 's1', [
            {'speaker': 'user', 'content': 'I will never forget my trip to Paris.'}])
        eng.add('u3', '', 's1', [
            {'speaker': 'user', 'content': 'The Eiffel Tower was beautiful at night.'}])
        n_active = sum(1 for v in active_contents(eng, 'u3') if v[1])
        report('S3 "never forget" 不触发遗忘动作',
               PASS if n_active == 2 and eng.stats['forget'] == 0 else FAIL,
               f'active={n_active} forget_actions={eng.stats["forget"]}')
    else:
        report('S3 never forget 防护', SKIP, '无 key')

    # ---- 场景 4：异主题零触发 ----
    if has_vec:
        eng, _ = fresh_engine()
        eng.add('u4', '', 's1', [
            {'speaker': 'user', 'content': 'I enjoy playing chess on weekends.'}])
        eng.add('u4', '', 's2', [
            {'speaker': 'user', 'content': 'My dentist appointment is on Friday.'}])
        n_active = sum(1 for v in active_contents(eng, 'u4') if v[1])
        report('S4 异主题事实全部保留',
               PASS if n_active == 2 and eng.stats['update'] == 0 else FAIL,
               f'active={n_active} update_actions={eng.stats["update"]}')
    else:
        report('S4 异主题零触发', SKIP, '无 key')

    # ---- 场景 5：跨说话人不 supersede（护 D2 各执一词）----
    if has_vec:
        eng, _ = fresh_engine()
        eng.add('u5', '', 's1', [
            {'speaker': 'user', 'content': 'The project deadline is March 15th, that is final.'}])
        eng.add('u5', '', 's2', [
            {'speaker': 'assistant',
             'content': 'The project deadline is March 15th, noted and confirmed.'}])
        n_active = sum(1 for v in active_contents(eng, 'u5') if v[1])
        report('S5 跨 speaker 近似复述不 supersede',
               PASS if n_active == 2 else FAIL, f'active={n_active}')
    else:
        report('S5 跨 speaker 防护', SKIP, '无 key')

    # ---- 场景 6：同文重放去重（无需向量，降级模式也生效）----
    eng, _ = fresh_engine()
    msg = {'speaker': 'user', 'content': 'Remember my membership number is A-1024.'}
    eng.add('u6', '', 's1', [msg])
    eng.add('u6', '', 's1', [dict(msg)])  # 平台重试重放
    n_active = sum(1 for v in active_contents(eng, 'u6') if v[1])
    report('S6 同文重放仅保留 1 条 active',
           PASS if n_active == 1 and eng.stats['dedup'] >= 1 else FAIL,
           f'active={n_active} dedup={eng.stats["dedup"]}')

    # ---- 场景 7：开关回退（governance=false 时行为同 v0.2.0）----
    eng, _ = fresh_engine(governance=False)
    msg = {'speaker': 'user', 'content': 'Remember my membership number is A-1024.'}
    eng.add('u7', '', 's1', [msg])
    eng.add('u7', '', 's1', [dict(msg)])
    n_active = sum(1 for v in active_contents(eng, 'u7') if v[1])
    report('S7 治理关闭：重放两条都保留（v0.2.0 行为）',
           PASS if n_active == 2 else FAIL, f'active={n_active}')

    # ---- 场景 8：会话摘要生成 + aggregate 条件提升（F1）----
    if has_vec:
        eng, _ = fresh_engine()
        msgs = [{'speaker': 'user' if i % 2 == 0 else 'assistant',
                 'content': f'We talked about topic number {i} in detail today.'}
                for i in range(21)]
        eng.add('u8', '', 's1', msgs)
        sums = [r for r in eng._load_snapshot('u8')
                if r['speaker'] == SUMMARY_SPEAKER]
        ok = len(sums) == 1 and eng.stats['summary'] == 1
        report('S8 摘要记录：满 20 条会话生成且唯一',
               PASS if ok else FAIL, f'summaries={len(sums)}')
        hits = eng.search('u8', 'Can you summarize what we discussed today?',
                          top_k=5)
        ok = hits and hits[0]['speaker'] == SUMMARY_SPEAKER
        report('S8b aggregate query：摘要记录置顶', PASS if ok else FAIL,
               f'top1_speaker={hits[0]["speaker"] if hits else None}')
        hits = eng.search('u8', 'What did we say about topic number 3?', top_k=5)
        ok = any('topic number 3' in h['content'] for h in hits)
        report('S8c 常规 query：原始证据仍命中（摘要不霸榜）',
               PASS if ok else FAIL)
    else:
        report('S8 会话摘要', SKIP, '无 key')

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f'\n合计: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
