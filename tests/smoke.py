# -*- coding: utf-8 -*-
"""冒烟自测脚本

不依赖网络与 MySQL：使用 sqlite + 关键词降级模式验证端到端链路。
目的：在 AML 平台 Smoke 测试前，先把本地协议校验做到零偏差。
运行：python tests/smoke.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core import MemoryEngine, EmbeddingClient

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = ''):
    global PASS, FAIL
    status = 'PASS' if cond else 'FAIL'
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f'[{status}] {name}' + (f'  -> {detail}' if detail and not cond else ''))


def main():
    db = os.path.join(tempfile.gettempdir(), 'aml_smoke_test.db')
    uri = f'sqlite:///{db}'
    if os.path.exists(db):
        os.remove(db)

    # 无 Key 模式：embedding 自动降级
    engine = MemoryEngine(uri, embedding_client=EmbeddingClient(api_key='your-dummy'))

    # --- 场景1：多轮对话写入 ---
    r = engine.add('user_A', 'task_1', 's1', [
        {'speaker': 'user', 'content': '我是宁夏大学大一新生，学的是软件工程，10月份满19岁'},
        {'speaker': 'assistant', 'content': '好的，已记录你的专业信息'},
        {'speaker': 'user', 'content': '我上周接到了一个冒充客服的诈骗电话，对方说我网购的商品有问题要退款'},
        {'speaker': 'user', 'content': '对了，我手机号换成138xxxx8888了，原来那个不用了', 'event_time': '2026-09-25T10:00:00'},
    ])
    check('批量写入4条记忆', r['added'] == 4, str(r))

    # --- 场景2：事实召回 ---
    hits = engine.search('user_A', '这位同学的专业是什么？')
    check('事实召回-专业', any('软件工程' in h['content'] for h in hits), str([h['content'][:20] for h in hits[:3]]))

    hits = engine.search('user_A', '他最近遇到什么诈骗？')
    check('事实召回-诈骗经历', any('客服' in h['content'] for h in hits))

    # --- 场景3：user_id 严格隔离 ---
    engine.add('user_B', 'task_1', 's1', [
        {'speaker': 'user', 'content': '我是浙江大学的老师，教计算机网络的'}
    ])
    hits = engine.search('user_A', '用户的职业是什么？')
    leak = any('老师' in h['content'] or '浙江' in h['content'] for h in hits)
    check('跨用户隔离（user_B 数据不可见）', not leak)

    # --- 场景4：task_id 过滤 ---
    hits = engine.search('user_A', '专业', task_id='task_999')
    check('task 过滤为空', len(hits) == 0)

    # --- 场景5：响应结构完整性（AML 协议关键） ---
    need_keys = {'memory_id', 'speaker', 'content', 'keywords', 'event_time', 'ingested_at', 'score'}
    hits = engine.search('user_A', '用户的手机号是多少')
    check('返回字段完整', all(need_keys <= set(h.keys()) for h in hits),
          str(set(hits[0].keys()) if hits else 'no hits'))

    # --- 场景6：治理-冲突消解 ---
    ok = engine.supersede(4, 5)
    hits = engine.search('user_A', '手机号')
    check('旧记忆已被失效', ok and not any(h['memory_id'] == 4 for h in hits))

    print(f'\n结果: {PASS} 通过 / {FAIL} 失败')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
