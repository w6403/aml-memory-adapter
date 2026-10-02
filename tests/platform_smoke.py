# -*- coding: utf-8 -*-
"""平台契约 Smoke：模拟 AML 平台调用方式，逐项校验 Add/Search 契约

在拿到 Eval Key、跑平台正式 Smoke 之前，先用本脚本对自己部署的接口做
零偏差校验。所有请求格式严格按官网 API Guide（2026-10-01 版）构造。

用法：
  # 打本地起的服务
  python tests/platform_smoke.py --base-url http://127.0.0.1:8600
  # 打线上部署（token 为服务器 .env 的 API_AUTH_TOKEN）
  python tests/platform_smoke.py --base-url https://aml.ningxia-tour.top --token <TOKEN>
"""
import argparse
import json
import sys
import time

import requests

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = ''):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f'[{"PASS" if cond else "FAIL"}] {name}' + (f'  -> {detail}' if detail and not cond else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--token', default='', help='API_AUTH_TOKEN；留空则假定服务未开鉴权')
    args = ap.parse_args()
    base = args.base_url.rstrip('/')
    headers = {'Authorization': f'Bearer {args.token}'} if args.token else {}
    # user_id 格式仿照平台：eval:run_xxx:benchmark:conv-N
    run_tag = f'local{int(time.time())}'
    user_a = f'eval:{run_tag}:locomo_refined:conv-0'
    user_b = f'eval:{run_tag}:locomo_refined:conv-1'
    session_a = f'eval:{run_tag}:sample:0'

    # --- 1. Health：无鉴权 GET，任意 2xx ---
    r = requests.get(f'{base}/health', timeout=10)
    check('health 无鉴权 2xx', 200 <= r.status_code < 300, f'status={r.status_code}')

    # --- 2. 未授权拒绝（仅当服务开了鉴权） ---
    if args.token:
        r = requests.post(f'{base}/search', json={
            'query': 'test', 'user_id': user_a, 'top_k': 100}, timeout=10)
        check('无 token 返回 401', r.status_code == 401, f'status={r.status_code}')

    # --- 3. Add 契约：请求/响应字段 ---
    req_id = f'eval:{run_tag}:locomo_refined:conv-0:chunk-0'
    add_payload = {
        'request_id': req_id,
        'messages': [
            {'role': 'user', 'timestamp': 1683479160000,
             'content': 'I moved from Sweden 4 years ago and I play the clarinet.'},
            {'role': 'assistant', 'timestamp': 1683479220000,
             'content': 'That is wonderful! How do you like it here so far?'},
        ],
        'user_id': user_a,
        'session_id': session_a,
    }
    r = requests.post(f'{base}/add', json=add_payload, headers=headers, timeout=60)
    ok = r.status_code == 200
    check('add HTTP 200', ok, f'status={r.status_code} body={r.text[:200]}')
    if ok:
        body = r.json()
        check('add success 为布尔 true', body.get('success') is True, str(body))
        check('add echo request_id', body.get('request_id') == req_id, str(body))
        check('add echo user_id', body.get('user_id') == user_a, str(body))
        check('add echo session_id', body.get('session_id') == session_a, str(body))

    # --- 4. 同步语义：Add 返回后立即可检索 ---
    search_payload = {'query': 'What instrument does the user play?',
                      'user_id': user_a, 'top_k': 100}
    r = requests.post(f'{base}/search', json=search_payload, headers=headers, timeout=60)
    ok = r.status_code == 200
    check('search HTTP 200', ok, f'status={r.status_code} body={r.text[:200]}')
    if ok:
        body = r.json()
        data = body.get('data')
        check('search 响应含 data 数组', isinstance(data, list), str(body)[:200])
        if isinstance(data, list):
            check('data 数量不超过 top_k', len(data) <= 100, f'len={len(data)}')
            check('同步语义：写入立即可检索',
                  any('clarinet' in (h.get('content') or '') for h in data),
                  str([h.get('content', '')[:40] for h in data[:5]]))
            if data:
                h = data[0]
                check('hit.id 为非空字符串',
                      isinstance(h.get('id'), str) and len(h['id']) > 0, str(h))
                check('hit.content 为非空字符串',
                      isinstance(h.get('content'), str) and len(h['content']) > 0, str(h))
                if 'score' in h and h['score'] is not None:
                    check('hit.score 为数值', isinstance(h['score'], (int, float)), str(h))

    # --- 5. 空结果返回空数组（不是错误） ---
    r = requests.post(f'{base}/search', json={
        'query': 'completely unrelated xyzzy', 'user_id': user_a, 'top_k': 100},
        headers=headers, timeout=60)
    if r.status_code == 200:
        check('无结果返回空数组', r.json().get('data') == [] or isinstance(r.json().get('data'), list),
              r.text[:200])
    else:
        check('无结果返回空数组', False, f'status={r.status_code}')

    # --- 6. user_id 隔离 ---
    requests.post(f'{base}/add', json={
        'request_id': f'eval:{run_tag}:locomo_refined:conv-1:chunk-0',
        'messages': [{'role': 'user', 'content': 'secret: I collect antique typewriters.'}],
        'user_id': user_b,
        'session_id': f'eval:{run_tag}:sample:1',
    }, headers=headers, timeout=60)
    r = requests.post(f'{base}/search', json={
        'query': 'What does the user collect?', 'user_id': user_a, 'top_k': 100},
        headers=headers, timeout=60)
    if r.status_code == 200:
        leak = any('typewriter' in (h.get('content') or '') for h in r.json().get('data', []))
        check('user_id 严格隔离', not leak)
    else:
        check('user_id 严格隔离', False, f'status={r.status_code}')

    # --- 7. 缺字段返回 4xx（契约错误前置发现） ---
    r = requests.post(f'{base}/search', json={'user_id': user_a}, headers=headers, timeout=10)
    check('缺 query/top_k 返回 4xx', 400 <= r.status_code < 500, f'status={r.status_code}')

    print(f'\n结果: {PASS} 通过 / {FAIL} 失败  (base={base})')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
