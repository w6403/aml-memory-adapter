# -*- coding: utf-8 -*-
"""embedding 客户端

复用智防校园 server/services/ai_service.py 的通义千问调用模式
（OpenAI 兼容模式 + 重试 + 无 Key 降级），改造为 text-embedding 调用。
降级策略：无 Key 或调用失败时返回 None，检索层自动退化为纯关键词模式，
保证冒烟自测离线可跑。

本地评测调参时同一批文本会被反复 embedding（LoCoMo 全量约 3300 次调用），
因此内置 sqlite 磁盘缓存：同一 (model, text) 只调一次远程接口。
缓存文件 .embedding_cache.db（已被 .gitignore 的 *.db 覆盖），失败结果不缓存。
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
import logging
import requests

logger = logging.getLogger(__name__)

DASHSCOPE_EMBED_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.embedding_cache.db')


class EmbeddingClient:
    def __init__(self, api_key: str = '', model: str = 'text-embedding-v3',
                 timeout: int = 10, max_retries: int = 2,
                 cache_path: str = _DEFAULT_CACHE):
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY', '')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # cache_path=None 关闭缓存；默认项目根目录 .embedding_cache.db
        self._lock = threading.Lock()
        self._cache = None
        if cache_path:
            self._cache = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache.execute(
                'CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT)')

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f'{self.model}\n{text}'.encode('utf-8')).hexdigest()

    def _cache_get(self, text: str):
        if not self._cache:
            return None
        with self._lock:
            row = self._cache.execute(
                'SELECT v FROM emb WHERE k = ?', (self._cache_key(text),)).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, text: str, vec):
        if not self._cache or not vec:
            return
        with self._lock:
            self._cache.execute(
                'INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)',
                (self._cache_key(text), json.dumps(vec)))
            self._cache.commit()

    @property
    def available(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith('your-')

    def embed(self, texts: list) -> list:
        """批量向量化。返回与 texts 等长的列表，失败项为 None。命中缓存的不发请求。"""
        if not self.available or not texts:
            return [None] * len(texts)
        results = [None] * len(texts)
        missing = []
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                missing.append(i)
        # dashscope 单批上限：按 10 条分批
        for start in range(0, len(missing), 10):
            idxs = missing[start:start + 10]
            vectors = self._call([texts[i] for i in idxs])
            if vectors is None:
                continue
            for i, vec in zip(idxs, vectors):
                results[i] = vec
                self._cache_put(texts[i], vec)
        return results

    def _call(self, batch: list):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {'model': self.model, 'input': batch}
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(DASHSCOPE_EMBED_URL, headers=headers,
                                     json=data, timeout=self.timeout)
                if resp.status_code == 429:
                    logger.warning('embedding 配额超限 (429)，尝试 %d/%d',
                                   attempt + 1, self.max_retries)
                    time.sleep(1 * (attempt + 1))
                    continue
                result = resp.json()
                if 'data' in result:
                    return [item['embedding'] for item in result['data']]
                logger.warning('embedding 响应异常: %s', str(result)[:200])
                return None
            except requests.RequestException as e:
                logger.warning('embedding 调用失败: %s', e)
        return None
