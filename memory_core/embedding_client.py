# -*- coding: utf-8 -*-
"""embedding 客户端

复用智防校园 server/services/ai_service.py 的通义千问调用模式
（OpenAI 兼容模式 + 重试 + 无 Key 降级），改造为 text-embedding 调用。
降级策略：无 Key 或调用失败时返回 None，检索层自动退化为纯关键词模式，
保证冒烟自测离线可跑。
"""
import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

DASHSCOPE_EMBED_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'


class EmbeddingClient:
    def __init__(self, api_key: str = '', model: str = 'text-embedding-v3',
                 timeout: int = 10, max_retries: int = 2):
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY', '')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def available(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith('your-')

    def embed(self, texts: list) -> list:
        """批量向量化。返回与 texts 等长的列表，失败项为 None。"""
        if not self.available or not texts:
            return [None] * len(texts)
        results = [None] * len(texts)
        # dashscope 单批上限：按 10 条分批
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            vectors = self._call(batch)
            if vectors is None:
                continue
            for i, vec in enumerate(vectors):
                results[start + i] = vec
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
