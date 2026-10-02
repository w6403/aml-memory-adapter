# -*- coding: utf-8 -*-
"""AML 参赛适配器 API 入口

职责：把 AML 平台的 Add/Search 调用转交给 memory_core 引擎。
不含答案生成（由 AML 平台负责），不含任何智防校园业务逻辑。
请求/响应契约见 app/schemas.py 头注释（2026-10-01 按官网 API Guide 校准）。
"""
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import Config
from memory_core import MemoryEngine, EmbeddingClient
from app.schemas import AddRequest, AddResponse, SearchRequest, SearchResponse, SearchHit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('aml-adapter')

app = FastAPI(
    title='aml-memory-adapter',
    description='Agent Memory Challenge 参赛适配器（memory engine: memory-core）',
    version='0.1.0',
)

embedder = EmbeddingClient(
    api_key=Config.DASHSCOPE_API_KEY,
    model=Config.EMBEDDING_MODEL,
    timeout=Config.EMBEDDING_TIMEOUT,
    max_retries=Config.EMBEDDING_MAX_RETRIES,
)
engine = MemoryEngine(
    database_uri=Config.DATABASE_URI,
    embedding_client=embedder,
    vector_weight=Config.VECTOR_WEIGHT,
    keyword_weight=Config.KEYWORD_WEIGHT,
    recency_weight=Config.RECENCY_WEIGHT,
)


async def check_auth(request: Request):
    """鉴权：契约支持 Token / Bearer / X-Api-Key 三种携带方式，
    与申请时声明的方式保持一致即可；这里三种都接受，比对同一个密钥值。"""
    if Config.API_AUTH_TOKEN:
        auth = request.headers.get('Authorization', '')
        token = auth[7:] if auth.startswith(('Bearer ', 'bearer ')) else \
            auth[6:] if auth.startswith(('Token ', 'token ')) else \
            request.headers.get('X-Api-Key', '')
        if token != Config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail='unauthorized')


@app.get('/health')
async def health():
    """契约：无鉴权 GET，任意 2xx 即正常。"""
    return {
        'status': 'ok',
        'service': 'aml-memory-adapter',
        'embedding_available': embedder.available,
        'vector_weight': engine.vector_weight,
    }


@app.post('/add', response_model=AddResponse)
async def add_memory(payload: AddRequest, request: Request):
    await check_auth(request)
    try:
        messages = []
        for m in payload.messages:
            msg = {'speaker': m.role, 'content': m.content}
            if m.timestamp:  # Unix 毫秒 -> naive datetime（UTC）
                msg['event_time'] = datetime.fromtimestamp(
                    m.timestamp / 1000, tz=timezone.utc).replace(tzinfo=None).isoformat()
            messages.append(msg)
        engine.add(
            user_id=payload.user_id,
            task_id='',  # 现行契约不发送 task_id
            session_id=payload.session_id,
            messages=messages,
        )
        # 同步语义：engine.add 提交完成后才走到这里，记忆立即可检索
        return AddResponse(
            success=True,
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )
    except Exception as e:
        logger.exception('add failed')
        return JSONResponse(status_code=500, content={'detail': {'reason': str(e)}})


@app.post('/search', response_model=SearchResponse)
async def search_memory(payload: SearchRequest, request: Request):
    await check_auth(request)
    try:
        hits = engine.search(
            user_id=payload.user_id,
            query=payload.query,
            top_k=payload.top_k,  # engine 内部切片，保证不超过 top_k
        )
        data = [
            SearchHit(
                id=str(h['memory_id']),
                content=h['content'],
                score=h['score'],
                created_at=h['event_time'] or h['ingested_at'],
            )
            for h in hits
        ]
        return SearchResponse(data=data)
    except Exception as e:
        logger.exception('search failed')
        return JSONResponse(status_code=500, content={'detail': {'reason': str(e)}})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=Config.PORT)
