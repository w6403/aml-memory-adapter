# -*- coding: utf-8 -*-
"""AML 参赛适配器 API 入口

职责：把 AML 平台的 Add/Search 调用转交给 memory_core 引擎。
不含答案生成（由 AML 平台负责），不含任何智防校园业务逻辑。
"""
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import Config
from memory_core import MemoryEngine, EmbeddingClient
from app.schemas import AddRequest, SearchRequest, SearchResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('aml-adapter')

app = FastAPI(
    title='zhifang-aml-adapter',
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
    """鉴权：AML Key 下发后按官网规范校准此处。"""
    # TODO(AML契约): 核对鉴权头格式（Bearer / X-API-Key / query param）
    if Config.API_AUTH_TOKEN:
        auth = request.headers.get('Authorization', '')
        token = auth[7:] if auth.startswith('Bearer ') else auth
        if token != Config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail='unauthorized')


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'service': 'zhifang-aml-adapter',
        'embedding_available': embedder.available,
        'vector_weight': engine.vector_weight,
    }


@app.post('/add')
async def add_memory(payload: AddRequest, request: Request):
    await check_auth(request)
    try:
        result = engine.add(
            user_id=payload.user_id,
            task_id=payload.task_id,
            session_id=payload.session_id,
            messages=[m.model_dump() for m in payload.messages],
        )
        return {'status': 'ok', **result}
    except Exception as e:
        logger.exception('add failed')
        return JSONResponse(status_code=500, content={'status': 'error', 'detail': str(e)})


@app.post('/search', response_model=SearchResponse)
async def search_memory(payload: SearchRequest, request: Request):
    await check_auth(request)
    try:
        hits = engine.search(
            user_id=payload.user_id,
            query=payload.query,
            task_id=payload.task_id,
            top_k=payload.top_k,
        )
        return SearchResponse(results=hits)
    except Exception as e:
        logger.exception('search failed')
        return JSONResponse(status_code=500, content={'status': 'error', 'detail': str(e)})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=Config.PORT)
