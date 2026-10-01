# -*- coding: utf-8 -*-
"""AML API 请求/响应模型

!!! 重要 !!!
以下字段是按 AML 公开评测管线推断的通用结构。
申请评测 Key 后，必须打开官网 API Guide 逐字段核对：
  - 请求字段名（user_id / task_id / session_id / query 等）
  - 响应包裹结构（是否有 status / code 字段）
  - 鉴权头格式（Bearer / X-API-Key）
  - 是否为异步任务 + 轮询接口
任何偏差都会导致冒烟测试失败。核对完成后同步修改本文件与 main.py。
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    speaker: str = Field(default='', description='说话人标识')
    content: str = Field(..., description='记忆原文')
    event_time: Optional[str] = Field(default=None, description='事件发生时间 ISO8601')


class AddRequest(BaseModel):
    user_id: str = Field(..., description='作用域：用户级隔离')
    task_id: str = Field(default='', description='任务标识')
    session_id: str = Field(default='', description='会话标识')
    messages: List[MessageIn] = Field(..., description='待写入的记忆批次')


class SearchRequest(BaseModel):
    user_id: str = Field(..., description='必须与 Add 的 user_id 一致')
    task_id: Optional[str] = Field(default=None, description='可选：限定任务范围')
    query: str = Field(..., description='检索问题')
    top_k: int = Field(default=20, ge=1, le=100)


class SearchHit(BaseModel):
    memory_id: int
    speaker: str = ''
    content: str
    keywords: List[str] = []
    event_time: Optional[str] = None
    ingested_at: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    results: List[SearchHit]
