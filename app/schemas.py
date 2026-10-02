# -*- coding: utf-8 -*-
"""AML API 请求/响应模型

2026-10-01 已按官网 API Guide（Add/Search 接入规范）逐字段校准：
  - Add 请求：request_id / messages[role, content, timestamp?] / user_id / session_id
  - Add 响应：{success: true, request_id, user_id, session_id}（echo 必须与请求一致）
  - Search 请求：query / options? / user_id / top_k（必填，正式评测固定 100）
  - Search 响应：{data: [{id, content, score?, created_at?}]}，无 items 包装层
  - 未声明字段（metadata/filters/rerank 等）平台不发送；响应多余字段被平台忽略
  - 返回数量不得超过 top_k，超量判契约错误
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    role: str = Field(..., description='说话人角色（平台契约字段）')
    content: str = Field(..., min_length=1, description='记忆原文，非空')
    timestamp: Optional[int] = Field(default=None, description='Unix 毫秒时间戳')


class AddRequest(BaseModel):
    request_id: str = Field(..., description='写入请求唯一标识，响应需原样返回')
    messages: List[MessageIn] = Field(..., description='按原顺序排列的消息批次')
    user_id: str = Field(..., description='检索范围标识，Search 必须使用相同值')
    session_id: str = Field(..., description='来源会话标识（不作为 Search 筛选条件）')


class AddResponse(BaseModel):
    success: bool = Field(..., description='必须为布尔 true')
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(..., description='基准原文查询')
    options: Optional[List[str]] = Field(default=None, description='选择题选项（开放题不发送）')
    user_id: str = Field(..., description='只能在该 user_id 范围内检索')
    top_k: int = Field(..., ge=1, le=100, description='返回数量上限，正式评测固定 100')


class SearchHit(BaseModel):
    id: str = Field(..., description='非空字符串，稳定标识该条记忆')
    content: str = Field(..., min_length=1)
    score: Optional[float] = Field(default=None, description='越大越相关')
    created_at: Optional[str] = Field(default=None, description='记忆来源时间或持久化时间')


class SearchResponse(BaseModel):
    data: List[SearchHit] = Field(..., description='按相关性排序，无结果时为空数组')
