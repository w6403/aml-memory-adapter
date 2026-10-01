# -*- coding: utf-8 -*-
"""记忆数据模型

表结构设计对齐 AML 文本赛道考点：
- 双时间戳（event_time / ingested_at）支撑时序推理（LoCoMo / LongMemEval）
- superseded_by 字段支撑记忆治理与冲突消解（Memory Governance）
- user_id 强制隔离，禁止跨用户检索（合规硬性要求）
写法参考智防校园 server/models/ 的 SQLAlchemy 模式。
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, DateTime, Boolean,
    JSON, Index
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MemoryRecord(Base):
    __tablename__ = 'memory_records'

    id = Column(BigInteger().with_variant(Integer, 'sqlite'), primary_key=True, autoincrement=True)

    # 作用域：评测记忆严格按 user_id 隔离，task_id 辅助过滤
    user_id = Column(String(128), nullable=False, index=True)
    task_id = Column(String(128), nullable=False, default='', index=True)
    session_id = Column(String(128), nullable=False, default='', index=True)

    # 记忆内容
    speaker = Column(String(64), default='')          # 说话人（user / assistant / agent）
    content = Column(Text, nullable=False)            # 原文
    keywords = Column(JSON, default=list)             # 抽取的关键词
    embedding = Column(JSON, default=list)            # 向量（JSON 存储，评测规模内足够）

    # 双时间戳：event_time = 事件发生时间（可从文本抽取）；ingested_at = 写入时间
    event_time = Column(DateTime, nullable=True)
    ingested_at = Column(DateTime, nullable=False, default=datetime.now)

    # 记忆治理：被更新的记忆置 False 并指向新记录
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    superseded_by = Column(BigInteger, nullable=True)

    __table_args__ = (
        Index('ix_mem_user_active', 'user_id', 'is_active'),
        Index('ix_mem_user_task', 'user_id', 'task_id'),
    )

    def to_dict(self):
        return {
            'memory_id': self.id,
            'user_id': self.user_id,
            'task_id': self.task_id,
            'session_id': self.session_id,
            'speaker': self.speaker or '',
            'content': self.content,
            'keywords': self.keywords or [],
            'event_time': self.event_time.isoformat() if self.event_time else None,
            'ingested_at': self.ingested_at.isoformat() if self.ingested_at else None,
            'is_active': self.is_active,
        }
