# -*- coding: utf-8 -*-
"""AML 参赛适配器配置

复用自智防校园 server/config.py 的 dotenv 配置模式（宁夏大学·智防校园团队）。
本配置与主项目完全隔离：独立数据库、独立密钥，不读取主项目任何 .env。
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    # 评测数据存储：独立 MySQL 库（aml_eval），本地自测可回退 sqlite
    DATABASE_URI = os.getenv(
        'DATABASE_URI',
        'sqlite:///./aml_eval.db'
    )

    # 通义千问 embedding（与主项目共用 dashscope 平台，但 Key 独立申请）
    DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-v3')
    EMBEDDING_TIMEOUT = 10  # 秒
    EMBEDDING_MAX_RETRIES = 2

    # API 鉴权：AML 下发 Key 后替换为平台要求的鉴权方式
    # TODO(AML契约): 申请评测 Key 后按官网 API Guide 校准鉴权头与字段
    API_AUTH_TOKEN = os.getenv('API_AUTH_TOKEN', '')

    PORT = int(os.getenv('PORT', '8600'))

    # 检索参数（2026-10-02 按 LoCoMo Recall@100 权重扫描确定，详见材料文档 §四）
    TOP_K_DEFAULT = 20
    VECTOR_WEIGHT = 0.5      # 向量相似度权重
    KEYWORD_WEIGHT = 0.5     # 关键词重合权重（IDF 加权）
    RECENCY_WEIGHT = 0.0     # 时间近因权重（LoCoMo 证据时间随机分布，此项为噪声）
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 3600,
        'pool_pre_ping': True,
    }
