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
    # v0.3.0 起默认 text-embedding-v4（开源方法榜合规要求，Qwen3-Embedding 系）；
    # 默认维度 1024 与 v3 相同，显式传 dimensions 作护栏（防默认值变动）
    DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-v4')
    EMBEDDING_DIMENSIONS = int(os.getenv('EMBEDDING_DIMENSIONS', '1024'))
    EMBEDDING_TIMEOUT = 10  # 秒
    EMBEDDING_MAX_RETRIES = 2

    # API 鉴权：AML 下发 Key 后替换为平台要求的鉴权方式
    # TODO(AML契约): 申请评测 Key 后按官网 API Guide 校准鉴权头与字段
    API_AUTH_TOKEN = os.getenv('API_AUTH_TOKEN', '')

    PORT = int(os.getenv('PORT', '8600'))

    # 检索参数（2026-10-02 按 LoCoMo Recall@100 权重扫描确定，详见材料文档 §四）
    VECTOR_WEIGHT = 0.5      # 向量相似度权重
    KEYWORD_WEIGHT = 0.5     # 关键词重合权重（IDF 加权）
    RECENCY_WEIGHT = 0.0     # 时间近因权重（LoCoMo 证据时间随机分布，此项为噪声）

    # ---- v0.3.0 记忆治理（D1 新值覆盖 / D3 删除遗忘）：全部条件触发 ----
    GOVERNANCE_ENABLED = os.getenv('GOVERNANCE_ENABLED', 'true').lower() == 'true'
    FORGET_SIM_THRESHOLD = 0.45    # 遗忘关键词驱动的相似度下限（需共享 >=2 内容关键词）
    FORGET_STRONG_SIM = 0.85       # 遗忘直接命中阈值（无需关键词交集）
    FORGET_MAX_TARGETS = 3         # 每条遗忘指令最多 supersede 条数
    UPDATE_SIM_THRESHOLD = 0.55    # 新值覆盖相似度下限（v4 实测真实覆盖对 ~0.6，异主题 ~0.39）
    UPDATE_MIN_SHARED_KWS = 2      # 覆盖判定的关键词交集下限
    DEDUP_SIM_THRESHOLD = 0.97     # 近同文去重阈值

    # ---- v0.3.0 摘要压缩（F1）与 query 条件注入 ----
    SUMMARY_ENABLED = os.getenv('SUMMARY_ENABLED', 'true').lower() == 'true'
    SUMMARY_MIN_MESSAGES = 20      # session 累计消息达到此数才生成摘要
    SUMMARY_MAX_SENTS = 5          # 摘要代表句数
    QUERY_BOOST_ENABLED = os.getenv('QUERY_BOOST_ENABLED', 'true').lower() == 'true'
    SUMMARY_BOOST = 0.5            # aggregate query 下摘要记录乘法提升（否则 ×0.8）
    RECENCY_QUERY_BOOST = 0.15     # recency query 的加法近因分
    FORGET_QUERY_BOOST = 0.15      # forget query 下遗忘指令记录的加法提升
