# zhifang-aml-adapter

Agent Memory Challenge（CSIG，2026 Cycle 2）参赛适配器：把记忆引擎以标准化 **Add / Search** 接口接入 [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/) 评测平台。

## 来源与改动声明（AML 开源榜合规要求）

本项目复用了 **《基于微信生态的大学生AI反诈实训小程序开发及应用》**（宁夏大学 · 智防校园团队，指导教师：刘立波，负责人：王俊森）服务端的部分工程模式，具体为：

| 复用文件 | 改动说明 |
|---------|---------|
| `server/config.py` 的 dotenv 配置模式 | 改写为 `config.py`，全部指向独立数据库与独立密钥 |
| `server/services/ai_service.py` 的通义千问调用（兼容模式 + 重试 + 降级） | 改写为 `memory_core/embedding_client.py`，从 chat 调用改为 text-embedding 调用 |
| `server/models/*.py` 的 SQLAlchemy 建模范式 | 新写 `memory_core/models.py`，表结构完全不同 |

未复用任何业务代码、业务数据、密钥或反诈知识库内容。记忆引擎 `memory_core`（混合检索、双时间戳、记忆治理、用户隔离）为本项目新写。

## 结构

```
aml-adapter/
├── config.py               # dotenv 配置（隔离于主项目）
├── memory_core/            # 记忆引擎（参赛核心，无业务耦合）
│   ├── models.py           # MemoryRecord：双时间戳 + 冲突消解字段
│   ├── engine.py           # Add 管线 + 混合检索（向量+关键词+近因）
│   └── embedding_client.py # 通义千问 text-embedding（降级安全）
├── app/
│   ├── main.py             # FastAPI: POST /add, POST /search
│   └── schemas.py          # 请求/响应模型（字段待 AML Key 后校准！）
└── tests/
    ├── smoke.py            # 本地冒烟自测（离线可跑）
    └── local_eval.py       # LoCoMo 本地评测（Evidence Recall@K，调参用）
```

> 本地评测数据 `tests/locomo10.json` 不入库，请从 [snap-research/locomo](https://github.com/snap-research/locomo) 下载后放入 tests/ 目录。

## 快速开始

```bash
cd aml-adapter
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 1. 本地冒烟自测（sqlite + 关键词降级，无需任何 Key）
python tests/smoke.py

# 2. 启动服务（先复制 .env.example 为 .env 并填写）
uvicorn app.main:app --host 0.0.0.0 --port 8600
```

## 参赛待办（截至 2026-10-31）

- [ ] 官网申请评测 Key：<https://agentmemoryleaderboard.ai/evaluation>
- [ ] **按官网 API Guide 逐字段校准 `app/schemas.py`（当前为推断结构）**
- [ ] 配置独立 MySQL 库 `aml_eval`，部署到公网服务器（须保持可用至 11-04）
- [ ] 平台 Smoke 测试通过后，发起第一次 Full（不晚于 10-15）
- [ ] GitHub 公开仓库 + 固定 commit（开源榜要求）
