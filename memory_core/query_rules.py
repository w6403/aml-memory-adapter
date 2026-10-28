# -*- coding: utf-8 -*-
"""query 意图规则分类（v0.3.0 新增）

纯规则、无状态。Search 侧"条件注入"的触发器：仅当 query 命中某类特征时
才启用对应的加分/合并行为，常规 query 行为与 v0.2.0 完全一致（回归保护）。

类别：
- aggregate：聚合/总结类（救 F1 摘要压缩）→ 提升会话摘要记录
- recency：最新状态类（救 D1 新值覆盖的检索侧）→ 仅该 query 加 recency 分
- forget：遗忘相关问法（救 D3 回忆类）→ 提升带 '__forget__' 哨兵的指令记录
- conflict：矛盾/改口类（护 D2）→ 额外合并已被 supersede 的旧记录作证据
"""
import re

_AGGREGATE_ZH = re.compile(r'总结|摘要|概括|总体|主要(?:内容|话题)|都(?:聊了|讨论|提到|说了)')
_AGGREGATE_EN = re.compile(
    r"\bsummar\w*\b|\boverview\b|\bin\s+general\b|\boverall\b"
    r"|\bmain\s+topics?\b|\bwhat\s+(?:did|have)\s+we\s+(?:discuss|talk)\b"
    r"|\beverything\s+we\b", re.I)

_RECENCY_ZH = re.compile(r'现在|目前|最近|最新|当前|眼下')
_RECENCY_EN = re.compile(
    r"\bnow\b|\bcurrently\b|\blatest\b|\bcurrent\b|\brecently\b"
    r"|\bthese\s+days\b|\bat\s+the\s+moment\b", re.I)

_FORGET_ZH = re.compile(r'忘记|忘掉|忘了|删除|删掉')
_FORGET_EN = re.compile(r"\b(forget|forgot|forgotten|erase|delete|remove)\b", re.I)

_CONFLICT_ZH = re.compile(r'矛盾|冲突|不一致|改口|反悔|前后不符')
_CONFLICT_EN = re.compile(
    r"\bcontradict\w*\b|\bconflict\w*\b|\binconsistent\b"
    r"|\bchanged\s+(?:his|her|their)\s+mind\b|\bdiscrepan\w+\b", re.I)


def classify_query(query: str) -> set:
    """返回命中的特征集合（可能为空集 / 多标签）。"""
    if not query:
        return set()
    flags = set()
    if _AGGREGATE_ZH.search(query) or _AGGREGATE_EN.search(query):
        flags.add('aggregate')
    if _RECENCY_ZH.search(query) or _RECENCY_EN.search(query):
        flags.add('recency')
    if _FORGET_ZH.search(query) or _FORGET_EN.search(query):
        flags.add('forget')
    if _CONFLICT_ZH.search(query) or _CONFLICT_EN.search(query):
        flags.add('conflict')
    return flags
