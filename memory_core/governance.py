# -*- coding: utf-8 -*-
"""记忆治理规则模块（v0.3.0 新增）

纯规则、无状态、不依赖 LLM（开源方法榜合规：LLM 组件须 gpt-4o-mini，
本管线保持零生成式 LLM）。提供两类判定：

1. 遗忘/删除指令检测（强弱分级）：
   - 强命中（明确指令："删掉刚才说的"、"forget what I said"）→ 触发遗忘动作
   - 弱命中（裸词 "forget/忘记"，可能是 "I'll never forget..." 这类正常陈述）
     → 不触发动作，仅由调用方在 keywords 里追加 '__forget__' 哨兵供检索侧提升
2. 新值覆盖判定：新记录与旧记录"同说话人 + 高相似 + 有关键词交集但非同文"
   才认定旧值被覆盖，四道闸防误杀（误 supersede 会直接抹掉可检索证据）。
"""
import re

# 遗忘指令检测：中英文模式
# 强：明确的对象指代（that/this/what I said/刚才说的），动作意图清晰
FORGET_STRONG_ZH = re.compile(
    r'删除|删掉|抹去|清除|别再提|不要再提|别记|不要记|当我没说|撤回|作废')
FORGET_STRONG_EN = re.compile(
    r"\b(erase|delete|wipe|remove)\s+(that|this|it|my|the|what)\b"
    r"|\bforget\s+(about\s+)?(that|this|it|what\s+i)\b"
    r"|\bdon'?t\s+(remember|mention|bring\s+up)\b"
    r"|\bdo\s+not\s+(remember|mention)\b"
    r"|\bnever\s+mind\b|\bscratch\s+that\b"
    r"|\bdisregard\s+(that|this|what)\b"
    r"|\bpretend\s+i\s+(never|didn'?t)\b"
    r"|\bignore\s+(that|what\s+i\s+said)\b",
    re.I)
# 弱：裸 forget/忘记 词，需要人判断语境，不触发动作
FORGET_WEAK_ZH = re.compile(r'忘记|忘掉|忘了')
FORGET_WEAK_EN = re.compile(r'\b(forget|forgot|forgotten)\b', re.I)

# 覆盖触发词：新记录本身必须带"状态变更"语义才允许走覆盖判定。
# LoCoMo 回归教训（2026-10-28）：裸"高相似+关键词交集"在自然对话里
# 误杀率灾难级（2014 次/1986 题，overall 0.879→0.684）——同话题回声
# 与真更新无法用 embedding 区分，变更措辞标记才是可分信号。
UPDATE_TRIGGER_ZH = re.compile(
    r'不再是|改成|换了|搬家|搬去|搬到|调任|转行|辞职|分手|结婚|离婚|'
    r'已经不|不再|改成|变更为|更新为|现在是|目前是')
UPDATE_TRIGGER_EN = re.compile(
    r"\b(no longer|not anymore|moved|switched|changed|updated|now lives?|"
    r"now works?|just got|recently became|used to .{1,40} but now|"
    r"my new (job|home|address|city|car|phone|number|email))\b", re.I)


def has_update_trigger(text: str) -> bool:
    """新记录是否自带状态变更措辞（覆盖判定的前置闸）。"""
    if not text:
        return False
    return bool(UPDATE_TRIGGER_ZH.search(text) or UPDATE_TRIGGER_EN.search(text))


# 哨兵：写入 keywords JSON（内部字段，不影响 Search 返回的 content），
# 检索侧 query 命中 forget 特征时据此提升
FORGET_SENTINEL = '__forget__'

# 触发遗忘动作的说话人角色：评测对话里指令来自用户侧
# （assistant 说 "let me forget that" 不构成用户授权）
FORGET_ACTOR_ROLES = {'user'}


def classify_forget(text: str, speaker: str = '') -> str:
    """分级检测遗忘/删除指令。返回 'strong' | 'weak' | ''。

    strong 仅在指令发出者是用户角色时出现；weak 不区分角色。
    """
    if not text:
        return ''
    if speaker.lower() in FORGET_ACTOR_ROLES:
        if FORGET_STRONG_ZH.search(text) or FORGET_STRONG_EN.search(text):
            return 'strong'
    if FORGET_WEAK_ZH.search(text) or FORGET_WEAK_EN.search(text):
        return 'weak'
    return ''


def shared_keywords(kws_a: list, kws_b: list, exclude: set = None) -> set:
    """两组关键词的交集（小写规范化），可排除说话人名等泛词。"""
    ex = exclude or set()
    a = {w.lower() for w in (kws_a or [])} - ex
    b = {w.lower() for w in (kws_b or [])} - ex
    return a & b


def is_update_candidate(cosine: float, new_kws: list, old_kws: list,
                        same_speaker: bool,
                        sim_threshold: float = 0.82,
                        dedup_threshold: float = 0.97,
                        min_shared_kws: int = 2) -> str:
    """新值覆盖/去重判定。返回 'dedup' | 'update' | ''。

    四道闸：相似度区间 + 同说话人 + 关键词交集 + （调用方保证）每条上限。
    - dedup：同说话人近同文（cosine >= dedup_threshold），旧记录 supersede
      防评测重放造成的重复证据稀释；跨说话人同文（"Yes"类短句）不动
    - update：paraphrase 级相似但非同文（[sim, dedup)），且同说话人、
      关键词交集达标，认定新值覆盖旧值
    """
    if not same_speaker:
        return ''
    if cosine >= dedup_threshold:
        return 'dedup'
    if cosine < sim_threshold:
        return ''
    if len(shared_keywords(new_kws, old_kws)) < min_shared_kws:
        return ''
    return 'update'
