"""
意图识别测试集：覆盖 4 类意图 × 语义层/LLM 两条路径 × 边界陷阱用例。

用例分组：
    A. 语义层强信号（与种子句同义，应 ≥0.75 直接命中，零延迟）
    B. LLM 判定（语义 <0.75，应走 LLM 分类，每条约 1s）
    C. 边界陷阱（歧义词、问候+真实问题、混合表达）

运行：python tests/test_intent.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.core.intent_recognizer import IntentRecognizer

# (输入, 期望意图)
CASES = [
    # ---------- A. 语义层强信号（种子句同义，零延迟命中） ----------
    ("我想转人工", "request_human"),
    ("我要找真人客服", "request_human"),
    ("帮我转人工，我有点急", "request_human"),
    ("我的工单处理到哪了", "query_ticket"),
    ("帮我查询工单进度", "query_ticket"),
    ("帮我催一下报销单", "query_ticket"),
    ("工单编号是TK001，处理到哪了", "query_ticket"),
    ("你好呀", "chitchat"),
    ("在吗", "chitchat"),
    ("谢谢你", "chitchat"),

    # ---------- B. LLM 判定（语义 <0.75，走 LLM 分类） ----------
    ("人工审核要多久", "query_knowledge"),          # 歧义"人工"
    ("投诉处理流程是什么", "query_knowledge"),      # 歧义"投诉"
    ("工单系统怎么使用", "query_knowledge"),        # 歧义"工单"
    ("我想问一下怎么提交工单", "query_knowledge"),  # 歧义"工单"
    ("麻烦让个活人跟我聊", "request_human"),        # 口语化，语义层漏判，LLM 应懂
    ("服务器连不上帮我查下工单", "query_ticket"),   # 混合表达
    ("系统业务优势有哪些", "query_knowledge"),
    ("年假可以提前几天申请", "query_knowledge"),
    ("请问你们几点下班", "query_knowledge"),
    ("今天过得怎么样", "chitchat"),
    ("你吃饭了吗", "chitchat"),
    ("你是机器人吗", "chitchat"),

    # ---------- C. 边界陷阱 ----------
    ("你好，年假可以提前几天申请", "query_knowledge"),  # 问候+真实问题，不得判闲聊
    ("", "query_knowledge"),                              # 空输入兜底
]


def main() -> None:
    r = IntentRecognizer()
    passed = 0
    for q, expect in CASES:
        got = r.recognize(q)
        ok = got == expect
        passed += ok
        print(f"{'✓' if ok else '✗'} {q!r:35s} → {got:15s}（期望 {expect}）")
    total = len(CASES)
    print(f"\n通过 {passed}/{total}" + ("，全部通过 ✓" if passed == total else "，请检查 ✗ 项"))


if __name__ == "__main__":
    main()
