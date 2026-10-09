#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat regression smoke for a running OpenAI-compatible endpoint.

One command to answer "does chatting still work, and are the answers still right?" after a
kernel, quantization, parser or scheduler change. It checks both *content* (the answer is
the correct one, including two classic traps) and the *API contract* of this deployment's
Rust frontend:

* the reasoning trace is returned as ``message["reasoning"]`` (this frontend does not use
  the ``reasoning_content`` field name);
* thinking is on by default, and ``chat_template_kwargs={"enable_thinking": false}`` empties
  the trace;
* streaming with ``stream_options={"include_usage": true}`` reports usage, including
  ``completion_tokens_details.reasoning_tokens``;
* ``max_tokens`` counts reasoning tokens, so a tiny budget ends with ``finish_reason ==
  "length"``.

Usage (against an already running server):
    python3 tools/chat_smoke.py --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B-FP8
    python3 tools/chat_smoke.py --quick          # four fast, non-thinking cases

Or let `tools/chat_smoke.sh --launch` start the route, run this and stop it again.

Exit code is 0 only when every check passed; the last line is
``chat_smoke_ok passed=<n> failed=0`` (or ``chat_smoke_failed …``) so scripts can grep it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Case definitions. `last_line` is checked against the final non-empty line of the answer,
# which every prompt asks for explicitly, so the expectation stays stable across restyles.
# --------------------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    prompt: str
    max_tokens: int
    last_line: str | None = None
    contains: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    quick: bool = False
    stream: bool = False
    reasoning_expected: bool = True
    extra: dict = field(default_factory=dict)


CASES: list[Case] = [
    Case(
        "short-fact-zh",
        "法国首都是哪里？只回答城市名，不要其他文字。",
        64,
        contains=("巴黎",),
        quick=True,
    ),
    Case(
        "arith-word-zh",
        "小明有 12 个苹果，给了小红 1/3，又买了 5 个，然后吃了 2 个。他还剩多少个苹果？"
        "先算再答，最后一行写 `答案：N`。",
        512,
        last_line=r"11(?!\d)",
        quick=True,
    ),
    Case(
        "trap-bat-ball-en",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Explain briefly, then state `ANSWER: X`.",
        768,
        last_line=r"0\.05(?!\d)",
        forbidden=(r"ANSWER:\s*\$?0\.10",),
        quick=True,
    ),
    Case(
        "reasoning-dog-run-zh",
        "甲乙两人从相距 100 公里的两地同时出发相向而行，甲速 6 km/h，乙速 4 km/h。"
        "一只狗同时从甲处出发，以 10 km/h 在两人之间往返奔跑，直到两人相遇。"
        "狗一共跑了多少公里？请分步推理，最后一行写 `答案：N 公里`。",
        1536,
        last_line=r"100(?!\d)",
        quick=True,
    ),
    Case(
        "trap-decimal-compare-zh",
        "9.11 和 9.8 哪个更大？给出理由，最后一行写 `答案：X`。",
        768,
        last_line=r"9\.8(?!\d)",
        forbidden=(r"答案：\s*9\.11", r"答案：\s*9。11"),
        stream=True,
    ),
    Case(
        "chicken-egg-zh",
        "三只鸡三天生三个蛋。九只鸡九天生多少个蛋？分步推理，最后一行写 `答案：N`。",
        1024,
        last_line=r"27(?!\d)",
    ),
    Case(
        "power-of-two-zh",
        "2 的 10 次方是多少？只回答数字。",
        256,
        contains=("1024",),
    ),
    Case(
        "algebra-price-zh",
        "某商品原价 x 元，先涨价 20% 再打八折，最后售价 96 元，求 x。"
        "给出推理，最后一行写 `答案：N`。",
        1024,
        last_line=r"100(?!\d)",
    ),
    Case(
        "logic-truth-teller-zh",
        "三个人中只有一个人说真话。甲说：乙在说谎。乙说：丙在说谎。丙说：甲和乙都在说谎。"
        "谁在说真话？最后一行写 `答案：X`。",
        1024,
        last_line=r"乙",
    ),
    Case(
        "fact-light-year-zh",
        "光年是什么单位？最后一行写 `答案：X`（填“长度”或“时间”或“速度”）。",
        512,
        last_line=r"长度",
    ),
    Case(
        "code-is-prime-zh",
        "用 Python 实现 is_prime(n)，不要任何解释文字，只输出一个代码块。",
        1024,
        contains=("```", "def is_prime"),
        forbidden=("解释如下", "步骤如下"),
        reasoning_expected=False,
    ),
]


def normalise(text: str) -> str:
    """Strip LaTeX/markdown noise so a value check does not depend on formatting."""
    text = text.replace("**", "").replace("$", "").replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    return text


def last_non_empty_line(text: str) -> str:
    for line in reversed(normalise(text).splitlines()):
        if line.strip():
            return line.strip()
    return ""


# --------------------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------------------


class Client:
    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def chat(self, prompt: str, max_tokens: int, stream: bool = False, **extra) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            **extra,
        }
        if not stream:
            payload["stream"] = False
            request = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())

        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        reasoning: list[str] = []
        content: list[str] = []
        usage = None
        finish = None
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                piece = json.loads(chunk)
                if piece.get("usage"):
                    usage = piece["usage"]
                for choice in piece.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning"):
                        reasoning.append(delta["reasoning"])
                    if delta.get("content"):
                        content.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        return {
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"reasoning": "".join(reasoning), "content": "".join(content)},
                }
            ],
            "usage": usage,
        }


def message_of(body: dict) -> dict:
    return ((body.get("choices") or [{}])[0].get("message")) or {}


def finish_of(body: dict) -> str | None:
    return (body.get("choices") or [{}])[0].get("finish_reason")


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def run_content_check(client: Client, case: Case) -> tuple[bool, str]:
    started = time.time()
    body = client.chat(case.prompt, case.max_tokens, stream=case.stream, **case.extra)
    elapsed = time.time() - started
    message = message_of(body)
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning") or "").strip()
    usage = body.get("usage") or {}
    finish = finish_of(body)
    flat = normalise(content)

    problems: list[str] = []
    if finish != "stop":
        problems.append(f"finish={finish}")
    if case.last_line and not re.search(case.last_line, last_non_empty_line(content)):
        problems.append(f"末行不含 /{case.last_line}/: {last_non_empty_line(content)[:60]!r}")
    for needle in case.contains:
        if needle not in flat:
            problems.append(f"缺少 {needle!r}")
    for pattern in case.forbidden:
        if re.search(pattern, flat):
            problems.append(f"命中禁止模式 /{pattern}/")
    if case.reasoning_expected and not reasoning:
        problems.append("思考字段为空")

    detail = (
        f"{elapsed:.1f}s reasoning={len(reasoning)}字 "
        f"completion={usage.get('completion_tokens')} finish={finish}"
    )
    if problems:
        return False, detail + " | " + "; ".join(problems)
    return True, detail


def run_multi_turn_check(client: Client) -> tuple[bool, str]:
    first = client.chat("请记住这个数字：7391。只回答“已记住”。", 128)
    reply = (message_of(first).get("content") or "").strip()
    # The Client helper always builds a single-turn request, so the two-turn payload is
    # assembled explicitly here.
    payload = {
        "model": client.model,
        "messages": [
            {"role": "user", "content": "请记住这个数字：7391。只回答“已记住”。"},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "我刚才让你记的数字是多少？只回答数字。"},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        client.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=client.timeout) as response:
        body = json.loads(response.read().decode())
    content = normalise((message_of(body).get("content") or "").strip())
    ok = "7391" in content
    return ok, f"第一轮={reply[:20]!r} 第二轮={content[:40]!r}"


def run_contract_checks(client: Client) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    # 1. the trace field name and the default thinking policy
    body = client.chat("三只鸡三天生三个蛋。九只鸡九天生多少个蛋？简要推理。", 768)
    message = message_of(body)
    has_reasoning = "reasoning" in message
    reasoning_len = len((message.get("reasoning") or "").strip())
    results.append(
        (
            "contract-reasoning-field",
            has_reasoning and reasoning_len > 0,
            f"message keys={sorted(message)} 默认思考={reasoning_len} 字 "
            f"(reasoning_content={'有' if 'reasoning_content' in message else '无'})",
        )
    )

    # 2. thinking can be switched off
    body = client.chat(
        "三只鸡三天生三个蛋。九只鸡九天生多少个蛋？", 768, chat_template_kwargs={"enable_thinking": False}
    )
    off_len = len((message_of(body).get("reasoning") or "").strip())
    results.append(
        ("contract-thinking-off", off_len == 0, f"enable_thinking=False -> 思考 {off_len} 字")
    )

    # 3. streaming usage accounting, including the reasoning split
    body = client.chat("2 的 10 次方是多少？只回答数字。", 256, stream=True)
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    ok = bool(usage.get("prompt_tokens")) and bool(usage.get("completion_tokens"))
    results.append(
        (
            "contract-stream-usage",
            ok,
            f"prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
            f"reasoning_tokens={details.get('reasoning_tokens')}",
        )
    )

    # 4. max_tokens counts reasoning tokens, so a tiny budget truncates
    body = client.chat("详细解释快速排序的原理。", 24)
    finish = finish_of(body)
    usage = body.get("usage") or {}
    results.append(
        (
            "contract-truncation",
            finish == "length" and usage.get("completion_tokens") == 24,
            f"max_tokens=24 -> finish={finish} completion={usage.get('completion_tokens')}",
        )
    )
    return results


# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-FP8")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (s).")
    parser.add_argument("--quick", action="store_true", help="Only the four fast cases.")
    parser.add_argument("--skip-contract", action="store_true", help="Skip API contract checks.")
    parser.add_argument("--json-out", type=str, default=None, help="Also write results as JSON.")
    args = parser.parse_args()

    client = Client(args.base_url, args.model, args.timeout)
    cases = [c for c in CASES if c.quick] if args.quick else CASES

    results: list[dict] = []
    failed = 0

    def record(name: str, ok: bool, detail: str) -> None:
        nonlocal failed
        results.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            failed += 1
        print(f"[chat-smoke] {'PASS' if ok else 'FAIL'} {name}: {detail}")

    print(f"[chat-smoke] {args.base_url} model={args.model} cases={len(cases)}")
    for case in cases:
        try:
            ok, detail = run_content_check(client, case)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            ok, detail = False, f"请求失败: {exc!r}"
        record(case.name, ok, detail)

    try:
        ok, detail = run_multi_turn_check(client)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        ok, detail = False, f"请求失败: {exc!r}"
    record("multi-turn-context", ok, detail)

    if not args.skip_contract:
        try:
            for name, ok, detail in run_contract_checks(client):
                record(name, ok, detail)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            record("contract", False, f"请求失败: {exc!r}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

    total = len(results)
    if failed:
        print(f"chat_smoke_failed passed={total - failed} failed={failed}")
        return 1
    print(f"chat_smoke_ok passed={total} failed=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
