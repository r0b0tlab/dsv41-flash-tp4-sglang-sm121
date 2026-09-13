#!/usr/bin/env python3
"""Post-readiness semantic ladder for dsv4.1-flash.

Arithmetic -> code -> tool-call JSON -> vision (two-image order) -> effort
control probe -> determinism. Any corrupt/repetitive output is a hard FAIL:
it blocks every downstream benchmark and publication work.

Usage: python3 scripts/boot_smoke.py [--base-url http://127.0.0.1:30000]
Exit 0 = all gates PASS. Each gate prints PASS/FAIL with evidence.
"""

import argparse
import base64
import io
import json
import sys
import time
import urllib.request

import numpy as np


def chat(base_url, payload, timeout=600):
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def png_data_url(color: str) -> str:
    """Tiny solid-color PNG as a data URL (no PIL dependency)."""
    import struct
    from zlib import compress, crc32

    w = h = 64
    rgb = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}[color]
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk2(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk2(b"IHDR", ihdr)
           + chunk2(b"IDAT", compress(raw)) + chunk2(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def gate(name, ok, evidence):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as r:
        models = json.loads(r.read())
    model = args.model or models["data"][0]["id"]
    print(f"served model: {model}")

    results = []

    # 1. arithmetic
    r = chat(base, {"model": model, "max_tokens": 64, "temperature": 0,
                    "messages": [{"role": "user", "content": "What is 19 + 23? Answer with the number only."}]})
    txt = r["choices"][0]["message"]["content"] or ""
    results.append(gate("arithmetic", "42" in txt, repr(txt[:80])))

    # 2. code gen
    r = chat(base, {"model": model, "max_tokens": 256, "temperature": 0,
                    "messages": [{"role": "user", "content": "Write a Python function is_palindrome(s). Code only."}]})
    txt = r["choices"][0]["message"]["content"] or ""
    ok = "def " in txt and ("s[::-1]" in txt or "reversed" in txt or "s == s" in txt)
    results.append(gate("codegen", ok, repr(txt[:80])))

    # 3. tool call JSON
    r = chat(base, {"model": model, "max_tokens": 256, "temperature": 0,
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather", "description": "Get weather",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}}],
                    "messages": [{"role": "user", "content": "What's the weather in Paris?"}]})
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "paris" in tc[0]["function"]["arguments"].lower()
    results.append(gate("toolcall", ok, json.dumps(tc[:1])[:120] if tc else repr(msg.get("content", ""))[:80]))

    # 4. vision: two images, order matters
    r = chat(base, {"model": model, "max_tokens": 64, "temperature": 0,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "I show you two solid-color images in order. Reply with exactly: first=<color1>, second=<color2>."},
                        {"type": "image_url", "image_url": {"url": png_data_url("red")}},
                        {"type": "image_url", "image_url": {"url": png_data_url("blue")}}]}]})
    txt = (r["choices"][0]["message"]["content"] or "").lower()
    ok = "first=red" in txt.replace(" ", "") and "second=blue" in txt.replace(" ", "")
    ptok = r.get("usage", {}).get("prompt_tokens", -1)
    results.append(gate("vision-order", ok, f"{txt[:60]!r} prompt_tokens={ptok}"))

    # 5. vision counterfactual (swap order)
    r = chat(base, {"model": model, "max_tokens": 64, "temperature": 0,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "I show you two solid-color images in order. Reply with exactly: first=<color1>, second=<color2>."},
                        {"type": "image_url", "image_url": {"url": png_data_url("blue")}},
                        {"type": "image_url", "image_url": {"url": png_data_url("red")}}]}]})
    txt = (r["choices"][0]["message"]["content"] or "").lower().replace(" ", "")
    ok = "first=blue" in txt and "second=red" in txt
    results.append(gate("vision-counterfactual", ok, repr(txt[:60])))

    # 6. effort control (no Jinja template: prove the knob reaches encoding).
    # V4.1 reasoning_effort scales generated reasoning, not prompt length:
    # compare reasoning_content length on a reasoning-heavy prompt.
    rl = {}
    for effort in (5, 100):
        r = chat(base, {"model": model, "max_tokens": 1500, "temperature": 0,
                        "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort},
                        "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]})
        rl[effort] = len(r["choices"][0]["message"].get("reasoning_content") or "")
    results.append(gate("effort-control", rl[100] > rl[5] * 1.2,
                        f"reasoning_len 5->{rl[5]} 100->{rl[100]}"))

    # 7. determinism: same greedy prompt x3
    outs = []
    for _ in range(3):
        r = chat(base, {"model": model, "max_tokens": 48, "temperature": 0,
                        "messages": [{"role": "user", "content": "Name three primary colors, comma-separated."}]})
        outs.append(r["choices"][0]["message"]["content"])
    results.append(gate("determinism", len(set(outs)) == 1, f"{len(set(outs))} distinct outputs"))

    print(f"\n{sum(results)}/{len(results)} gates passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
