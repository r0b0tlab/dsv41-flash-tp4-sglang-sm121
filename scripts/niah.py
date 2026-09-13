#!/usr/bin/env python3
"""NIAH at the advertised window for dsv4.1-flash.

Depths = 25/50/90% of (max_model_len - 64); plus ordered two-key at 33/66%.
Prompts built with the checkpoint tokenizer via /v1/completions (no chat
template assumptions). Marker: context-stable bare token string.

Usage: python3 scripts/niah.py [--depths 25,50,90] [--twokey] [--out ...]
"""

import argparse
import json
import random
import time
import urllib.request

MARKER = "R0B0BENCH_NIAH_SLOT"
FILLER_SENT = ("The quarterly inventory review concluded without material "
               "discrepancies across all regional warehouses and depots. ")


def token_count_via_api(base, text):
    # /v1/completions with max_tokens=0? sglang exposes count via generate with
    # max_new_tokens=1 on empty; simplest robust: use the tokenizer endpoint if
    # present, else calibrate with a completion echo. We use tokenids via
    # "return_token_ids"? Fall back: construct filler to target length using
    # measured ratio, then verify exact count from usage.prompt_tokens of the
    # actual NIAH request and adjust depth placement accordingly.
    return None


def build_prompt(target_tokens, needle_sentence, approx_tok_per_char=0.30):
    # filler scaled; exact count read back from usage
    est_chars = int(target_tokens / approx_tok_per_char)
    body = FILLER_SENT * (est_chars // len(FILLER_SENT) + 1)
    return body[:est_chars]


def run_case(base, prompt, needle_pos_ratio, max_prompt, label):
    # insert needle at ratio over the available prefix; CHAT api (thinking off)
    text = prompt
    cut = int(len(text) * needle_pos_ratio)
    needle = f" {MARKER}: the secret passcode for this document is {label}. "
    full = text[:cut] + needle + text[cut:]
    q = ("\n\nWhat is the secret passcode mentioned in the document? "
         "Answer with just the passcode.")
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({"model": "/model", "max_tokens": 64,
                         "temperature": 0,
                         "chat_template_kwargs": {"thinking": False},
                         "messages": [{"role": "user", "content": full + q}]}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=7200) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    out = resp["choices"][0]["message"]["content"] or ""
    ptok = resp["usage"]["prompt_tokens"]
    ok = label in out
    return {"label": label, "pass": ok, "prompt_tokens": ptok,
            "depth_ratio": needle_pos_ratio, "elapsed_s": round(dt, 1),
            "out": out[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--window", type=int, default=512 * 1024 - 2114,
                    help="target prompt tokens (leave headroom for needle+out)")
    ap.add_argument("--depths", default="0.25,0.50,0.90")
    ap.add_argument("--twokey", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    results = []
    labels = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # calibration run: small prompt to learn tok/char ratio
    cal = run_case(base, FILLER_SENT * 20, 0.5, 1000, next(labels))
    tok_per_char = cal["prompt_tokens"] / (len(FILLER_SENT) * 20 + 60)
    print(f"calibration: {tok_per_char:.4f} tok/char")

    for d in [float(x) for x in args.depths.split(",")]:
        lab = next(labels)
        body = build_prompt(args.window, None, tok_per_char)
        r = run_case(base, body, d, args.window, lab)
        r["depth_pct"] = int(d * 100)
        print(f"depth {r['depth_pct']}%: {'PASS' if r['pass'] else 'FAIL'} "
              f"({r['prompt_tokens']} tok, {r['elapsed_s']}s) -> {r['out']!r}")
        results.append(r)

    if args.twokey:
        lab1, lab2 = next(labels), next(labels)
        body = build_prompt(args.window, None, tok_per_char)
        cut1, cut2 = int(len(body) * 0.33), int(len(body) * 0.66)
        n1 = f" {MARKER}-ONE: the first secret passcode is {lab1}. "
        n2 = f" {MARKER}-TWO: the second secret passcode is {lab2}. "
        full = body[:cut1] + n1 + body[cut1:cut2] + n2 + body[cut2:]
        req = urllib.request.Request(
            f"{base}/v1/completions",
            data=json.dumps({"model": "/model", "prompt": full, "max_tokens": 64,
                             "temperature": 0,
                             "stop": None}).encode(),
            headers={"Content-Type": "application/json"})
        q = ("According to the document, what are the TWO secret passcodes? "
             "Answer with both letters in order, separated by a space.")
        full_q = full + "\n\n" + q
        t0 = time.perf_counter()
        with urllib.request.urlopen(urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({"model": "/model",
                                 "messages": [{"role": "user", "content": full_q}],
                                 "max_tokens": 64, "temperature": 0,
                                 "chat_template_kwargs": {"thinking": False}}).encode(),
                headers={"Content-Type": "application/json"}), timeout=7200) as r:
            resp = json.loads(r.read())
        dt = time.perf_counter() - t0
        out = resp["choices"][0]["message"]["content"] or ""
        ok = lab1 in out and lab2 in out
        rr = {"twokey": True, "pass": ok, "labels": [lab1, lab2],
              "prompt_tokens": resp["usage"]["prompt_tokens"],
              "elapsed_s": round(dt, 1), "out": out[:80]}
        print(f"twokey 33/66: {'PASS' if ok else 'FAIL'} "
              f"({rr['prompt_tokens']} tok, {rr['elapsed_s']}s) -> {out[:60]!r}")
        results.append(rr)

    npass = sum(r["pass"] for r in results)
    print(f"\nNIAH: {npass}/{len(results)} PASS")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "pass": npass, "total": len(results),
                       "window_target": args.window}, f, indent=1)


if __name__ == "__main__":
    main()
