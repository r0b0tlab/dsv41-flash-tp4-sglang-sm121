#!/usr/bin/env python3
"""Single multineedle (two-key) case on the dsv4.1-flash 1M profile.

One request only: prompt sized to the advertised 1,048,576-token window,
needles at 33% and 66% depth, ordered recall demanded. Client timeout is
generous (chunked prefill at 2048 makes a 1M prefill take hours); log every
progress marker so the durable tmux owner can see liveness without polling
the serve.
"""

import argparse
import json
import time
import urllib.request

MARKER = "R0B0BENCH_NIAH_SLOT"
FILLER_SENT = ("The quarterly inventory review concluded without material "
               "discrepancies across all regional warehouses and depots. ")


def build_prompt(target_tokens, approx_tok_per_char=0.30):
    est_chars = int(target_tokens / approx_tok_per_char)
    body = FILLER_SENT * (est_chars // len(FILLER_SENT) + 1)
    return body[:est_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--window", type=int, default=1024 * 1024 - 2114,
                    help="target prompt tokens (leave room for needles+q+out)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    t_start = time.time()
    print(f"[niah1m] window target: {args.window} tokens", flush=True)

    # calibration: learn tok/char ratio for THIS serve/tokenizer first
    cal_body = FILLER_SENT * 20
    cal_req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({
            "model": "/model", "max_tokens": 8, "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": cal_body + " Reply OK."}]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(cal_req, timeout=300) as r:
        cal = json.loads(r.read())
    tok_per_char = cal["usage"]["prompt_tokens"] / (len(cal_body) + 60)
    print(f"[niah1m] calibration: {tok_per_char:.4f} tok/char", flush=True)

    body = build_prompt(args.window, tok_per_char)
    print(f"[niah1m] prompt built: {len(body)} chars "
          f"(~{int(len(body) * tok_per_char)} tokens)", flush=True)

    cut1, cut2 = int(len(body) * 0.33), int(len(body) * 0.66)
    n1 = f" {MARKER}-ONE: the first secret passcode is KM4DX-7F2Q. "
    n2 = f" {MARKER}-TWO: the second secret passcode is VH9LP-3W8T. "
    full = body[:cut1] + n1 + body[cut1:cut2] + n2 + body[cut2:]
    q = ("\n\nAccording to the document, what are the TWO secret passcodes? "
         "Answer with both codes in order, separated by a space.")

    # Node-safety gate: refuse to submit if head unified-memory headroom is thin
    # (2026-09-13 abort: NVRM NO_MEMORY at 5.1 GiB during prefill).
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                avail_mib = int(line.split()[1]) // 1024
                break
    if avail_mib < 14000:
        raise SystemExit(f"[niah1m] REFUSED: MemAvailable {avail_mib} MiB < 14000 MiB floor")
    print(f"[niah1m] memory gate OK ({avail_mib} MiB)", flush=True)

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({
            "model": "/model", "max_tokens": 128, "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": full + q}]}).encode(),
        headers={"Content-Type": "application/json"})

    print(f"[niah1m] submitting request ({time.strftime('%H:%M:%S')})", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=12 * 3600) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0

    out = resp["choices"][0]["message"]["content"] or ""
    ptok = resp["usage"]["prompt_tokens"]
    ok = "KM4DX-7F2Q" in out and "VH9LP-3W8T" in out
    result = {
        "twokey": True, "pass": ok, "depths": [0.33, 0.66],
        "prompt_tokens": ptok, "window_target": args.window,
        "elapsed_s": round(dt, 1), "out": out[:160],
        "total_wall_s": round(time.time() - t_start, 1),
    }
    print(f"[niah1m] {'PASS' if ok else 'FAIL'} ({ptok} tok, {dt:.0f}s) "
          f"-> {out[:120]!r}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"[niah1m] written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
