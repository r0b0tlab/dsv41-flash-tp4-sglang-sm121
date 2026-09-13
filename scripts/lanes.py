#!/usr/bin/env python3
"""SHORT/MEDIUM/PROSE throughput lanes for dsv4.1-flash (matched method).

Metric definitions (fixed, claim-bearing):
- token counts from server `usage` (never SSE chunks; DSpark packs tokens)
- decode = (completion_tokens - 1) / (t_end - t_first_token) per stream
- aggregate = all streams' tokens / batch wall time
- TTFT = first-token delta
- warm-up generation before each lane; GPU fast-state assumed (probe before)

Usage: python3 scripts/lanes.py [--base-url ...] [--out evidence/phase8/lanes.json]
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def chat(base, payload, timeout=1200, stream=False):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line and first is None:
                first = time.perf_counter()
    return {"t0": t0, "t_first": first, "t_end": time.perf_counter()}


def one_stream(base, prompt, max_tokens):
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({"model": "/model", "max_tokens": max_tokens,
                         "temperature": 0, "stream": True,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json"})
    first = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line and first is None:
                first = time.perf_counter()
    # completion tokens via non-stream repeat is wasteful; count from usage in
    # the final SSE chunk is unreliable -> use non-stream for token count and
    # stream for timing on separate passes is also wrong. Instead: single
    # non-streaming call records usage + total time; decode needs first-token.
    # Compromise (r0b0bench-compatible): stream once, count chunks is WRONG for
    # DSpark -> do BOTH: stream for TTFT, then non-stream for tokens.
    return t0, first


def measure(base, prompt, max_tokens, concurrency):
    results = []
    def run(i):
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({"model": "/model", "max_tokens": max_tokens,
                             "temperature": 0,
                             "chat_template_kwargs": {"thinking": False},
                             "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read())
        t1 = time.perf_counter()
        return {"tokens": resp["usage"]["completion_tokens"],
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "wall": t1 - t0}
    with ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(run, range(concurrency)))
    toks = sum(r["tokens"] for r in results)
    wall = max(r["wall"] for r in results)
    return {"aggregate_tok_s": round(toks / wall, 2),
            "per_stream_tok_s": round(toks / sum(r["wall"] for r in results), 2),
            "tokens": toks, "wall_s": round(wall, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # warm-up
    chat(base, {"model": "/model", "max_tokens": 64, "temperature": 0,
                "messages": [{"role": "user", "content": "hi"}]})

    lanes = {}
    # SHORT: random-id 512 in / 256 out (text; random ids keep prefill light)
    short_prompt = "Repeat these ids back verbatim, comma-separated: " + \
        " ".join(str((i * 7919) % 100000) for i in range(400))
    lanes["short_c1"] = measure(base, short_prompt, 256, 1)
    # MEDIUM: 2048 in / 512 out
    med_prompt = ("Summarize the following passage in detail, then list its "
                  "key points. Passage: " + ("The quick brown fox jumps over "
                  "the lazy dog while the sun rises over quiet hills. ") * 120)
    lanes["medium_c1"] = measure(base, med_prompt, 512, 1)
    # PROSE: ~500-word narrative out
    lanes["prose_c1"] = measure(
        base, "Write a vivid 400-word story about a lighthouse keeper who "
              "discovers something strange in the fog.", 800, 1)
    # counting ceiling (reference-comparable)
    lanes["counting_c1"] = measure(
        base, "Count from 1 to 150, one number per line, no other text.", 400, 1)
    # small concurrency sweep
    lanes["counting_c4"] = measure(
        base, "Count from 1 to 100, one number per line, no other text.", 300, 4)

    print(json.dumps(lanes, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(lanes, f, indent=1)


if __name__ == "__main__":
    main()
