#!/usr/bin/env python3
"""Concurrency ladder c1..c8 for phase 9 (serial orchestrator lane).

Proves true concurrency via /get_server_info sampling (running vs queued)
while stepping client concurrency 1,2,4,8. One lane at a time; no other
benchmarks may be in flight (2026-09-13 freeze root cause).
"""

import argparse
import json
import threading
import time
import urllib.request

PROMPT = ("Count from 1 to {n}, one number per line, no other text.")


def server_running(base):
    try:
        with urllib.request.urlopen(f"{base}/get_server_info", timeout=10) as r:
            d = json.loads(r.read())
        return int(d.get("num_running_reqs", d.get("running_requests", 0)) or 0)
    except Exception:
        return -1


def chat(base, n, timeout=900):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({
            "model": "/model", "max_tokens": 300, "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": PROMPT.format(n=100)}],
        }).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return (resp["usage"]["completion_tokens"], time.perf_counter() - t0)


def ladder_step(base, conc, dur_s=60):
    stop = time.perf_counter() + dur_s
    peak_running = 0
    samples = []
    done = []
    lock = threading.Lock()

    def worker():
        while time.perf_counter() < stop:
            try:
                ct, dt = chat(base, 100)
                with lock:
                    done.append((ct, dt))
            except Exception as e:
                with lock:
                    done.append((None, str(e)))
                time.sleep(1)

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        run = server_running(base)
        if run >= 0:
            samples.append(run)
            peak_running = max(peak_running, run)
        time.sleep(1)
    for t in threads:
        t.join()
    ok = [d for d in done if d[0] is not None]
    toks = sum(d[0] for d in ok)
    wall = dur_s
    return {
        "concurrency": conc,
        "peak_server_running": peak_running,
        "server_running_samples": samples,
        "completed": len(ok),
        "errors": len(done) - len(ok),
        "aggregate_tok_s": round(toks / wall, 1) if ok else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", default="1,2,4,8")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    results = []
    for c in [int(x) for x in args.steps.split(",")]:
        print(f"c{c} ...", flush=True)
        r = ladder_step(base, c)
        results.append(r)
        print(f"c{c}: peak_running={r['peak_server_running']} "
              f"completed={r['completed']} agg={r['aggregate_tok_s']} tok/s")
    json_out = {"steps": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(json_out, f, indent=1)
    print(json.dumps(json_out))


if __name__ == "__main__":
    main()
