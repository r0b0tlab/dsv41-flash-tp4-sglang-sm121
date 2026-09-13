#!/usr/bin/env python3
"""Q200v2-lite for dsv4.1-flash: quality-text-180 subset + fixed grader.

Grader: numeric_exact -> last number in output (never first); exact -> exact
string after strip. Thinking off, temperature 0, generous completion budget.
Reports per-family and total accuracy.

Usage: python3 scripts/q200_lite.py [--n 60] [--out evidence/phase9/q200.json]
"""

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def grade_numeric_exact(out, ref):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", out)
    if not nums:
        return False
    def norm(s):
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    want = norm(str(ref))
    got = norm(nums[-1])
    return want is not None and got is not None and abs(want - got) < 1e-6


def grade_exact(out, ref):
    return out.strip() == str(ref).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--n", type=int, default=60, help="cases per run (sampled)")
    ap.add_argument("--dataset", default="/home/r0b0tdgx/qwen38-flash-next-w4a16/operator-impl/artifacts/quality-text-180-v2.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.dataset)]
    # deterministic sample across families
    import collections
    by_fam = collections.defaultdict(list)
    for c in cases:
        by_fam[c["family"]].append(c)
    fams = sorted(by_fam)
    per_fam = max(1, args.n // len(fams))
    sampled = []
    for f in fams:
        take = by_fam[f][:per_fam]
        sampled.extend(take)
    print(f"{len(sampled)} cases across {len(fams)} families: {fams}")

    def run(c):
        req = urllib.request.Request(
            f"{args.base_url}/v1/chat/completions",
            data=json.dumps({"model": "/model", "max_tokens": 2048,
                             "temperature": 0,
                             "chat_template_kwargs": {"thinking": False},
                             "messages": [{"role": "user", "content": c["prompt"]}]}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.loads(r.read())
            out = resp["choices"][0]["message"]["content"] or ""
        except Exception as e:
            return c, False, f"ERR {type(e).__name__}"
        ok = (grade_numeric_exact if c["grade"] == "numeric_exact" else grade_exact)(out, c["reference"])
        return c, ok, out

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(run, sampled))
    dt = time.time() - t0

    fam_stats = collections.defaultdict(lambda: [0, 0])
    for c, ok, _ in results:
        fam_stats[c["family"]][0] += ok
        fam_stats[c["family"]][1] += 1
    total_ok = sum(v[0] for v in fam_stats.values())
    total = sum(v[1] for v in fam_stats.values())
    print(f"\nQ200v2-lite: {total_ok}/{total} = {100*total_ok/total:.1f}%  ({dt:.0f}s)")
    for f in fams:
        o, t = fam_stats[f]
        print(f"  {f:14s} {o:3d}/{t:<3d}")
    fails = [(c["id"], out[:100]) for c, ok, out in results if not ok and not out.startswith("ERR")]
    print("\nsample failures:")
    for i, (cid, out) in enumerate(fails[:5]):
        print(f"  {cid}: {out!r}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"total": total, "ok": total_ok, "families": {k: v for k, v in fam_stats.items()},
                       "elapsed_s": dt}, f, indent=1)


if __name__ == "__main__":
    main()
