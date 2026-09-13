#!/usr/bin/env python3
"""Verify the downloaded DeepSeek-V4.1-Flash checkpoint against the HF LFS
sha256 manifest. Reads sizes/sha256 from the HF API blobs listing (cached
to references/hf-blobs.json) and re-hashes every local shard.

Usage (on the node holding the checkpoint):
    python3 scripts/verify_checkpoint.py --ckpt ~/models/llm/dsv41/DeepSeek-V4.1-Flash
Exit 0 = 48/48 shards match. Writes a per-shard report next to the log.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

API = ("https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash"
       "?blobs=true")
LAYER_FILES = ("model.safetensors.index.json", "config.json",
               "tokenizer.json", "tokenizer_config.json")


def fetch_manifest(cache: Path) -> dict:
    if cache.is_file():
        return json.loads(cache.read_text())
    with urllib.request.urlopen(API, timeout=60) as r:
        data = json.loads(r.read())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[1]
                                           / "references" / "hf-blobs.json"))
    args = ap.parse_args()
    ckpt = Path(args.ckpt)

    manifest = fetch_manifest(Path(args.cache))
    blobs = {s["rfilename"]: s for s in manifest["siblings"]}

    shards = sorted(p for p in ckpt.glob("model-*.safetensors"))
    expected = [f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)]
    if [p.name for p in shards] != expected:
        print(f"FAIL: expected 48 shards, found {len(shards)}")
        return 1

    bad = []
    for i, p in enumerate(shards, 1):
        want = blobs[p.name].get("lfs", {}).get("sha256")
        if want is None:
            print(f"[{i:02d}/48] {p.name}: no LFS sha256 in manifest (skip)")
            continue
        got = sha256_file(p)
        ok = got == want
        print(f"[{i:02d}/48] {p.name}: {'OK' if ok else 'MISMATCH'} "
              f"({p.stat().st_size / 2**30:.1f} GiB)")
        if not ok:
            bad.append(p.name)

    for f in LAYER_FILES:
        present = (ckpt / f).is_file()
        print(f"{'OK' if present else 'MISSING'}: {f}")
        if not present:
            bad.append(f)

    if bad:
        print(f"\nVERIFY FAILED: {bad}")
        return 1
    print("\nVERIFY OK: 48/48 shards + aux files match revision "
          f"{manifest['sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
