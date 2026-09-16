"""Exact-token NIAH construction and bounded HTTP helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
from typing import Any, Mapping


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def strict_json_loads(raw: bytes | str) -> Any:
    """Parse an API body without duplicate keys, NaN, or Infinity."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_constant)


_PROM_SAMPLE_RE = re.compile(
    r'^sglang:prefill_effective_tokens_total\{(?P<labels>.*)\}\s+(?P<value>[^\s]+)$'
)
_PROM_LABEL_RE = re.compile(r'(?:^|,)([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


def prefill_input_counter(metrics: str, model: str) -> int | None:
    """Read the one rank-0 input-token counter exposed by SGLang metrics."""
    if not isinstance(metrics, str) or not isinstance(model, str) or not model:
        return None
    matches: list[int] = []
    for line in metrics.splitlines():
        sample = _PROM_SAMPLE_RE.fullmatch(line.strip())
        if sample is None:
            continue
        labels = {
            key: bytes(value, "utf-8").decode("unicode_escape")
            for key, value in _PROM_LABEL_RE.findall(sample.group("labels"))
        }
        if labels.get("mode") != "input" or labels.get("model_name") != model:
            continue
        if labels.get("tp_rank") not in (None, "0") or labels.get("pp_rank") not in (None, "0"):
            continue
        try:
            value = float(sample.group("value"))
        except ValueError:
            return None
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            return None
        matches.append(int(value))
    if len(matches) != 1:
        return None
    return matches[0]


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("needle token sequence is empty")
    width = len(needle)
    for index in range(len(haystack) - width + 1):
        if haystack[index : index + width] == needle:
            return index
    raise ValueError("marker token sequence not found")


def repeated_to_length(unit: list[int], length: int) -> list[int]:
    if not unit:
        raise ValueError("filler token unit is empty")
    if length < 0:
        raise ValueError("requested filler length is negative")
    return (unit * ((length // len(unit)) + 1))[:length]


def single_needle_prompt(
    *,
    prefix: list[int],
    suffix: list[int],
    needle: list[int],
    filler_unit: list[int],
    target_tokens: int,
    fraction: float,
) -> tuple[list[int], int]:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in (*prefix, *suffix, *needle, *filler_unit)):
        raise ValueError("prompt pieces must contain integer token IDs")
    filler = target_tokens - len(prefix) - len(suffix) - len(needle)
    if filler < 0:
        raise ValueError("target too small for template and needle")
    left_len = int(filler * fraction)
    right_len = filler - left_len
    ids = prefix + repeated_to_length(filler_unit, left_len) + needle + repeated_to_length(filler_unit, right_len) + suffix
    if len(ids) != target_tokens:
        raise AssertionError((len(ids), target_tokens))
    return ids, len(prefix) + left_len


def multi_needle_prompt(
    *,
    prefix: list[int],
    suffix: list[int],
    needles: list[list[int]],
    filler_unit: list[int],
    target_tokens: int,
    fractions: list[float],
) -> tuple[list[int], list[int]]:
    if len(needles) != len(fractions) or not needles:
        raise ValueError("needles and fractions must have equal non-zero length")
    if any(not 0 <= fraction <= 1 for fraction in fractions):
        raise ValueError("fractions must be between 0 and 1")
    if list(fractions) != sorted(fractions):
        raise ValueError("fractions must be non-decreasing")
    if any(not needle or any(isinstance(token, bool) or not isinstance(token, int) for token in needle) for needle in needles):
        raise ValueError("needles must contain non-empty integer token sequences")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in (*prefix, *suffix, *filler_unit)):
        raise ValueError("prompt pieces must contain integer token IDs")
    if any(left == right for left, right in zip(fractions, fractions[1:])):
        raise ValueError("needle fractions must be strictly increasing")
    insertion = target_tokens - len(prefix) - len(suffix)
    needle_total = sum(len(needle) for needle in needles)
    filler_total = insertion - needle_total
    if filler_total < 0:
        raise ValueError("target too small")
    desired = [int(filler_total * fraction) for fraction in fractions]
    segments: list[int] = []
    offsets: list[int] = []
    consumed_filler = 0
    for index, needle in enumerate(needles):
        segment_len = max(0, desired[index] - consumed_filler)
        segments.extend(repeated_to_length(filler_unit, segment_len))
        consumed_filler += segment_len
        offsets.append(len(prefix) + len(segments))
        segments.extend(needle)
    remaining = filler_total - consumed_filler
    segments.extend(repeated_to_length(filler_unit, remaining))
    ids = prefix + segments + suffix
    if len(ids) != target_tokens:
        raise AssertionError((len(ids), target_tokens))
    return ids, offsets


def scoped_prompt_counter(snapshot: Any, scope: str) -> int | None:
    """Read one unambiguous scope-labelled server prompt counter.

    A global counter or an API usage field is not sufficient: NIAH proof must
    identify the counter stream that belongs to this run and compare a before
    and after snapshot of that stream.  Conflicting duplicate occurrences are
    rejected instead of silently selecting the first nested value.
    """
    if not isinstance(scope, str) or not scope:
        raise ValueError("counter scope must be non-empty")
    matches: list[int] = []

    def add(value: Any) -> None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            matches.append(value)

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in ("prompt_counters", "prompt_counter_by_scope"):
                counters = value.get(key)
                if isinstance(counters, Mapping):
                    add(counters.get(scope))
            counter = value.get("prompt_counter")
            if isinstance(counter, Mapping) and counter.get("scope") == scope:
                for key in ("tokens", "prompt_tokens", "total", "value"):
                    add(counter.get(key))
            if value.get("scope") == scope:
                for key in ("prompt_tokens_total", "prompt_tokens", "tokens", "total"):
                    add(value.get(key))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(snapshot)
    if not matches or len(set(matches)) != 1:
        return None
    return matches[0]


def validate_niah_token_proof(
    *, local_token_count: int, usage: Mapping[str, Any] | None, before: Any,
    after: Any, target_tokens: int, scope: str,
) -> dict[str, Any]:
    """Require image-tokenizer count == API usage == scoped counter delta."""
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    local_ok = isinstance(local_token_count, int) and not isinstance(local_token_count, bool) and local_token_count == target_tokens
    api_value = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    api_ok = isinstance(api_value, int) and not isinstance(api_value, bool) and api_value == target_tokens
    before_value = scoped_prompt_counter(before, scope)
    after_value = scoped_prompt_counter(after, scope)
    counter_delta = after_value - before_value if before_value is not None and after_value is not None else None
    counter_ok = counter_delta == target_tokens
    return {
        "scope": scope,
        "local_tokenizer_tokens": local_token_count,
        "api_prompt_tokens": api_value,
        "server_counter_before": before_value,
        "server_counter_after": after_value,
        "server_counter_delta": counter_delta,
        "target_tokens": target_tokens,
        "passed": bool(local_ok and api_ok and counter_ok and local_token_count == api_value == counter_delta),
    }


def validate_completion_result(
    result: Mapping[str, Any],
    *,
    expected_prompt_tokens: int,
    max_completion_tokens: int,
    allowed_finish_reasons: set[str],
    required_texts: tuple[str, ...] = (),
    expected_completion_tokens: int | None = None,
) -> tuple[bool, str | None]:
    """Validate the response contract shared by NIAH and capacity gates."""
    if result.get("status") != 200:
        return False, "http_status"
    if result.get("error") is not None:
        return False, "response_error"
    text = result.get("text")
    if not isinstance(text, str) or not text:
        return False, "empty_completion_text"
    if any(required not in text for required in required_texts):
        return False, "missing_required_text"
    if result.get("finish_reason") not in allowed_finish_reasons:
        return False, "invalid_finish_reason"
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        return False, "missing_usage"
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, bool) or prompt_tokens != expected_prompt_tokens:
        return False, "prompt_token_mismatch"
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 1
        or completion_tokens > max_completion_tokens
    ):
        return False, "invalid_completion_tokens"
    if expected_completion_tokens is not None and completion_tokens != expected_completion_tokens:
        return False, "completion_token_mismatch"
    return True, None


def request_json(base: str, path: str = "/server_info", timeout: int = 600) -> tuple[int, Any, float, str | None]:
    req = urllib.request.Request(base.rstrip("/") + path, method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, strict_json_loads(response.read()), time.perf_counter() - started, None
    except Exception as exc:
        return 0, {}, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def request_text(base: str, path: str, timeout: int = 600) -> tuple[int, str, float, str | None]:
    req = urllib.request.Request(base.rstrip("/") + path, method="GET")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8"), time.perf_counter() - started, None
    except Exception as exc:
        return 0, "", time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def post_completion(
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    timeout: int,
    *,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    if isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "skip_special_tokens": True,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    data = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = strict_json_loads(response.read())
            if not isinstance(body, dict):
                raise ValueError("completion response must be a JSON object")
            status = response.status
        error = None
    except Exception as exc:
        status = 0
        body = {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    choices = body.get("choices") if isinstance(body, dict) and isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    usage = body.get("usage") if isinstance(body, dict) and isinstance(body.get("usage"), dict) else {}
    return {
        "status": status,
        "error": error,
        "elapsed_seconds": elapsed,
        "text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": dict(usage),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt_ids, separators=(",", ":")).encode()).hexdigest(),
    }


def graph_delta(before: Any, after: Any) -> dict[str, Any]:
    """Extract graph replay/fallback deltas from a server_info snapshot."""
    def locate(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "graph_replay_stats" in obj:
                return obj["graph_replay_stats"]
            for value in obj.values():
                result = locate(value)
                if result is not None:
                    return result
        elif isinstance(obj, list):
            for value in obj:
                result = locate(value)
                if result is not None:
                    return result
        return None

    old = locate(before) or {}
    new = locate(after) or {}
    rows: dict[str, Any] = {}
    for owner, component in (("target", "target_verify"), ("draft", "draft_decode"), ("draft", "draft_extend")):
        old_row = ((old.get(owner) or {}).get(component) or {})
        new_row = ((new.get(owner) or {}).get(component) or {})
        old_reasons_value = old_row.get("fallback_reasons")
        new_reasons_value = new_row.get("fallback_reasons")
        old_reasons: Mapping[str, Any] = old_reasons_value if isinstance(old_reasons_value, Mapping) else {}
        new_reasons: Mapping[str, Any] = new_reasons_value if isinstance(new_reasons_value, Mapping) else {}
        reason_deltas: dict[str, int | None] = {}
        for reason in set(old_reasons) | set(new_reasons):
            old_count = old_reasons.get(reason, 0)
            new_count = new_reasons.get(reason, 0)
            if (
                isinstance(old_count, int)
                and not isinstance(old_count, bool)
                and isinstance(new_count, int)
                and not isinstance(new_count, bool)
            ):
                delta = new_count - old_count
                if delta:
                    reason_deltas[str(reason)] = delta
            else:
                reason_deltas[str(reason)] = None
        rows[f"{owner}.{component}"] = {
            "captures": new_row.get("captures"),
            "replay_delta": (new_row.get("replays", 0) - old_row.get("replays", 0)),
            "fallback_delta": (new_row.get("fallbacks", 0) - old_row.get("fallbacks", 0)),
            "fallback_reasons": reason_deltas,
        }
    return rows
