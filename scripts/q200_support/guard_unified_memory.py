#!/usr/bin/env python3
"""Fail-closed GB10 unified-memory guard for one TP=2 campaign rank.

There are two independent thresholds:

* the hard safety threshold (8 GiB for the configured low-memory window and
  immediate kernel/cgroup/PSI/swap/rank-death signals), which aborts the owned
  rank and peer; and
* the pre-request admission floor, calibrated from measured negative swings,
  which prevents the next heavy row but never interrupts an active prefill.

All persisted rows carry rank and epoch identity so samples from two campaigns
cannot be combined accidentally.
"""
from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import re
import signal
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

try:
    from .admission_control import (
        CANDIDATE_RE,
        GUARD_SCHEMA,
        REQUEST_SCHEMA,
        SOURCE_RE,
        AdmissionError,
        atomic_write_json,
        load_config,
        validate_request_state,
    )
    from .niah_common import strict_json_loads
except ImportError:  # direct script execution
    from admission_control import (
        CANDIDATE_RE,
        GUARD_SCHEMA,
        REQUEST_SCHEMA,
        SOURCE_RE,
        AdmissionError,
        atomic_write_json,
        load_config,
        validate_request_state,
    )
    from niah_common import strict_json_loads

GIB = 1 << 30
HARD_MEMORY_FLOOR = 8 * GIB
DEFAULT_ADMISSION_FLOOR = 16 * GIB
DEFAULT_RESERVE = 8 * GIB
DEFAULT_LOW_MEMORY_SAMPLES = 15
READY_SCHEMA = "r0b0tlab.qwen38.full_context.ready.v1"
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
PROFILE_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MEMINFO_KEYS = frozenset(
    {
        "MemAvailable",
        "MemFree",
        "Cached",
        "AnonPages",
        "Unevictable",
        "SwapTotal",
        "SwapFree",
    }
)
REQUIRED_MEMORY_EVENT_KEYS = frozenset({"oom", "oom_kill"})


class GuardError(RuntimeError):
    """A missing or mismatched ownership/safety input; never continue blind."""


def ownership_nonce(
    *,
    candidate_id: str,
    candidate_source_sha: str,
    epoch: str,
    rank: str | int,
    profile_sha256: str,
    image_id: str,
) -> str:
    if not isinstance(candidate_id, str) or not CANDIDATE_RE.fullmatch(candidate_id):
        raise GuardError("candidate_id must be a safe immutable identity")
    if not isinstance(candidate_source_sha, str) or not SOURCE_RE.fullmatch(candidate_source_sha):
        raise GuardError("candidate_source_sha must be a full lowercase Git SHA")
    if (
        not isinstance(epoch, str)
        or not epoch
        or epoch == "unknown"
        or "\x00" in epoch
        or "\n" in epoch
        or "\r" in epoch
    ):
        raise GuardError("epoch must be a non-empty known single-line string")
    rank_text = str(rank)
    if rank_text not in {"0", "1", "2", "3"}:
        raise GuardError("rank must be 0, 1, 2 or 3")
    if not PROFILE_RE.fullmatch(profile_sha256):
        raise GuardError("profile SHA-256 must be full lowercase hex")
    if not IMAGE_ID_RE.fullmatch(image_id):
        raise GuardError("image ID must be sha256:<64 lowercase hex>")
    import hashlib
    return hashlib.sha256(
        f"{candidate_id}\0{candidate_source_sha}\0{epoch}\0{rank_text}\0{profile_sha256}\0{image_id}".encode("utf-8")
    ).hexdigest()


def validate_ownership_arguments(
    *,
    candidate_id: str,
    candidate_source_sha: str,
    epoch: str,
    rank: str | int,
    profile_sha256: str,
    container_id: str,
    image_id: str,
    owner_nonce: str,
    peer_rank: str | int,
    peer_candidate_id: str,
    peer_candidate_source_sha: str,
    peer_profile_sha256: str,
    peer_container_id: str,
    peer_image_id: str,
    peer_owner_nonce: str,
) -> None:
    """Validate the complete immutable local/peer ownership tuple.

    This must run before creating evidence paths or inspecting/stopping any
    container so malformed or cross-campaign inputs fail without side effects.
    """
    rank_text = str(rank)
    peer_rank_text = str(peer_rank)
    if rank_text not in {"0", "1", "2", "3"} or peer_rank_text not in {"0", "1", "2", "3"}:
        raise GuardError("local and peer ranks must each be 0, 1, 2 or 3")
    if rank_text == peer_rank_text:
        raise GuardError("local and peer ranks must be distinct")
    if not CONTAINER_ID_RE.fullmatch(container_id):
        raise GuardError("container ID must be the full 64-character lowercase hex ID")
    if not CONTAINER_ID_RE.fullmatch(peer_container_id):
        raise GuardError("peer container ID must be the full 64-character lowercase hex ID")
    if profile_sha256 != peer_profile_sha256:
        raise GuardError("local and peer profile SHA-256 values must match")
    if candidate_id != peer_candidate_id:
        raise GuardError("local and peer candidate IDs must match")
    if candidate_source_sha != peer_candidate_source_sha:
        raise GuardError("local and peer candidate source SHAs must match")
    expected_local = ownership_nonce(
        candidate_id=candidate_id,
        candidate_source_sha=candidate_source_sha,
        epoch=epoch,
        rank=rank_text,
        profile_sha256=profile_sha256,
        image_id=image_id,
    )
    expected_peer = ownership_nonce(
        candidate_id=peer_candidate_id,
        candidate_source_sha=peer_candidate_source_sha,
        epoch=epoch,
        rank=peer_rank_text,
        profile_sha256=peer_profile_sha256,
        image_id=peer_image_id,
    )
    if not NONCE_RE.fullmatch(owner_nonce) or not hmac.compare_digest(
        owner_nonce, expected_local
    ):
        raise GuardError("local ownership nonce does not match candidate/source/epoch/rank/profile/image")
    if not NONCE_RE.fullmatch(peer_owner_nonce) or not hmac.compare_digest(
        peer_owner_nonce, expected_peer
    ):
        raise GuardError("peer ownership nonce does not match candidate/source/epoch/rank/profile/image")


def validate_ready_state(
    value: object,
    *,
    candidate_id: str,
    candidate_source_sha: str,
    epoch: str,
    rank: str | int,
    profile_sha256: str,
    image_id: str,
    container_id: str,
    container_name: str,
    owner_nonce: str,
) -> dict[str, Any]:
    """Validate the launcher's exact readiness record before arming PSI."""
    if not isinstance(value, Mapping):
        raise GuardError("ready-state must be a JSON object")
    expected = {
        "schema": READY_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_source_sha": candidate_source_sha,
        "epoch": epoch,
        "rank": int(rank),
        "profile_sha256": profile_sha256,
        "image_id": image_id,
        "container_id": container_id,
        "container_name": container_name,
        "owner_nonce": owner_nonce,
        "warmups_status": "PASS" if str(rank) == "0" else "PEER_PASS",
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise GuardError(f"ready-state {key} does not match owned launch")
    if not isinstance(value.get("warmup_report"), str) or not value["warmup_report"]:
        raise GuardError("ready-state warmup_report must be a non-empty path")
    return dict(value)


def parse_meminfo(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if len(fields) > 1 and fields[1].lower() in {"kb", "kib"}:
            value *= 1024
        out[key] = value
    missing = sorted(REQUIRED_MEMINFO_KEYS - out.keys())
    if missing:
        raise GuardError(f"required /proc/meminfo keys are missing: {missing}")
    return out


def parse_psi_full_avg10(text: str) -> float:
    for line in text.splitlines():
        if not line.startswith("full "):
            continue
        for item in line.split()[1:]:
            if item.startswith("avg10="):
                try:
                    value = float(item.split("=", 1)[1])
                except ValueError as exc:
                    raise GuardError("memory PSI full avg10 is malformed") from exc
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise GuardError("memory PSI full avg10 must be finite and within 0..100")
                return value
    raise GuardError("required memory PSI full avg10 is missing")


def parse_memory_events(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise GuardError(f"malformed memory.events line: {line!r}")
        if fields[0] in out:
            raise GuardError(f"duplicate memory.events key: {fields[0]}")
        try:
            value = int(fields[1])
        except ValueError as exc:
            raise GuardError(f"invalid memory.events value: {line!r}") from exc
        if value < 0:
            raise GuardError(f"negative memory.events value: {line!r}")
        out[fields[0]] = value
    missing = sorted(REQUIRED_MEMORY_EVENT_KEYS - out.keys())
    if missing:
        raise GuardError(f"owned cgroup memory.events keys are missing: {missing}")
    return out


def p99_negative_swing(samples: list[int]) -> int:
    """Return the p99 decrease from the best preceding value in a trace."""
    if len(samples) < 2:
        return 0
    drops = [max(0, samples[i - 1] - samples[i]) for i in range(1, len(samples))]
    drops.sort()
    return drops[min(len(drops) - 1, int(0.99 * len(drops)))]


def admission_floor_bytes(
    measured_p99_negative_swing: int,
    *,
    minimum: int = DEFAULT_ADMISSION_FLOOR,
    reserve: int = DEFAULT_RESERVE,
) -> int:
    if measured_p99_negative_swing < 0 or minimum < 0 or reserve < 0:
        raise ValueError("memory floor inputs must be non-negative")
    if minimum < DEFAULT_ADMISSION_FLOOR:
        raise ValueError("minimum admission floor must be at least 16 GiB")
    return max(minimum, measured_p99_negative_swing + reserve)


def admission_status(mem_available: int, admission_floor: int) -> str:
    if admission_floor < 0:
        raise ValueError("admission floor must be non-negative")
    return "ADMIT" if mem_available >= admission_floor else "NOT_ADMITTED"


def read_nvrm_count() -> int:
    """Read the mandatory kernel OOM counter or fail closed."""
    errors: list[str] = []
    commands = (
        ["sudo", "-n", "journalctl", "-k", "--no-pager", "-o", "cat"],
    )
    for command in commands:
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{command[0]}:{type(exc).__name__}:{exc}")
            continue
        # Unprivileged journalctl can return rc0 with an empty/partial journal.
        # This safety signal requires privileged kernel visibility on every rank.
        if proc.returncode == 0 and proc.stdout.strip() and proc.stdout.strip() != "-- No entries --":
            return proc.stdout.count("NV_ERR_NO_MEMORY")
        errors.append(f"{command[0]}:rc={proc.returncode}:{proc.stderr.strip()}")
    raise GuardError("kernel NV_ERR_NO_MEMORY telemetry is unavailable: " + " | ".join(errors))


def docker_inspect(name_or_id: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["docker", "inspect", name_or_id, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode != 0:
        raise GuardError(f"docker inspect failed: {proc.stderr.strip()}")
    try:
        value = json.loads(proc.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GuardError("docker inspect returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise GuardError("docker inspect result is not an object")
    return value


def validate_container_identity(
    inspected: Mapping[str, Any], *, expected_container_id: str,
    expected_image_id: str, owner_nonce: str,
    expected_labels: Mapping[str, str] | None = None,
    expected_name: str | None = None,
) -> dict[str, Any]:
    if not CONTAINER_ID_RE.fullmatch(expected_container_id):
        raise GuardError("expected container ID must be the full 64-character ID")
    if not IMAGE_ID_RE.fullmatch(expected_image_id):
        raise GuardError("expected image ID must be sha256:<64 hex>")
    if not owner_nonce or not NONCE_RE.fullmatch(owner_nonce):
        raise GuardError("ownership nonce must be a full lowercase SHA-256")
    actual_id = str(inspected.get("Id", ""))
    actual_image = str(inspected.get("Image", ""))
    if not CONTAINER_ID_RE.fullmatch(actual_id) or not IMAGE_ID_RE.fullmatch(actual_image):
        raise GuardError("inspected container identity is not full and immutable")
    if actual_id != expected_container_id:
        raise GuardError(f"container ID mismatch: {actual_id!r} != {expected_container_id!r}")
    if actual_image != expected_image_id:
        raise GuardError(f"container image ID mismatch: {actual_image!r} != {expected_image_id!r}")
    if expected_name is not None:
        actual_name = str(inspected.get("Name", ""))
        if actual_name not in {expected_name, "/" + expected_name}:
            raise GuardError(f"container name mismatch: {actual_name!r} != {expected_name!r}")
    config = inspected.get("Config")
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    if not isinstance(labels, Mapping):
        raise GuardError("owned container labels are missing")
    required = {
        "org.r0b0tlab.owner_nonce": owner_nonce,
        "org.r0b0tlab.image_id": expected_image_id,
    }
    if expected_labels:
        required.update(expected_labels)
    for key, expected in required.items():
        if str(labels.get(key, "")) != expected:
            raise GuardError(f"owned container label mismatch: {key}")
    state = inspected.get("State")
    if not isinstance(state, Mapping):
        raise GuardError("owned container state is missing")
    return dict(state)


def resolve_cgroup_v2(inspected: Mapping[str, Any], *, previous: Path | None = None) -> Path:
    state = inspected.get("State")
    pid = state.get("Pid") if isinstance(state, Mapping) else None
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise GuardError("owned container has no valid init PID for cgroup resolution")
    try:
        rows = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GuardError(f"cannot read owned container cgroup membership: {exc}") from exc
    unified = [row.split("::", 1)[1] for row in rows if row.startswith("0::") and "::" in row]
    if len(unified) != 1 or not unified[0].startswith("/") or ".." in Path(unified[0]).parts:
        raise GuardError("owned container does not have exactly one safe cgroup v2 path")
    path = Path("/sys/fs/cgroup") / unified[0].lstrip("/")
    events = path / "memory.events"
    if path.is_symlink() or not path.is_dir() or events.is_symlink() or not events.is_file():
        raise GuardError(f"exact owned cgroup memory.events is unavailable: {events}")
    resolved = events.resolve(strict=True)
    cgroup_root = Path("/sys/fs/cgroup").resolve(strict=True)
    if cgroup_root != resolved and cgroup_root not in resolved.parents:
        raise GuardError("resolved memory.events escaped the cgroup v2 root")
    if previous is not None and resolved != previous:
        raise GuardError("owned container cgroup changed during guard lifetime")
    return resolved


def docker_state(name: str) -> dict[str, object] | None:
    try:
        value = docker_inspect(name)
    except (GuardError, OSError, subprocess.SubprocessError):
        return None
    state = value.get("State")
    return dict(state) if isinstance(state, Mapping) else None


def stop_exact(
    name: str, *, container_id: str, image_id: str, owner_nonce: str,
    expected_labels: Mapping[str, str] | None = None,
) -> None:
    inspected = docker_inspect(name)
    validate_container_identity(inspected, expected_container_id=container_id, expected_image_id=image_id, owner_nonce=owner_nonce, expected_labels=expected_labels, expected_name=name)
    proc = subprocess.run(["docker", "stop", "--time", "5", container_id], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=15, check=False)
    if proc.returncode:
        raise GuardError(f"owned container stop failed: {proc.stderr.strip()}")


def stop_peer(
    host: str | None, name: str | None, *, container_id: str | None,
    image_id: str | None, owner_nonce: str | None,
    ssh_identity_file: Path,
    expected_labels: Mapping[str, str] | None = None,
) -> None:
    if not host or not name or not container_id or not image_id or not owner_nonce:
        raise GuardError("guarded peer stop requires host, name, exact IDs, and nonce")
    ssh = ["ssh", "-i", str(ssh_identity_file), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + os.environ.get("DSV41_SSH_KNOWN_HOSTS", str(Path.home() / ".ssh/known_hosts_crs812_fabric"))]
    inspect = subprocess.run([*ssh, host, "docker", "inspect", container_id, "--format", "'{{json .}}'"], capture_output=True, text=True, timeout=15, check=False)
    if inspect.returncode:
        raise GuardError(f"peer inspect failed: {inspect.stderr.strip()}")
    try:
        value = json.loads(inspect.stdout)
    except json.JSONDecodeError as exc:
        raise GuardError("peer inspect returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise GuardError("peer inspect result is not an object")
    validate_container_identity(value, expected_container_id=container_id, expected_image_id=image_id, owner_nonce=owner_nonce, expected_labels=expected_labels, expected_name=name)
    stopped = subprocess.run([*ssh, host, "docker", "stop", "--time", "5", container_id], capture_output=True, text=True, timeout=15, check=False)
    if stopped.returncode:
        raise GuardError(f"peer stop failed: {stopped.stderr.strip()}")


def evaluate_abort(
    *,
    mem_available: int,
    low_samples: int,
    psi_full_avg10: float,
    swap_growth: int,
    events: dict[str, int],
    baseline_events: dict[str, int],
    nvrm_count: int | None,
    baseline_nvrm: int | None,
    state: dict[str, object] | None,
    seen_running: bool,
    low_memory_samples: int = 5,
    psi_hard_enabled: bool = True,
) -> list[str]:
    """Evaluate only immediate hard-abort signals (legacy API preserved)."""
    reasons: list[str] = []
    if mem_available < HARD_MEMORY_FLOOR and low_samples >= low_memory_samples:
        reasons.append(f"mem_available_below_8g_for_{low_memory_samples}s")
    if psi_hard_enabled and psi_full_avg10 > 5.0:
        reasons.append("memory_psi_full_avg10_gt_5")
    if swap_growth > 256 * (1 << 20):
        reasons.append("swap_growth_gt_256m_10s")
    for key in ("oom", "oom_kill"):
        if events.get(key, 0) > baseline_events.get(key, 0):
            reasons.append(f"memory_events_{key}")
    if nvrm_count is not None and baseline_nvrm is not None and nvrm_count > baseline_nvrm:
        reasons.append("new_nvrm_no_memory")
    if state is not None and bool(state.get("OOMKilled")):
        reasons.append("container_oomkilled")
    if seen_running and (state is None or not bool(state.get("Running"))):
        reasons.append("owned_container_stopped")
    return reasons


def evaluate_guard(
    *,
    mem_available: int,
    admission_floor: int,
    low_samples: int,
    psi_full_avg10: float = 0.0,
    swap_growth: int = 0,
    events: Mapping[str, int] | None = None,
    baseline_events: Mapping[str, int] | None = None,
    nvrm_count: int | None = 0,
    baseline_nvrm: int | None = 0,
    state: dict[str, object] | None = None,
    seen_running: bool = True,
    low_memory_samples: int = 15,
    rank: str | int = "unknown",
    epoch: str = "unknown",
    request_active: bool = False,
    psi_hard_enabled: bool = True,
    allow_load_swap: bool = False,
    pressure_admission_only: bool = False,
) -> dict[str, Any]:
    """Return a two-threshold decision with explicit rank/epoch identity."""
    load_swap_tolerated = (allow_load_swap and not psi_hard_enabled and not request_active
                           and mem_available >= DEFAULT_ADMISSION_FLOOR)
    hard = evaluate_abort(
        mem_available=mem_available,
        low_samples=low_samples,
        psi_full_avg10=psi_full_avg10,
        swap_growth=0 if load_swap_tolerated else swap_growth,
        events=dict(events or {}),
        baseline_events=dict(baseline_events or {}),
        nvrm_count=nvrm_count,
        baseline_nvrm=baseline_nvrm,
        state=state,
        seen_running=seen_running,
        low_memory_samples=low_memory_samples,
        psi_hard_enabled=psi_hard_enabled and not pressure_admission_only,
    )
    if hard:
        status = "HARD_ABORT"
        action = "STOP_LOCAL_AND_PEER"
    elif mem_available < admission_floor or (pressure_admission_only and psi_hard_enabled and psi_full_avg10 > 5.0):
        # An active request is never interrupted by this policy.  Both names
        # remain in the record to make the operator decision unambiguous.
        status = "WARNING" if request_active else "STOP_AFTER_CURRENT"
        action = "DO_NOT_ADMIT_NEXT"
    else:
        status = "GREEN"
        action = "ADMIT"
    return {
        "status": status,
        "action": action,
        "peer_stop": bool(hard),
        "rank": str(rank),
        "epoch": epoch,
        "mem_available": mem_available,
        "admission_floor": admission_floor,
        "admission": "NOT_ADMITTED" if action == "DO_NOT_ADMIT_NEXT" else admission_status(mem_available, admission_floor),
        "pressure_admission_only": pressure_admission_only,
        "hard_reasons": hard,
        "load_swap_tolerated": load_swap_tolerated,
        "request_active": request_active,
        "psi_hard_enabled": psi_hard_enabled,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-source-sha", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--owner-nonce", required=True)
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--peer-ssh-identity-file", type=Path, required=True)
    parser.add_argument("--peer-container", required=True)
    parser.add_argument("--peer-candidate-id", required=True)
    parser.add_argument("--peer-candidate-source-sha", required=True)
    parser.add_argument("--peer-container-id", required=True)
    parser.add_argument("--peer-image-id", required=True)
    parser.add_argument("--peer-owner-nonce", required=True)
    parser.add_argument("--peer-rank", required=True)
    parser.add_argument("--peer-profile-sha256", required=True)
    parser.add_argument("--rank", default=os.environ.get("RANK", "unknown"))
    parser.add_argument("--epoch", default=os.environ.get("EPOCH", "unknown"))
    parser.add_argument("--profile-sha256", default=os.environ.get("PROFILE_SHA256", ""))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-state", type=Path, required=True)
    parser.add_argument("--ready-state", type=Path, required=True)
    parser.add_argument("--admission-config", type=Path)
    parser.add_argument("--pressure-admission-only", action="store_true", help="PSI>5 pauses new requests; 8GiB/NVRM/cgroup-OOM remain hard aborts")
    parser.add_argument("--allow-load-swap", action="store_true", help="Allow swap-only load activity with >=16GiB reserve before first active request; NVRM/OOM gates remain hard")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--low-memory-samples", type=int, default=DEFAULT_LOW_MEMORY_SAMPLES
    )
    parser.add_argument("--admission-floor", type=int)
    parser.add_argument("--admission-floor-gib", type=float)
    parser.add_argument("--warning-floor-gib", type=float)
    args = parser.parse_args()
    if args.low_memory_samples < 1:
        parser.error("--low-memory-samples must be >= 1")
    if args.interval <= 0:
        parser.error("--interval must be > 0")
    if args.admission_floor is not None and args.admission_floor_gib is not None:
        parser.error("choose --admission-floor or --admission-floor-gib")
    if args.admission_floor is not None:
        admission_floor = args.admission_floor
    elif args.admission_floor_gib is not None:
        admission_floor = int(args.admission_floor_gib * GIB)
    else:
        admission_floor = DEFAULT_ADMISSION_FLOOR
    if admission_floor < DEFAULT_ADMISSION_FLOOR:
        parser.error("admission floor must be at least 16 GiB")
    if args.warning_floor_gib is None:
        warning_floor = admission_floor + 4 * GIB
    else:
        if not math.isfinite(args.warning_floor_gib) or args.warning_floor_gib < 0:
            parser.error("warning floor must be a finite non-negative number")
        warning_floor = int(args.warning_floor_gib * GIB)
    if warning_floor < admission_floor:
        parser.error("warning floor must be greater than or equal to admission floor")
    if not args.peer_ssh_identity_file.is_absolute() or not args.peer_ssh_identity_file.is_file():
        parser.error("peer SSH identity file must be an absolute readable file")

    try:
        validate_ownership_arguments(
            candidate_id=args.candidate_id,
            candidate_source_sha=args.candidate_source_sha,
            epoch=args.epoch,
            rank=args.rank,
            profile_sha256=args.profile_sha256,
            container_id=args.container_id,
            image_id=args.image_id,
            owner_nonce=args.owner_nonce,
            peer_rank=args.peer_rank,
            peer_candidate_id=args.peer_candidate_id,
            peer_candidate_source_sha=args.peer_candidate_source_sha,
            peer_profile_sha256=args.peer_profile_sha256,
            peer_container_id=args.peer_container_id,
            peer_image_id=args.peer_image_id,
            peer_owner_nonce=args.peer_owner_nonce,
        )
    except GuardError as exc:
        parser.error(str(exc))

    expected_ready_state = args.output.parent / (
        f"ready-{args.candidate_id}-{args.epoch}-rank{args.rank}.json"
    )
    if not args.ready_state.is_absolute() or args.ready_state != expected_ready_state:
        parser.error("ready-state must be the exact owned rank readiness path")

    if args.admission_config is not None:
        try:
            config = load_config(args.admission_config)
            endpoint = config.ranks[int(args.rank)]
            expected_state = args.output / "ADMISSION-STATE.json"
            if (
                config.epoch != args.epoch
                or config.candidate_id != args.candidate_id
                or config.candidate_source_sha != args.candidate_source_sha
                or config.profile_sha256 != args.profile_sha256
                or config.image_id != args.image_id
                or endpoint.request_state_path != args.request_state
                or endpoint.admission_state_path != expected_state
            ):
                raise GuardError("admission config does not bind this rank's exact request/state paths")
        except (AdmissionError, GuardError, OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid admission config handoff: {exc}")

    args.output.mkdir(parents=True, exist_ok=True)
    samples_path = args.output / "memory-guard.jsonl"
    abort_path = args.output / "MEMORY-GUARD-ABORT.json"
    admission_path = args.output / "ADMISSION-STATE.json"
    pid_path = args.output / "memory-guard.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    initial_request = {
        "schema": REQUEST_SCHEMA,
        "epoch": args.epoch,
        "candidate_id": args.candidate_id,
        "candidate_source_sha": args.candidate_source_sha,
        "profile_sha256": args.profile_sha256,
        "image_id": args.image_id,
        "active": False,
        "lease_id": None,
        "row_id": None,
        "updated_at": time.time(),
    }
    try:
        if args.request_state.exists():
            existing_request = validate_request_state(
                strict_json_loads(args.request_state.read_bytes()),
                epoch=args.epoch,
                candidate_id=args.candidate_id,
                candidate_source_sha=args.candidate_source_sha,
                profile_sha256=args.profile_sha256,
                image_id=args.image_id,
            )
            if existing_request["active"]:
                raise GuardError("request-state is active at guard startup")
        else:
            atomic_write_json(args.request_state, initial_request)
    except (AdmissionError, GuardError, OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid request-state handoff: {exc}") from exc

    running = True

    def stop_requested(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)

    labels = {
        "org.r0b0tlab.candidate_id": args.candidate_id,
        "org.r0b0tlab.candidate_source_sha": args.candidate_source_sha,
        "org.r0b0tlab.profile_sha256": args.profile_sha256,
        "org.r0b0tlab.epoch": args.epoch,
        "org.r0b0tlab.rank": str(args.rank),
    }

    def guarded_stop_local() -> None:
        stop_exact(args.container, container_id=args.container_id, image_id=args.image_id, owner_nonce=args.owner_nonce, expected_labels=labels)

    try:
        inspected = docker_inspect(args.container)
        state = validate_container_identity(inspected, expected_container_id=args.container_id, expected_image_id=args.image_id, owner_nonce=args.owner_nonce, expected_labels=labels)
        if not bool(state.get("Running")):
            raise GuardError("owned container is not running at guard handoff")
        events_path = resolve_cgroup_v2(inspected)
        baseline_events = parse_memory_events(events_path.read_text(encoding="utf-8"))
        baseline_nvrm = read_nvrm_count()
        swap_history: deque[int] = deque(maxlen=10)
        low_samples = 0
        seen_running = True
        sequence = 0
        psi_hard_armed = False
        request_seen_active = False

        with samples_path.open("a", buffering=1, encoding="utf-8") as stream:
            while running:
                request = validate_request_state(
                    strict_json_loads(args.request_state.read_bytes()),
                    epoch=args.epoch,
                    candidate_id=args.candidate_id,
                    candidate_source_sha=args.candidate_source_sha,
                    profile_sha256=args.profile_sha256,
                    image_id=args.image_id,
                )
                inspected = docker_inspect(args.container)
                state = validate_container_identity(inspected, expected_container_id=args.container_id, expected_image_id=args.image_id, owner_nonce=args.owner_nonce, expected_labels=labels)
                resolve_cgroup_v2(inspected, previous=events_path)
                mem = parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
                available = mem["MemAvailable"]
                swap_used = max(0, mem["SwapTotal"] - mem["SwapFree"])
                swap_history.append(swap_used)
                swap_growth = max(0, swap_history[-1] - swap_history[0]) if len(swap_history) == swap_history.maxlen else 0
                psi_path = Path("/proc/pressure/memory")
                psi = parse_psi_full_avg10(psi_path.read_text(encoding="utf-8"))
                events = parse_memory_events(events_path.read_text(encoding="utf-8"))
                nvrm = read_nvrm_count()
                low_samples = low_samples + 1 if available < HARD_MEMORY_FLOOR else 0
                if psi_hard_armed or args.ready_state.exists():
                    if args.ready_state.is_symlink():
                        raise GuardError("ready-state must not be a symlink")
                    validate_ready_state(
                        strict_json_loads(args.ready_state.read_bytes()),
                        candidate_id=args.candidate_id,
                        candidate_source_sha=args.candidate_source_sha,
                        epoch=args.epoch,
                        rank=args.rank,
                        profile_sha256=args.profile_sha256,
                        image_id=args.image_id,
                        container_id=args.container_id,
                        container_name=args.container,
                        owner_nonce=args.owner_nonce,
                    )
                    psi_hard_armed = True
                # Weight loading can legitimately create reclaim PSI while ample
                # memory remains.  Suppress only that one signal until readiness;
                # any active request arms it immediately.  OOM, NVRM, swap, low
                # memory, and container-death signals remain hard at all times.
                request_seen_active = request_seen_active or request["active"]
                psi_hard_enabled = psi_hard_armed or request_seen_active
                decision = evaluate_guard(
                    mem_available=available,
                    admission_floor=admission_floor,
                    low_samples=low_samples,
                    psi_full_avg10=psi,
                    swap_growth=swap_growth,
                    events=events,
                    baseline_events=baseline_events,
                    nvrm_count=nvrm,
                    baseline_nvrm=baseline_nvrm,
                    state=state,
                    seen_running=seen_running,
                    low_memory_samples=args.low_memory_samples,
                    rank=args.rank,
                    epoch=args.epoch,
                    request_active=request["active"],
                    psi_hard_enabled=psi_hard_enabled,
                    allow_load_swap=args.allow_load_swap,
                    pressure_admission_only=args.pressure_admission_only,
                )
                if decision["status"] == "GREEN" and available < warning_floor:
                    decision["status"] = "WARNING"
                    decision["action"] = "ADMIT_WITH_WARNING"
                sequence += 1
                row = {
                    "schema": GUARD_SCHEMA,
                    "sequence": sequence,
                    "ts": time.time(),
                    **decision,
                    "candidate_id": args.candidate_id,
                    "candidate_source_sha": args.candidate_source_sha,
                    "profile_sha256": args.profile_sha256,
                    "container_id": args.container_id,
                    "image_id": args.image_id,
                    "owner_nonce": args.owner_nonce,
                    "request_lease_id": request["lease_id"],
                    "request_row_id": request["row_id"],
                    "cgroup_memory_events": str(events_path),
                    "mem_free": mem["MemFree"],
                    "cached": mem["Cached"],
                    "anon_pages": mem["AnonPages"],
                    "unevictable": mem["Unevictable"],
                    "swap_used": swap_used,
                    "swap_growth_10s": swap_growth,
                    "psi_full_avg10": psi,
                    "psi_hard_armed": psi_hard_armed,
                    "memory_events": events,
                    "nvrm_no_memory_count": nvrm,
                    "container_state": state,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                atomic_write_json(admission_path, row)
                if decision["status"] == "HARD_ABORT":
                    atomic_write_json(abort_path, row)
                    guarded_stop_local()
                    stop_peer(args.peer_host, args.peer_container, container_id=args.peer_container_id, image_id=args.peer_image_id, owner_nonce=args.peer_owner_nonce, ssh_identity_file=args.peer_ssh_identity_file, expected_labels={"org.r0b0tlab.candidate_id": args.peer_candidate_id, "org.r0b0tlab.candidate_source_sha": args.peer_candidate_source_sha, "org.r0b0tlab.epoch": args.epoch, "org.r0b0tlab.profile_sha256": args.peer_profile_sha256, "org.r0b0tlab.rank": str(args.peer_rank)})
                    return 2
                time.sleep(args.interval)
    except (AdmissionError, GuardError, OSError, subprocess.SubprocessError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        stop_errors: list[str] = []
        try:
            guarded_stop_local()
        except (GuardError, OSError, subprocess.SubprocessError) as stop_exc:
            stop_errors.append(f"local:{type(stop_exc).__name__}:{stop_exc}")
        try:
            stop_peer(args.peer_host, args.peer_container, container_id=args.peer_container_id, image_id=args.peer_image_id, owner_nonce=args.peer_owner_nonce, ssh_identity_file=args.peer_ssh_identity_file, expected_labels={"org.r0b0tlab.candidate_id": args.peer_candidate_id, "org.r0b0tlab.candidate_source_sha": args.peer_candidate_source_sha, "org.r0b0tlab.epoch": args.epoch, "org.r0b0tlab.profile_sha256": args.peer_profile_sha256, "org.r0b0tlab.rank": str(args.peer_rank)})
        except (GuardError, OSError, subprocess.SubprocessError) as stop_exc:
            stop_errors.append(f"peer:{type(stop_exc).__name__}:{stop_exc}")
        atomic_write_json(args.output / "MEMORY-GUARD-ERROR.json", {"error": str(exc), "candidate_id": args.candidate_id, "candidate_source_sha": args.candidate_source_sha, "container_id": args.container_id, "image_id": args.image_id, "owner_nonce": args.owner_nonce, "stop_errors": stop_errors})
        return 3
    finally:
        (args.output / "MEMORY-GUARD-STOPPED").write_text(f"{time.time()}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
