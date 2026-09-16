#!/usr/bin/env python3
"""Four-rank fail-closed admission leases for claim-bearing requests.

A controller acquires a local exclusive lease, verifies fresh state from both
memory guards, asks both guards to acknowledge the same active lease, and only
then permits one heavy request/batch.  Release is acknowledged before the lock
is removed, so a transport failure cannot silently admit a second row.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import secrets
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    from .niah_common import strict_json_loads
except ImportError:  # direct script execution
    from niah_common import strict_json_loads

PROFILE_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTAINER_RE = re.compile(r"^[0-9a-f]{64}$")
LEASE_RE = re.compile(r"^[0-9a-f]{64}$")
HOST_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
CANDIDATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SOURCE_RE = re.compile(r"^[0-9a-f]{40}$")
GUARD_SCHEMA = "r0b0tlab.qwen38.guard_admission.v1"
REQUEST_SCHEMA = "r0b0tlab.qwen38.request_lease.v1"
CONFIG_SCHEMA = "r0b0tlab.dsv41.admission_config.v1"


class AdmissionError(RuntimeError):
    """Admission evidence is missing, stale, malformed, or unacknowledged."""


class AdmissionNotGranted(AdmissionError):
    """Both guards are healthy, but at least one rank is below its floor."""


@dataclass(frozen=True)
class RankEndpoint:
    rank: str
    admission_state_path: Path
    request_state_path: Path
    host: str | None = None
    ssh_identity_file: Path | None = None


@dataclass(frozen=True)
class AdmissionConfig:
    epoch: str
    candidate_id: str
    candidate_source_sha: str
    profile_sha256: str
    image_id: str
    lease_path: Path
    max_state_age_seconds: float
    ack_timeout_seconds: float
    poll_interval_seconds: float
    ranks: tuple[RankEndpoint, ...]


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _safe_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise AdmissionError(f"{label} must be a non-empty single-line string")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise AdmissionError(f"{label} must be a string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise AdmissionError(f"{label} must be an absolute non-traversing path")
    return path


def load_config(path: str | Path) -> AdmissionConfig:
    raw = strict_json_loads(Path(path).read_bytes())
    if not isinstance(raw, Mapping) or raw.get("schema") != CONFIG_SCHEMA:
        raise AdmissionError("admission config schema mismatch")
    epoch = _safe_identity(raw.get("epoch"), "epoch")
    candidate = raw.get("candidate_id")
    source = raw.get("candidate_source_sha")
    profile = raw.get("profile_sha256")
    image = raw.get("image_id")
    if not isinstance(candidate, str) or not CANDIDATE_RE.fullmatch(candidate):
        raise AdmissionError("candidate_id must be a safe immutable identity")
    if not isinstance(source, str) or not SOURCE_RE.fullmatch(source):
        raise AdmissionError("candidate_source_sha must be a full lowercase Git SHA")
    if not isinstance(profile, str) or not PROFILE_RE.fullmatch(profile):
        raise AdmissionError("profile_sha256 must be full lowercase hex")
    if not isinstance(image, str) or not IMAGE_RE.fullmatch(image):
        raise AdmissionError("image_id must be immutable sha256:<64 hex>")
    lease_path = _absolute_path(raw.get("lease_path"), "lease_path")
    max_age = raw.get("max_state_age_seconds", 5.0)
    ack_timeout = raw.get("ack_timeout_seconds", 15.0)
    poll = raw.get("poll_interval_seconds", 0.25)
    if not _finite_positive(max_age) or float(max_age) > 30:
        raise AdmissionError("max_state_age_seconds must be within (0, 30]")
    if not _finite_positive(ack_timeout) or float(ack_timeout) > 120:
        raise AdmissionError("ack_timeout_seconds must be within (0, 120]")
    if not _finite_positive(poll) or float(poll) > 5:
        raise AdmissionError("poll_interval_seconds must be within (0, 5]")
    rank_rows = raw.get("ranks")
    if not isinstance(rank_rows, list) or len(rank_rows) != 4:
        raise AdmissionError("admission config must contain exactly four ranks")
    endpoints: list[RankEndpoint] = []
    for row in rank_rows:
        if not isinstance(row, Mapping):
            raise AdmissionError("rank endpoint must be an object")
        rank = row.get("rank")
        if not isinstance(rank, str) or rank not in {"0", "1", "2", "3"}:
            raise AdmissionError("rank endpoint must be rank 0, 1, 2 or 3")
        host_value = row.get("host")
        if host_value in (None, "", "local"):
            host = None
        elif isinstance(host_value, str):
            host = host_value
        else:
            raise AdmissionError("remote host must be a string or local")
        if host is not None and not HOST_RE.fullmatch(host):
            raise AdmissionError("remote host contains unsafe characters")
        identity_value = row.get("ssh_identity_file")
        ssh_identity_file = None if identity_value in (None, "") else _absolute_path(
            identity_value, "ssh_identity_file"
        )
        endpoints.append(
            RankEndpoint(
                rank=rank,
                host=host,
                ssh_identity_file=ssh_identity_file,
                admission_state_path=_absolute_path(row.get("admission_state_path"), "admission_state_path"),
                request_state_path=_absolute_path(row.get("request_state_path"), "request_state_path"),
            )
        )
    if {endpoint.rank for endpoint in endpoints} != {"0", "1", "2", "3"}:
        raise AdmissionError("admission config ranks must be exactly 0, 1, 2 and 3")
    endpoints.sort(key=lambda endpoint: endpoint.rank)
    return AdmissionConfig(
        epoch=epoch,
        candidate_id=candidate,
        candidate_source_sha=source,
        profile_sha256=profile,
        image_id=image,
        lease_path=lease_path,
        max_state_age_seconds=float(max_age),
        ack_timeout_seconds=float(ack_timeout),
        poll_interval_seconds=float(poll),
        ranks=tuple(endpoints),
    )


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def request_state(*, config: AdmissionConfig, active: bool, lease_id: str | None, row_id: str | None, now: float) -> dict[str, Any]:
    if active:
        if not isinstance(lease_id, str) or not LEASE_RE.fullmatch(lease_id):
            raise AdmissionError("active request state requires a full lease ID")
        row_id = _safe_identity(row_id, "row_id")
    elif lease_id is not None or row_id is not None:
        raise AdmissionError("inactive request state cannot retain lease identity")
    return {
        "schema": REQUEST_SCHEMA,
        "epoch": config.epoch,
        "candidate_id": config.candidate_id,
        "candidate_source_sha": config.candidate_source_sha,
        "profile_sha256": config.profile_sha256,
        "image_id": config.image_id,
        "active": active,
        "lease_id": lease_id,
        "row_id": row_id,
        "updated_at": now,
    }


def validate_request_state(
    value: Any,
    *,
    epoch: str,
    candidate_id: str,
    candidate_source_sha: str,
    profile_sha256: str,
    image_id: str,
) -> dict[str, Any]:
    """Validate the guard-facing request flag without guessing defaults."""
    if not isinstance(value, Mapping) or value.get("schema") != REQUEST_SCHEMA:
        raise AdmissionError("request-state schema mismatch")
    if (
        value.get("epoch") != epoch
        or value.get("candidate_id") != candidate_id
        or value.get("candidate_source_sha") != candidate_source_sha
        or value.get("profile_sha256") != profile_sha256
        or value.get("image_id") != image_id
    ):
        raise AdmissionError("request-state identity mismatch")
    active = value.get("active")
    if not isinstance(active, bool):
        raise AdmissionError("request-state active flag must be boolean")
    updated_at = value.get("updated_at")
    if not _finite_positive(updated_at) or float(updated_at) > time.time() + 1:
        raise AdmissionError("request-state timestamp is invalid")
    lease_id = value.get("lease_id")
    row_id = value.get("row_id")
    if active:
        if not isinstance(lease_id, str) or not LEASE_RE.fullmatch(lease_id):
            raise AdmissionError("active request-state lease ID is invalid")
        _safe_identity(row_id, "row_id")
    elif lease_id is not None or row_id is not None:
        raise AdmissionError("inactive request state retained lease identity")
    return dict(value)


def validate_guard_state(
    state: Any,
    *,
    config: AdmissionConfig,
    endpoint: RankEndpoint,
    now: float,
    expected_active: bool | None = None,
    expected_lease_id: str | None = None,
    require_admit: bool = True,
) -> dict[str, Any]:
    if not isinstance(state, Mapping) or state.get("schema") != GUARD_SCHEMA:
        raise AdmissionError(f"rank {endpoint.rank}: guard state schema mismatch")
    if state.get("rank") != endpoint.rank or state.get("epoch") != config.epoch:
        raise AdmissionError(f"rank {endpoint.rank}: guard rank/epoch mismatch")
    if (
        state.get("candidate_id") != config.candidate_id
        or state.get("candidate_source_sha") != config.candidate_source_sha
        or state.get("profile_sha256") != config.profile_sha256
        or state.get("image_id") != config.image_id
    ):
        raise AdmissionError(f"rank {endpoint.rank}: guard candidate/source/profile/image mismatch")
    sequence = state.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise AdmissionError(f"rank {endpoint.rank}: guard sequence is invalid")
    container = state.get("container_id")
    if not isinstance(container, str) or not CONTAINER_RE.fullmatch(container):
        raise AdmissionError(f"rank {endpoint.rank}: guard container ID is not immutable")
    timestamp = state.get("ts")
    if not _finite_positive(timestamp) or float(timestamp) > now + 1 or now - float(timestamp) > config.max_state_age_seconds:
        raise AdmissionError(f"rank {endpoint.rank}: guard state is stale or future-dated")
    if state.get("hard_reasons") != []:
        raise AdmissionError(f"rank {endpoint.rank}: guard has hard reasons")
    available = state.get("mem_available")
    floor = state.get("admission_floor")
    if (
        not isinstance(available, int)
        or isinstance(available, bool)
        or available < 0
        or not isinstance(floor, int)
        or isinstance(floor, bool)
        or floor < 16 * (1 << 30)
    ):
        raise AdmissionError(f"rank {endpoint.rank}: guard memory/floor values are invalid")
    if state.get("admission") == "ADMIT" and available < floor:
        raise AdmissionError(f"rank {endpoint.rank}: guard admission contradicts memory floor")
    if state.get("status") == "HARD_ABORT" or state.get("action") == "STOP_LOCAL_AND_PEER":
        raise AdmissionError(f"rank {endpoint.rank}: guard is in hard-abort state")
    if expected_active is not None and state.get("request_active") is not expected_active:
        raise AdmissionError(f"rank {endpoint.rank}: request-active acknowledgement mismatch")
    if expected_active is True and state.get("request_lease_id") != expected_lease_id:
        raise AdmissionError(f"rank {endpoint.rank}: lease acknowledgement mismatch")
    if expected_active is False and state.get("request_lease_id") is not None:
        raise AdmissionError(f"rank {endpoint.rank}: inactive guard retained a lease")
    if require_admit and (
        state.get("admission") != "ADMIT"
        or state.get("status") not in {"GREEN", "WARNING"}
        or state.get("action") not in {"ADMIT", "ADMIT_WITH_WARNING"}
    ):
        raise AdmissionNotGranted(f"rank {endpoint.rank}: next row is NOT_ADMITTED")
    return dict(state)


class AdmissionCoordinator:
    def __init__(self, config: AdmissionConfig):
        self.config = config
        self._lease_id: str | None = None
        self._row_id: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "AdmissionCoordinator":
        return cls(load_config(path))

    @staticmethod
    def _remote_command(code: str, path: Path) -> str:
        return f"python3 -c {shlex.quote(code)} {shlex.quote(str(path))}"

    @staticmethod
    def _ssh_args(endpoint: RankEndpoint) -> list[str]:
        if endpoint.host is None:
            raise AdmissionError(f"rank {endpoint.rank}: local endpoint has no SSH host")
        args = ["ssh"]
        if endpoint.ssh_identity_file is not None:
            args.extend(["-i", str(endpoint.ssh_identity_file), "-o", "IdentitiesOnly=yes"])
        args.extend(
            [
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "UserKnownHostsFile=" + os.environ.get("DSV41_SSH_KNOWN_HOSTS", str(Path.home() / ".ssh/known_hosts_crs812_fabric")),
                endpoint.host,
            ]
        )
        return args

    def _read_bytes(self, endpoint: RankEndpoint, path: Path) -> bytes:
        if endpoint.host is None:
            try:
                return path.read_bytes()
            except OSError as exc:
                raise AdmissionError(f"rank {endpoint.rank}: cannot read {path}: {exc}") from exc
        code = "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
        proc = subprocess.run(
            [*self._ssh_args(endpoint), self._remote_command(code, path)],
            capture_output=True,
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            raise AdmissionError(f"rank {endpoint.rank}: remote state read failed: {proc.stderr.decode(errors='replace').strip()}")
        return proc.stdout

    def _write_json(self, endpoint: RankEndpoint, path: Path, value: Mapping[str, Any]) -> None:
        if endpoint.host is None:
            atomic_write_json(path, value)
            return
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        code = """import os, pathlib, sys
p = pathlib.Path(sys.argv[1])
p.parent.mkdir(parents=True, exist_ok=True)
t = p.with_name('.' + p.name + '.tmp-' + str(os.getpid()))
try:
    descriptor = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(sys.stdin.buffer.read())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(t, p)
    directory = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    try:
        t.unlink()
    except FileNotFoundError:
        pass
"""
        proc = subprocess.run(
            [*self._ssh_args(endpoint), self._remote_command(code, path)],
            input=payload,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if proc.returncode != 0:
            raise AdmissionError(f"rank {endpoint.rank}: remote request-state write failed: {proc.stderr.decode(errors='replace').strip()}")

    def _read_guard(self, endpoint: RankEndpoint) -> dict[str, Any]:
        try:
            value = strict_json_loads(self._read_bytes(endpoint, endpoint.admission_state_path))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise AdmissionError(f"rank {endpoint.rank}: guard state is invalid JSON") from exc
        return validate_guard_state(value, config=self.config, endpoint=endpoint, now=time.time())

    def _write_request(self, *, active: bool, lease_id: str | None, row_id: str | None) -> None:
        value = request_state(
            config=self.config,
            active=active,
            lease_id=lease_id,
            row_id=row_id,
            now=time.time(),
        )
        written: list[RankEndpoint] = []
        try:
            for endpoint in self.config.ranks:
                self._write_json(endpoint, endpoint.request_state_path, value)
                written.append(endpoint)
        except Exception:
            if active:
                inactive = request_state(config=self.config, active=False, lease_id=None, row_id=None, now=time.time())
                for endpoint in written:
                    with contextlib.suppress(Exception):
                        self._write_json(endpoint, endpoint.request_state_path, inactive)
            raise

    def _wait_ack(self, *, active: bool, lease_id: str | None, require_admit: bool) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.config.ack_timeout_seconds
        last_errors: list[str] = []
        while time.monotonic() < deadline:
            rows: list[dict[str, Any]] = []
            last_errors = []
            for endpoint in self.config.ranks:
                try:
                    raw = strict_json_loads(self._read_bytes(endpoint, endpoint.admission_state_path))
                    rows.append(
                        validate_guard_state(
                            raw,
                            config=self.config,
                            endpoint=endpoint,
                            now=time.time(),
                            expected_active=active,
                            expected_lease_id=lease_id,
                            require_admit=require_admit,
                        )
                    )
                except AdmissionNotGranted:
                    raise
                except (AdmissionError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                    last_errors.append(str(exc))
            if len(rows) == len(self.config.ranks):
                return rows
            time.sleep(self.config.poll_interval_seconds)
        raise AdmissionError("guard lease acknowledgement timed out: " + " | ".join(last_errors))

    def _create_lock(self, row_id: str, lease_id: str) -> None:
        path = self.config.lease_path
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema": REQUEST_SCHEMA,
            "epoch": self.config.epoch,
            "candidate_id": self.config.candidate_id,
            "candidate_source_sha": self.config.candidate_source_sha,
            "profile_sha256": self.config.profile_sha256,
            "image_id": self.config.image_id,
            "active": True,
            "lease_id": lease_id,
            "row_id": row_id,
            "controller_pid": os.getpid(),
            "created_at": time.time(),
        }
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise AdmissionError(f"another or stale request lease exists: {path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def acquire(self, row_id: str) -> dict[str, Any]:
        # A soft admission change during acknowledgement occurs BEFORE any
        # request is submitted. The one-attempt path rolls the lease back;
        # retry only that safe admission phase, never model generation.
        deadline = time.monotonic() + 600
        while True:
            try:
                return self._acquire_once(row_id, deadline)
            except AdmissionNotGranted:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(2)

    def _acquire_once(self, row_id: str, deadline: float) -> dict[str, Any]:
        if self._lease_id is not None:
            raise AdmissionError("coordinator already holds a request lease")
        row_id = _safe_identity(row_id, "row_id")
        lease_id = secrets.token_hex(32)
        self._create_lock(row_id, lease_id)
        active_written = False
        try:
            admission_deadline = deadline
            while True:
                try:
                    preflight = [self._read_guard(endpoint) for endpoint in self.config.ranks]
                    for endpoint, state in zip(self.config.ranks, preflight, strict=True):
                        validate_guard_state(state, config=self.config, endpoint=endpoint,
                                             now=time.time(), expected_active=False, require_admit=True)
                    break
                except AdmissionNotGranted:
                    if time.monotonic() >= admission_deadline:
                        raise
                    time.sleep(2)
            self._write_request(active=True, lease_id=lease_id, row_id=row_id)
            active_written = True
            acknowledged = self._wait_ack(active=True, lease_id=lease_id, require_admit=True)
            self._lease_id = lease_id
            self._row_id = row_id
            return {"lease_id": lease_id, "row_id": row_id, "guard_states": acknowledged}
        except Exception:
            cleanup_ok = True
            if active_written:
                try:
                    self._write_request(active=False, lease_id=None, row_id=None)
                    self._wait_ack(active=False, lease_id=None, require_admit=False)
                except Exception:
                    cleanup_ok = False
            if cleanup_ok:
                with contextlib.suppress(FileNotFoundError):
                    self.config.lease_path.unlink()
            raise

    def release(self) -> None:
        if self._lease_id is None:
            raise AdmissionError("coordinator does not hold a request lease")
        self._write_request(active=False, lease_id=None, row_id=None)
        self._wait_ack(active=False, lease_id=None, require_admit=False)
        self.config.lease_path.unlink()
        self._lease_id = None
        self._row_id = None

    @contextlib.contextmanager
    def request(self, row_id: str) -> Iterator[dict[str, Any]]:
        lease = self.acquire(row_id)
        try:
            yield lease
        finally:
            self.release()
