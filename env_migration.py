"""
Migrate a Merlina ``.env`` from the legacy storage variable names to the
role-named scheme (``merlina config migrate-env``).

Legacy                         New
-----------------------------  ---------------------------------------------
S3_ENDPOINT_URL                STORAGE_INTERNAL_ENDPOINT
S3_DATASET_BUCKET              STORAGE_INTERNAL_BUCKET
S3_ACCESS_KEY / S3_SECRET_KEY  STORAGE_INTERNAL_ACCESS_KEY / _SECRET_KEY
S3_REGION                      STORAGE_INTERNAL_REGION
R2_*                           STORAGE_EXTERNAL_* (same shape)
DATASET_STORE                  DATASET_TIER
PREFER_INTERNAL                DATASET_MIRROR
MERLINA_ARTIFACT_S3_PREFIX     ARTIFACT_PREFIX
MERLINA_ARTIFACT_STORE=s3 +    ARTIFACT_STORE=<tier> (+ ARTIFACT_BUCKET when the
  _S3_ENDPOINT/_S3_BUCKET/       bucket differs) — only when the endpoint and
  _S3_REGION/_ACCESS_KEY/        keys equal that tier's; otherwise the block is a
  _SECRET_KEY                    custom location and is kept as is.

Guarantees: the plan and every message name keys only, never values (values
are compared in memory); comments, ordering and unrelated lines are kept
byte-for-byte; ``--write`` first saves a timestamped backup (mode 600) and
leaves the file at mode 600; a second run reports nothing to do.

Deliberately dependency-free (no pydantic, no app config) so it runs anywhere.
"""

from __future__ import annotations

import io
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

RENAMES = {
    "S3_ENDPOINT_URL": "STORAGE_INTERNAL_ENDPOINT",
    "S3_DATASET_BUCKET": "STORAGE_INTERNAL_BUCKET",
    "S3_ACCESS_KEY": "STORAGE_INTERNAL_ACCESS_KEY",
    "S3_SECRET_KEY": "STORAGE_INTERNAL_SECRET_KEY",
    "S3_REGION": "STORAGE_INTERNAL_REGION",
    "S3_ADMIN_ACCESS_KEY": "STORAGE_INTERNAL_ADMIN_ACCESS_KEY",
    "S3_ADMIN_SECRET_KEY": "STORAGE_INTERNAL_ADMIN_SECRET_KEY",
    "R2_ENDPOINT_URL": "STORAGE_EXTERNAL_ENDPOINT",
    "R2_DATASET_BUCKET": "STORAGE_EXTERNAL_BUCKET",
    "R2_ACCESS_KEY": "STORAGE_EXTERNAL_ACCESS_KEY",
    "R2_SECRET_KEY": "STORAGE_EXTERNAL_SECRET_KEY",
    "R2_REGION": "STORAGE_EXTERNAL_REGION",
    "R2_ADMIN_ACCESS_KEY": "STORAGE_EXTERNAL_ADMIN_ACCESS_KEY",
    "R2_ADMIN_SECRET_KEY": "STORAGE_EXTERNAL_ADMIN_SECRET_KEY",
    "DATASET_STORE": "DATASET_TIER",
    "PREFER_INTERNAL": "DATASET_MIRROR",
    "MERLINA_ARTIFACT_S3_PREFIX": "ARTIFACT_PREFIX",
}
ARTIFACT_BLOCK = (
    "MERLINA_ARTIFACT_STORE", "MERLINA_ARTIFACT_S3_BUCKET", "MERLINA_ARTIFACT_S3_ENDPOINT",
    "MERLINA_ARTIFACT_S3_REGION", "MERLINA_ARTIFACT_ACCESS_KEY", "MERLINA_ARTIFACT_SECRET_KEY",
    "MERLINA_ARTIFACT_SESSION_TOKEN",
)
TIERS = ("external", "internal")  # match order when collapsing artifacts

_LINE = re.compile(r"^(?P<lead>\s*)(?P<export>export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<eq>\s*=)(?P<rest>.*)$")


@dataclass
class Change:
    kind: str           # "rename" | "remove" | "set"
    key: str
    new_key: Optional[str] = None
    reason: str = ""

    def describe(self) -> str:
        if self.kind == "rename":
            return f"rename  {self.key} -> {self.new_key}" + (f"  ({self.reason})" if self.reason else "")
        if self.kind == "set":
            return f"replace {self.key} -> {self.new_key}  ({self.reason})"
        return f"remove  {self.key}  ({self.reason})"


@dataclass
class Plan:
    path: str
    changes: list[Change] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # internal: line rewrites, keyed by upper-cased key
    _rename: dict = field(default_factory=dict)
    _remove: set = field(default_factory=set)
    _replace: dict = field(default_factory=dict)  # key -> (new_key, literal value)

    @property
    def has_changes(self) -> bool:
        return bool(self.changes)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _values(text: str) -> dict:
    """Parsed values keyed by upper-cased name (last one wins), in memory only."""
    from dotenv import dotenv_values
    return {k.upper(): v for k, v in dotenv_values(stream=io.StringIO(text)).items() if v is not None}


def _norm_endpoint(url: Optional[str]) -> str:
    return (url or "").strip().rstrip("/").lower()


def _tier_value(env: dict, tier: str, field_name: str) -> Optional[str]:
    """A tier setting from the new name, else its legacy alias."""
    new = f"STORAGE_{tier.upper()}_{field_name}"
    if new in env:
        return env[new]
    for old, mapped in RENAMES.items():
        if mapped == new and old in env:
            return env[old]
    return None


def plan_migration(text: str, path: str = ".env") -> Plan:
    env = _values(text)
    plan = Plan(path=path)

    for old, new in RENAMES.items():
        if old not in env:
            continue
        if new in env:
            if env[new] == env[old]:
                plan.changes.append(Change("remove", old, reason=f"duplicate of {new}"))
                plan._remove.add(old)
            else:
                plan.conflicts.append(f"{old} and {new} are both set with different values "
                                      f"({new} wins at runtime); left both in place — resolve by hand")
        else:
            plan.changes.append(Change("rename", old, new))
            plan._rename[old] = new

    present = [k for k in ARTIFACT_BLOCK if k in env]
    store = (env.get("MERLINA_ARTIFACT_STORE") or "").strip().lower()
    if present and "ARTIFACT_STORE" in env and "MERLINA_ARTIFACT_STORE" in env:
        plan.conflicts.append("MERLINA_ARTIFACT_* settings are present alongside ARTIFACT_STORE "
                              "(ARTIFACT_STORE wins); left in place — remove the legacy block by hand")
    elif present and not store:
        plan.notes.append("MERLINA_ARTIFACT_* credentials without MERLINA_ARTIFACT_STORE are only "
                          "read by remote workers (wire format); left as is")
    elif store == "hf_hub":
        plan.changes.append(Change("rename", "MERLINA_ARTIFACT_STORE", "ARTIFACT_STORE"))
        plan._rename["MERLINA_ARTIFACT_STORE"] = "ARTIFACT_STORE"
        unused = [k for k in present if k != "MERLINA_ARTIFACT_STORE"]
        if unused:
            plan.notes.append("unused with an hf_hub artifact store (left as is): " + ", ".join(unused))
    elif store == "s3":
        _plan_artifact_collapse(env, plan, present)
    elif store:
        plan.notes.append("MERLINA_ARTIFACT_STORE has an unrecognised value; left as is")
    return plan


def _plan_artifact_collapse(env: dict, plan: Plan, present: list) -> None:
    endpoint = _norm_endpoint(env.get("MERLINA_ARTIFACT_S3_ENDPOINT"))
    ak, sk = env.get("MERLINA_ARTIFACT_ACCESS_KEY"), env.get("MERLINA_ARTIFACT_SECRET_KEY")
    match = None
    if not env.get("MERLINA_ARTIFACT_SESSION_TOKEN"):
        for tier in TIERS:
            t_endpoint = _tier_value(env, tier, "ENDPOINT")
            if not (t_endpoint and _tier_value(env, tier, "BUCKET")):
                continue
            if _norm_endpoint(t_endpoint) != endpoint:
                continue
            if (ak is None and sk is None) or (ak == _tier_value(env, tier, "ACCESS_KEY")
                                               and sk == _tier_value(env, tier, "SECRET_KEY")):
                match = tier
                break
    if match is None:
        plan.notes.append("MERLINA_ARTIFACT_* kept: a custom artifact location (its endpoint or "
                          "credentials differ from every tier, or it uses a session token); "
                          "still supported")
        return
    plan.changes.append(Change("set", "MERLINA_ARTIFACT_STORE", "ARTIFACT_STORE",
                               reason=f"collapsed onto the {match} tier: same endpoint and credentials"))
    plan._replace["MERLINA_ARTIFACT_STORE"] = ("ARTIFACT_STORE", match)
    bucket = env.get("MERLINA_ARTIFACT_S3_BUCKET")
    if bucket is not None:
        if bucket != _tier_value(env, match, "BUCKET"):
            plan.changes.append(Change("rename", "MERLINA_ARTIFACT_S3_BUCKET", "ARTIFACT_BUCKET",
                                       reason=f"differs from the {match} tier's bucket"))
            plan._rename["MERLINA_ARTIFACT_S3_BUCKET"] = "ARTIFACT_BUCKET"
        else:
            plan.changes.append(Change("remove", "MERLINA_ARTIFACT_S3_BUCKET",
                                       reason=f"same as the {match} tier's bucket"))
            plan._remove.add("MERLINA_ARTIFACT_S3_BUCKET")
    region = env.get("MERLINA_ARTIFACT_S3_REGION")
    if region is not None and (region or "auto") != (_tier_value(env, match, "REGION") or "auto"):
        plan.notes.append(f"MERLINA_ARTIFACT_S3_REGION differed from the {match} tier's region; "
                          f"the tier's region will be used")
    for key in ("MERLINA_ARTIFACT_S3_ENDPOINT", "MERLINA_ARTIFACT_S3_REGION",
                "MERLINA_ARTIFACT_ACCESS_KEY", "MERLINA_ARTIFACT_SECRET_KEY"):
        if key in present:
            plan.changes.append(Change("remove", key, reason=f"now taken from the {match} tier"))
            plan._remove.add(key)


def apply_plan(text: str, plan: Plan) -> str:
    """Rewrite ``text`` per ``plan``; every other line is kept byte-for-byte."""
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        m = _LINE.match(line.rstrip("\r\n")) if stripped and not stripped.startswith("#") else None
        if not m:
            out.append(line)
            continue
        key = m.group("key").upper()
        ending = line[len(line.rstrip("\r\n")):]
        export = m.group("export") or ""
        if key in plan._remove:
            continue
        if key in plan._replace:
            new_key, literal = plan._replace[key]
            out.append(f"{m.group('lead')}{export}{new_key}={literal}{ending}")
        elif key in plan._rename:
            out.append(f"{m.group('lead')}{export}{plan._rename[key]}{m.group('eq')}{m.group('rest')}{ending}")
        else:
            out.append(line)
    return "".join(out)


def migrate_file(path: str, *, write: bool = False, now: Optional[datetime] = None) -> tuple[Plan, Optional[str]]:
    """Plan (and with ``write``, apply) the migration of ``path``.
    Returns ``(plan, backup_path)``."""
    text = _read(path)
    plan = plan_migration(text, path)
    if not write or not plan.has_changes:
        return plan, None
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    backup = f"{path}.bak-{stamp}"
    shutil.copyfile(path, backup)
    os.chmod(backup, 0o600)
    new_text = apply_plan(text, plan)
    tmp = f"{path}.migrate-tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(new_text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return plan, backup


def format_plan(plan: Plan, *, write: bool, backup: Optional[str] = None) -> str:
    """Human-readable report. Names keys only — never values."""
    lines = []
    if not plan.has_changes and not plan.conflicts:
        lines.append(f"{plan.path}: nothing to do — already on the new storage names.")
    else:
        verb = "applied" if write and plan.has_changes else "planned (dry run; nothing written)"
        lines.append(f"{plan.path}: {len(plan.changes)} change(s) {verb}")
        lines.extend(f"  {c.describe()}" for c in plan.changes)
    if plan.conflicts:
        lines.append("conflicts (left untouched):")
        lines.extend(f"  ! {c}" for c in plan.conflicts)
    if plan.notes:
        lines.append("notes:")
        lines.extend(f"  - {n}" for n in plan.notes)
    if backup:
        lines.append(f"backup written: {backup} (mode 600)")
    elif plan.has_changes and not write:
        lines.append("Re-run with --write to apply (a timestamped backup is written first).")
    return "\n".join(lines)
