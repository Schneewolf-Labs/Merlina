"""
HuggingFace namespace discovery and repo-id resolution.

Merlina's ``output_name`` is a bare model name (it also names the local
directory under ``./models/``), so an upload has to decide *whose* Hub
namespace the repo belongs to. Historically we passed the bare name
straight to ``upload_folder()``, which fails with a 404 whenever the
repo lives under an organization — ``create_repo()`` resolves a bare
name against the token's account, but ``upload_folder()`` does not
resolve anything at all.

This module provides:

* :func:`resolve_hub_repo_id` — combine ``output_name`` with an optional
  namespace into a fully-qualified ``owner/name`` repo id.
* :func:`list_namespaces` — ask the Hub which namespaces a token can
  publish to (the user's own account plus every org they belong to), so
  the UI can offer a picker instead of making people guess.

Kept free of torch/transformers imports so the API layer can use it
without dragging in the ML stack.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Org roles that cannot create or push repos. Anything else (admin,
# write, contributor, or an unknown/absent role) is treated as writable —
# we would rather show a namespace the user can't push to than hide one
# they can.
_READ_ONLY_ROLES = {"read", "reader"}

# Same character set the Hub allows in a namespace or repo name.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class HFNamespaceError(RuntimeError):
    """Raised when the Hub can't tell us which namespaces a token has."""


def normalize_namespace(namespace: Optional[str]) -> Optional[str]:
    """Normalize a user-supplied namespace to ``None`` or a bare owner name.

    Accepts sloppy input (surrounding whitespace, a trailing slash, a
    full ``https://huggingface.co/org`` URL) and returns just the owner.
    """
    if not namespace:
        return None

    value = str(namespace).strip()
    if not value:
        return None

    # Tolerate a pasted profile URL.
    value = re.sub(r"^https?://(www\.)?huggingface\.co/", "", value)
    value = value.strip("/").strip()

    return value or None


def is_valid_name(value: str) -> bool:
    """True if ``value`` is a legal Hub namespace/repo name component."""
    return bool(_NAME_RE.match(value or ""))


def split_repo_id(repo_id: str) -> tuple[Optional[str], str]:
    """Split ``owner/name`` into ``(owner, name)``; owner is None if absent."""
    value = (repo_id or "").strip().strip("/")
    if "/" in value:
        owner, _, name = value.rpartition("/")
        return (owner or None), name
    return None, value


def resolve_hub_repo_id(output_name: str, namespace: Optional[str] = None) -> str:
    """Build the repo id to create/upload to on the Hub.

    Rules, in order:

    1. An ``output_name`` that already contains a ``/`` is taken as-is —
       an explicit fully-qualified id always wins.
    2. Otherwise, if a namespace is configured, return ``ns/output_name``.
    3. Otherwise, return the bare name and let ``create_repo()`` resolve
       it against the token's own account.

    Note that the *authoritative* repo id is whatever ``create_repo()``
    returns; callers should prefer that for the subsequent upload. This
    function decides what we ask for.
    """
    name = (output_name or "").strip().strip("/")
    ns = normalize_namespace(namespace)

    if "/" in name:
        return name
    if ns and name:
        return f"{ns}/{name}"
    return name


def list_namespaces(token: Optional[str]) -> Dict[str, Any]:
    """List the Hub namespaces ``token`` can publish to.

    Args:
        token: A HuggingFace access token.

    Returns:
        ``{"user": <username>, "namespaces": [{"name", "type",
        "full_name", "role", "can_write"}, ...]}`` with the user's own
        account first, followed by their organizations (alphabetical).

    Raises:
        HFNamespaceError: No token, an invalid token, or an unusable
            response from the Hub.
    """
    if not token:
        raise HFNamespaceError(
            "No HuggingFace token available. Provide a token or set "
            "HF_TOKEN in the server's .env file."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - dependency always present in prod
        raise HFNamespaceError(f"huggingface_hub is not installed: {exc}") from exc

    try:
        info = HfApi().whoami(token=token)
    except Exception as exc:
        message = str(exc)
        if "401" in message or "Invalid" in message or "Unauthorized" in message:
            raise HFNamespaceError(
                "HuggingFace rejected the token. Check that it is valid and "
                "has write permissions."
            ) from exc
        raise HFNamespaceError(f"Could not reach HuggingFace Hub: {message}") from exc

    if not isinstance(info, dict):
        raise HFNamespaceError("Unexpected response from HuggingFace whoami().")

    user = info.get("name")
    if not isinstance(user, str) or not user:
        raise HFNamespaceError("HuggingFace did not return a username for this token.")

    namespaces: List[Dict[str, Any]] = [{
        "name": user,
        "type": "user",
        "full_name": info.get("fullname") if isinstance(info.get("fullname"), str) else "",
        "role": "owner",
        "can_write": True,
    }]

    orgs = info.get("orgs")
    org_entries: List[Dict[str, Any]] = []
    if isinstance(orgs, list):
        for org in orgs:
            if not isinstance(org, dict):
                continue
            name = org.get("name")
            if not isinstance(name, str) or not name:
                continue
            role = org.get("roleInOrg")
            role = role if isinstance(role, str) else ""
            org_entries.append({
                "name": name,
                "type": "org",
                "full_name": org.get("fullname") if isinstance(org.get("fullname"), str) else "",
                "role": role,
                "can_write": role.lower() not in _READ_ONLY_ROLES,
            })

    org_entries.sort(key=lambda entry: entry["name"].lower())
    namespaces.extend(org_entries)

    logger.debug("Resolved %d HuggingFace namespaces for %s", len(namespaces), user)
    return {"user": user, "namespaces": namespaces}
