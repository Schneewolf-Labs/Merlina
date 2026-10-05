"""
Tests for HuggingFace namespace discovery and repo-id resolution.

Covers the bug where an upload with a bare ``output_name`` 404'd because
``create_repo()`` resolves a bare name against the token's own account
while ``upload_folder()`` does not resolve anything at all — an org
upload has to be fully qualified before it leaves Merlina.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hf_namespaces import (
    HFNamespaceError,
    is_valid_name,
    list_namespaces,
    normalize_namespace,
    resolve_hub_repo_id,
    split_repo_id,
)


# --------------------------------------------------------------------------
# normalize_namespace
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("Schneewolf-Labs", "Schneewolf-Labs"),
    ("  Schneewolf-Labs  ", "Schneewolf-Labs"),
    ("Schneewolf-Labs/", "Schneewolf-Labs"),
    ("/Schneewolf-Labs", "Schneewolf-Labs"),
    ("https://huggingface.co/Schneewolf-Labs", "Schneewolf-Labs"),
    ("https://www.huggingface.co/Schneewolf-Labs/", "Schneewolf-Labs"),
])
def test_normalize_namespace(raw, expected):
    assert normalize_namespace(raw) == expected


@pytest.mark.parametrize("value,valid", [
    ("Schneewolf-Labs", True),
    ("user_name.1", True),
    ("has space", False),
    ("org/model", False),
    ("", False),
])
def test_is_valid_name(value, valid):
    assert is_valid_name(value) is valid


def test_split_repo_id():
    assert split_repo_id("Schneewolf-Labs/Qwen3.6-27B") == ("Schneewolf-Labs", "Qwen3.6-27B")
    assert split_repo_id("Qwen3.6-27B") == (None, "Qwen3.6-27B")


# --------------------------------------------------------------------------
# resolve_hub_repo_id — the actual 404 fix
# --------------------------------------------------------------------------

def test_namespace_is_prefixed_onto_bare_name():
    assert resolve_hub_repo_id(
        "Qwen3.6-27B-Chud-LoRA", "Schneewolf-Labs"
    ) == "Schneewolf-Labs/Qwen3.6-27B-Chud-LoRA"


def test_bare_name_without_namespace_is_unchanged():
    # create_repo() resolves this against the token's own account.
    assert resolve_hub_repo_id("Qwen3.6-27B-Chud-LoRA", None) == "Qwen3.6-27B-Chud-LoRA"
    assert resolve_hub_repo_id("Qwen3.6-27B-Chud-LoRA", "") == "Qwen3.6-27B-Chud-LoRA"


def test_explicit_namespace_in_output_name_wins():
    assert resolve_hub_repo_id(
        "someone-else/model", "Schneewolf-Labs"
    ) == "someone-else/model"


def test_sloppy_input_is_normalized():
    assert resolve_hub_repo_id("  my-model ", " Schneewolf-Labs/ ") == "Schneewolf-Labs/my-model"


def test_empty_output_name_does_not_produce_dangling_slash():
    assert resolve_hub_repo_id("", "Schneewolf-Labs") == ""


# --------------------------------------------------------------------------
# list_namespaces
# --------------------------------------------------------------------------

def _patch_whoami(payload):
    """Patch huggingface_hub.HfApi so whoami() returns ``payload``."""
    api = MagicMock()
    api.whoami.return_value = payload
    module = MagicMock()
    module.HfApi.return_value = api
    return patch.dict(sys.modules, {"huggingface_hub": module}), api


def test_list_namespaces_returns_user_first_then_sorted_orgs():
    patcher, api = _patch_whoami({
        "name": "nbeerbower",
        "fullname": "N B",
        "orgs": [
            {"name": "zeta-org", "fullname": "Zeta", "roleInOrg": "write"},
            {"name": "Schneewolf-Labs", "fullname": "Schneewolf Labs", "roleInOrg": "admin"},
        ],
    })
    with patcher:
        result = list_namespaces("hf_token")

    api.whoami.assert_called_once_with(token="hf_token")
    assert result["user"] == "nbeerbower"

    names = [ns["name"] for ns in result["namespaces"]]
    assert names == ["nbeerbower", "Schneewolf-Labs", "zeta-org"]

    assert result["namespaces"][0]["type"] == "user"
    assert result["namespaces"][0]["can_write"] is True
    assert result["namespaces"][1]["type"] == "org"


def test_read_only_org_is_listed_but_flagged():
    patcher, _ = _patch_whoami({
        "name": "user",
        "orgs": [{"name": "readonly-org", "roleInOrg": "read"}],
    })
    with patcher:
        result = list_namespaces("hf_token")

    org = result["namespaces"][1]
    assert org["name"] == "readonly-org"
    assert org["can_write"] is False


def test_unknown_role_is_assumed_writable():
    patcher, _ = _patch_whoami({
        "name": "user",
        "orgs": [{"name": "mystery-org"}],
    })
    with patcher:
        result = list_namespaces("hf_token")

    assert result["namespaces"][1]["can_write"] is True


def test_no_orgs_returns_just_the_user():
    patcher, _ = _patch_whoami({"name": "solo", "orgs": []})
    with patcher:
        result = list_namespaces("hf_token")

    assert [ns["name"] for ns in result["namespaces"]] == ["solo"]


def test_missing_token_raises():
    with pytest.raises(HFNamespaceError, match="No HuggingFace token"):
        list_namespaces(None)


def test_invalid_token_raises_friendly_error():
    api = MagicMock()
    api.whoami.side_effect = RuntimeError("401 Client Error: Unauthorized")
    module = MagicMock()
    module.HfApi.return_value = api
    with patch.dict(sys.modules, {"huggingface_hub": module}):
        with pytest.raises(HFNamespaceError, match="rejected the token"):
            list_namespaces("bad_token")


def test_garbage_response_raises():
    patcher, _ = _patch_whoami(["not", "a", "dict"])
    with patcher:
        with pytest.raises(HFNamespaceError, match="Unexpected response"):
            list_namespaces("hf_token")


def test_response_without_username_raises():
    patcher, _ = _patch_whoami({"fullname": "No Name"})
    with patcher:
        with pytest.raises(HFNamespaceError, match="did not return a username"):
            list_namespaces("hf_token")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
