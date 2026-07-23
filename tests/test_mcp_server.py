#!/usr/bin/env python3
"""
Tests for the Merlina MCP server (mcp_server.py).

The MCP server is a thin HTTP client to the Merlina REST API, so these tests
mock the httpx client and assert that each tool issues the right request and
surfaces results/errors readably. No running server (and no torch) required.

Runnable standalone (`python tests/test_mcp_server.py`) or under pytest.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mcp_server


class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeClient:
    """Records requests and returns canned responses."""

    def __init__(self, response=None, raise_error=None):
        self.calls = []
        self._response = response or FakeResponse(json_data={"ok": True})
        self._raise_error = raise_error

    async def request(self, method, path, params=None, json=None, files=None, data=None):
        self.calls.append({
            "method": method, "path": path, "params": params,
            "json": json, "files": files, "data": data,
        })
        if self._raise_error is not None:
            raise self._raise_error
        return self._response


def run(coro):
    return asyncio.run(coro)


def test_list_jobs_issues_get():
    client = FakeClient(FakeResponse(json_data={"job_1": {"status": "completed"}}))
    mcp_server.set_client(client)
    try:
        out = run(mcp_server.list_jobs())
    finally:
        mcp_server.set_client(None)

    assert client.calls[0]["method"] == "GET"
    assert client.calls[0]["path"] == "/jobs"
    assert json.loads(out) == {"job_1": {"status": "completed"}}
    print("✓ list_jobs issues GET /jobs")


def test_get_job_status_path():
    client = FakeClient(FakeResponse(json_data={"job_id": "job_x", "status": "running"}))
    mcp_server.set_client(client)
    try:
        out = run(mcp_server.get_job_status("job_x"))
    finally:
        mcp_server.set_client(None)

    assert client.calls[0]["path"] == "/status/job_x"
    assert json.loads(out)["status"] == "running"
    print("✓ get_job_status hits /status/{job_id}")


def test_job_history_params():
    client = FakeClient(FakeResponse(json_data=[]))
    mcp_server.set_client(client)
    try:
        run(mcp_server.get_job_history(limit=5, offset=10, status="failed"))
    finally:
        mcp_server.set_client(None)

    params = client.calls[0]["params"]
    assert params == {"limit": 5, "offset": 10, "status": "failed"}
    print("✓ get_job_history forwards limit/offset/status")


def test_start_training_builds_config():
    client = FakeClient(FakeResponse(json_data={"job_id": "job_1", "status": "queued"}))
    mcp_server.set_client(client)
    try:
        out = run(
            mcp_server.start_training(
                output_name="my-model",
                base_model="meta-llama/Meta-Llama-3-8B-Instruct",
                training_mode="sft",
                dataset_repo_id="org/dataset",
                priority="high",
                overrides={"lora_r": 32, "beta": 0.2},
            )
        )
    finally:
        mcp_server.set_client(None)

    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/train"
    assert call["params"] == {"priority": "high"}

    body = call["json"]
    assert body["output_name"] == "my-model"
    assert body["training_mode"] == "sft"
    assert body["dataset"]["source"]["repo_id"] == "org/dataset"
    assert body["dataset"]["source"]["source_type"] == "huggingface"
    assert body["dataset"]["format"]["format_type"] == "chatml"
    # overrides win / are merged at top level
    assert body["lora_r"] == 32
    assert body["beta"] == 0.2
    assert json.loads(out)["status"] == "queued"
    print("✓ start_training builds a valid config with overrides merged")


def test_start_training_upload_source():
    client = FakeClient(FakeResponse(json_data={"job_id": "job_1", "status": "queued"}))
    mcp_server.set_client(client)
    try:
        run(
            mcp_server.start_training(
                output_name="m",
                dataset_repo_id="abc123",
                dataset_source_type="upload",
            )
        )
    finally:
        mcp_server.set_client(None)

    source = client.calls[0]["json"]["dataset"]["source"]
    assert source["source_type"] == "upload"
    assert source["dataset_id"] == "abc123"
    assert "repo_id" not in source
    print("✓ start_training maps upload source to dataset_id")


def test_preview_dataset_body():
    client = FakeClient(FakeResponse(json_data={"status": "success", "samples": []}))
    mcp_server.set_client(client)
    try:
        run(mcp_server.preview_dataset(repo_id="org/ds", limit=3, offset=2))
    finally:
        mcp_server.set_client(None)

    call = client.calls[0]
    assert call["path"] == "/dataset/preview"
    assert call["params"] == {"limit": 3, "offset": 2}
    assert call["json"]["source"]["repo_id"] == "org/ds"
    print("✓ preview_dataset posts a DatasetConfig body with limit/offset")


def test_upload_dataset_posts_multipart():
    import tempfile

    client = FakeClient(FakeResponse(json_data={
        "status": "success", "dataset_id": "upload_x", "filename": "d.jsonl", "size": 2,
    }))
    mcp_server.set_client(client)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.jsonl")
            with open(p, "w") as f:
                f.write('{"prompt": "hi", "chosen": "yo"}\n')
            out = run(mcp_server.upload_dataset(p))
    finally:
        mcp_server.set_client(None)

    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/dataset/upload-file"
    field, (name, content) = call["files"][0]
    assert field == "file"
    assert name == "d.jsonl"
    assert b'"prompt"' in content
    parsed = json.loads(out)
    assert parsed["dataset_id"] == "upload_x"
    assert "upload" in parsed["hint"]
    print("✓ upload_dataset posts the file as multipart and returns the dataset_id")


def test_upload_dataset_missing_file_is_friendly():
    client = FakeClient()
    mcp_server.set_client(client)
    try:
        out = run(mcp_server.upload_dataset("/nonexistent/nope.jsonl"))
    finally:
        mcp_server.set_client(None)

    parsed = json.loads(out)
    assert parsed["error"] == "file_not_found"
    assert client.calls == []  # never hit the network
    print("✓ upload_dataset surfaces a missing local file without an HTTP call")


def test_upload_image_dataset_from_manifest():
    import tempfile

    client = FakeClient(FakeResponse(json_data={
        "status": "success", "batch_id": "images_x",
        "jsonl_path": "/srv/uploads/images_x/train.jsonl", "image_count": 2,
    }))
    mcp_server.set_client(client)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.png", "b.png"):
                with open(os.path.join(tmp, name), "wb") as f:
                    f.write(b"fakepng-" + name.encode())
            manifest = os.path.join(tmp, "train.jsonl")
            with open(manifest, "w") as f:
                f.write(json.dumps({"prompt": "a cat", "image": "a.png"}) + "\n")
                f.write(json.dumps({"caption": "a dog", "image": os.path.join(tmp, "b.png")}) + "\n")
            out = run(mcp_server.upload_image_dataset(manifest))
    finally:
        mcp_server.set_client(None)

    call = client.calls[0]
    assert call["path"] == "/dataset/upload-images"
    names = [f[1][0] for f in call["files"]]
    assert names == ["a.png", "b.png"]
    captions = json.loads(call["data"]["captions"])
    assert captions == {"a.png": "a cat", "b.png": "a dog"}
    parsed = json.loads(out)
    assert parsed["jsonl_path"] == "/srv/uploads/images_x/train.jsonl"
    assert "dataset_jsonl_path" in parsed["hint"]
    print("✓ upload_image_dataset ships manifest-referenced images with captions")


def test_upload_image_dataset_from_directory_with_sidecars():
    import tempfile

    client = FakeClient(FakeResponse(json_data={
        "status": "success", "jsonl_path": "/srv/uploads/images_y/train.jsonl",
    }))
    mcp_server.set_client(client)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "cat.jpg"), "wb") as f:
                f.write(b"fakejpg")
            with open(os.path.join(tmp, "cat.txt"), "w") as f:
                f.write("a fluffy cat\n")
            with open(os.path.join(tmp, "dog.jpg"), "wb") as f:
                f.write(b"fakejpg2")
            with open(os.path.join(tmp, "notes.md"), "w") as f:
                f.write("not an image")
            out = run(mcp_server.upload_image_dataset(tmp))
    finally:
        mcp_server.set_client(None)

    call = client.calls[0]
    names = sorted(f[1][0] for f in call["files"])
    assert names == ["cat.jpg", "dog.jpg"]  # sidecar + non-image skipped
    captions = json.loads(call["data"]["captions"])
    assert captions == {"cat.jpg": "a fluffy cat"}  # dog.jpg falls back server-side
    assert json.loads(out)["jsonl_path"].endswith("train.jsonl")
    print("✓ upload_image_dataset reads a directory with .txt caption sidecars")


def test_upload_image_dataset_missing_path_is_friendly():
    client = FakeClient()
    mcp_server.set_client(client)
    try:
        out = run(mcp_server.upload_image_dataset("/nonexistent/imgs"))
    finally:
        mcp_server.set_client(None)

    parsed = json.loads(out)
    assert parsed["error"] == "path_not_found"
    assert client.calls == []
    print("✓ upload_image_dataset surfaces a missing local path without an HTTP call")


def test_http_error_is_returned_not_raised():
    client = FakeClient(FakeResponse(status_code=404, json_data={"detail": "Job not found"}))
    mcp_server.set_client(client)
    try:
        out = run(mcp_server.get_job_status("nope"))
    finally:
        mcp_server.set_client(None)

    parsed = json.loads(out)
    assert parsed["error"] == "http_404"
    assert parsed["detail"] == "Job not found"
    print("✓ HTTP errors are surfaced as readable error dicts")


def test_connection_error_is_friendly():
    import httpx

    client = FakeClient(raise_error=httpx.ConnectError("refused"))
    mcp_server.set_client(client)
    try:
        out = run(mcp_server.health_check())
    finally:
        mcp_server.set_client(None)

    parsed = json.loads(out)
    assert parsed["error"] == "connection_failed"
    assert "merlina serve" in parsed["hint"]
    print("✓ connection failures yield a friendly hint to start the server")


def test_build_server_registers_all_tools():
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ImportError:
        print("⊘ skipped build_server check (mcp SDK not installed)")
        return

    server = mcp_server.build_server()
    tools = run(server.list_tools())
    names = {t.name for t in tools}
    expected = {fn.__name__ for fn in mcp_server.TOOLS}
    assert expected.issubset(names), f"missing tools: {expected - names}"
    print(f"✓ build_server registers all {len(expected)} tools")


def main():
    tests = [
        test_list_jobs_issues_get,
        test_get_job_status_path,
        test_job_history_params,
        test_start_training_builds_config,
        test_start_training_upload_source,
        test_preview_dataset_body,
        test_upload_dataset_posts_multipart,
        test_upload_dataset_missing_file_is_friendly,
        test_upload_image_dataset_from_manifest,
        test_upload_image_dataset_from_directory_with_sidecars,
        test_upload_image_dataset_missing_path_is_friendly,
        test_http_error_is_returned_not_raised,
        test_connection_error_is_friendly,
        test_build_server_registers_all_tools,
    ]
    print("=" * 60)
    print("Testing Merlina MCP server")
    print("=" * 60)
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"✗ {t.__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"{failed} test(s) failed")
        return 1
    print("All MCP server tests passed ✨")
    return 0


if __name__ == "__main__":
    sys.exit(main())
