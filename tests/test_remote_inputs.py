"""Remote input handoff tests: snapshot, reconstruct, reject tampering and leaks."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.remote.artifacts import LocalArtifactStore
from src.remote.inputs import RemoteInputError, restore_remote_inputs, stage_remote_inputs
from src.remote.plan import check_remote_compatible


def config(source, additional=None):
    return {"dataset": {"source": source, "additional_sources": additional or []}}


def test_uploaded_snapshot_survives_upload_expiration(tmp_path):
    store = LocalArtifactStore(tmp_path / "store")
    data = b'{"prompt":"hi","chosen":"hello"}\n'
    uploads = {"123": {"filename": "/secret/local/customer.jsonl", "content": data}}
    original = config({"source_type": "upload", "dataset_id": "123",
                       "column_mapping": {"chosen": "answer"}})
    prepared = stage_remote_inputs(original, uploads, store)
    uploads.clear()
    assert original["dataset"]["source"]["source_type"] == "upload"
    assert "123" not in json.dumps(prepared)
    restored = restore_remote_inputs(prepared, store, tmp_path / "worker")
    source = restored["dataset"]["source"]
    assert Path(source["file_path"]).read_bytes() == data
    assert source["column_mapping"] == {"chosen": "answer"}
    manifests = list((tmp_path / "store").glob("*/input.json"))
    assert len(manifests) == 1
    assert "secret" not in manifests[0].read_text()
    assert "customer" not in manifests[0].read_text()


def test_local_additional_sources_snapshot_and_hf_unchanged(tmp_path):
    data = tmp_path / "customer.json"
    data.write_text('[{"prompt":"hey","chosen":"hi"}]')
    cfg = config({"source_type": "huggingface", "repo_id": "org/dataset"},
                 [{"source_type": "local_file", "file_path": str(data)}])
    store = LocalArtifactStore(tmp_path / "store")
    prepared = stage_remote_inputs(cfg, {}, store)
    data.unlink()
    assert str(tmp_path) not in json.dumps(prepared)
    restored = restore_remote_inputs(prepared, store, tmp_path / "worker")
    assert restored["dataset"]["source"] == cfg["dataset"]["source"]
    assert Path(restored["dataset"]["additional_sources"][0]["file_path"]).is_file()


def test_changed_local_file_cannot_reuse_old_reference(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n")
    cfg = config({"source_type": "local_file", "file_path": str(data)})
    store = LocalArtifactStore(tmp_path / "store")
    first = stage_remote_inputs(cfg, {}, store)
    data.write_text("a,b\n3,4\n")
    second = stage_remote_inputs(cfg, {}, store)
    assert first != second
    assert Path(restore_remote_inputs(first, store, tmp_path / "worker")
                ["dataset"]["source"]["file_path"]).read_text() == "a,b\n1,2\n"


@pytest.mark.parametrize("mutation", ["bytes", "traversal", "symlink", "version", "reference"])
def test_tampered_snapshot_fails_before_training(tmp_path, mutation):
    store = LocalArtifactStore(tmp_path / "store")
    prepared = stage_remote_inputs(config({"source_type": "upload", "dataset_id": "u"}),
                                   {"u": {"filename": "d.json", "content": b"[]"}}, store)
    bundle = next((tmp_path / "store").iterdir())
    manifest_path = bundle / "input.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "bytes":
        (bundle / "data.json").write_text("{}")
    elif mutation == "traversal":
        manifest["file"] = "../outside.json"
    elif mutation == "version":
        manifest["version"] = 999
    elif mutation == "reference":
        manifest["sha256"] = "a" * 64
    else:
        # A malicious store returning a symlink must be rejected (copytree's
        # default dereferences; use an injected pull preserving symlinks).
        import shutil
        (bundle / "data.json").unlink()
        secret = tmp_path / "secret"
        secret.write_text("[]")
        (bundle / "data.json").symlink_to(secret)
        store.pull_dir = lambda name, dest: shutil.copytree(bundle, dest, symlinks=True)
    manifest_bytes = json.dumps(manifest).encode()
    manifest_path.write_bytes(manifest_bytes)
    if mutation in {"traversal", "version"}:
        # Even an attacker able to replace both reference and manifest cannot
        # bypass the file/version checks.
        import hashlib
        name = "input-0-" + hashlib.sha256(manifest_bytes).hexdigest()
        bundle.rename(bundle.parent / name)
        prepared["dataset"]["source"]["file_path"] = "artifact://" + name
    with pytest.raises(RemoteInputError):
        restore_remote_inputs(prepared, store, tmp_path / "worker")


@pytest.mark.parametrize("source", [
    {"source_type": "local_file", "file_path": "/etc/shadow"},
    {"source_type": "local_file", "file_path": "artifact://../../escape"},
    {"source_type": "internal", "uri": "swl://foo@rev"},
])
def test_worker_rejects_unstaged_paths(tmp_path, source):
    with pytest.raises(RemoteInputError, match="staged"):
        restore_remote_inputs(config(source), LocalArtifactStore(tmp_path / "store"), tmp_path / "worker")


def test_failed_upload_does_not_return_portable_config(tmp_path):
    store = LocalArtifactStore(tmp_path / "store")
    def fail(*args):
        raise RuntimeError("provider key=SECRET /private/path")
    store.push_dir = fail
    with pytest.raises(RemoteInputError) as error:
        stage_remote_inputs(config({"source_type": "upload", "dataset_id": "u"}),
                            {"u": {"filename": "d.json", "content": b"[]"}}, store)
    assert "SECRET" not in str(error.value)
    assert "/private" not in str(error.value)


class InternalStore:
    def __init__(self, tmp_path, *, private=False, revision="rev"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from dataset_handlers.internal_store import Manifest
        self.shards = {}
        self.manifest = Manifest("demo", revision, "parquet", [], private=private)
        for index in range(2):
            path = tmp_path / f"part-{index}.parquet"
            pq.write_table(pa.table({"prompt": [str(index)], "chosen": ["ok"]}), path)
            uri = self.data_uri("demo", revision, path.name)
            self.manifest.data_files.append(uri)
            self.shards[uri] = path

    def get_manifest(self, name, revision):
        return self.manifest

    def data_uri(self, name, revision, filename):
        return f"s3://internal/datasets/{name}/{revision}/{filename}"

    def download_file(self, uri, destination):
        import shutil
        shutil.copyfile(self.shards[uri], destination)


def test_pinned_internal_shards_reconstruct_without_internal_credentials(tmp_path):
    import pyarrow.parquet as pq
    internal = InternalStore(tmp_path)
    store = LocalArtifactStore(tmp_path / "store")
    prepared = stage_remote_inputs(config({"source_type": "internal", "uri": "swl://demo@rev"}),
                                   {}, store, internal_store=internal)
    restored = restore_remote_inputs(prepared, store, tmp_path / "worker")
    assert pq.read_table(restored["dataset"]["source"]["file_path"]).to_pydict() == {
        "prompt": ["0", "1"], "chosen": ["ok", "ok"]}
    manifest = json.loads(next((tmp_path / "store").glob("*/input.json")).read_text())
    assert manifest["provenance"] == {"source_type": "internal", "uri": "swl://demo@rev"}
    assert "s3://internal" not in json.dumps(manifest)


@pytest.mark.parametrize("problem", ["private", "other_bucket", "wrong_revision"])
def test_private_or_untrusted_internal_manifest_never_exports(tmp_path, problem):
    internal = InternalStore(tmp_path, private=problem == "private",
                             revision="other" if problem == "wrong_revision" else "rev")
    if problem == "other_bucket":
        internal.manifest.data_files[0] = "s3://other/secrets.parquet"
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(RemoteInputError):
        stage_remote_inputs(config({"source_type": "internal", "uri": "swl://demo@rev"}),
                            {}, store, internal_store=internal)
    assert list((tmp_path / "store").iterdir()) == []


@pytest.mark.parametrize("uri", ["swl://demo", "swl://demo@latest", "swl://demo@../secret", "swl://../demo@rev", "swl://west1-incident@rev", "nope"])
def test_remote_plan_rejects_unpinned_or_invalid_internal_revision(uri):
    cfg = SimpleNamespace(dataset=SimpleNamespace(
        source=SimpleNamespace(source_type="internal", uri=uri), additional_sources=[]), training_mode="sft")
    assert check_remote_compatible(cfg)


@pytest.mark.parametrize("source", [
    {"source_type": "upload", "dataset_id": "id"},
    {"source_type": "local_file", "file_path": "/chosen/data.jsonl"},
    {"source_type": "internal", "uri": "swl://demo@rev"},
])
def test_remote_plan_accepts_stageable_sources(source):
    cfg = SimpleNamespace(dataset=SimpleNamespace(source=SimpleNamespace(**source), additional_sources=[]),
                          training_mode="sft")
    assert check_remote_compatible(cfg) == []


def test_worker_restores_before_training_and_cleans_local_files(tmp_path, monkeypatch):
    import sys
    from src.job_manager import JobManager
    from src.remote import worker_entry
    store = LocalArtifactStore(tmp_path / "store")
    prepared = stage_remote_inputs(config({"source_type": "upload", "dataset_id": "u"}),
                                   {"u": {"filename": "d.json", "content": b"[]"}}, store)
    prepared.update(output_name="model", multi_gpu_strategy="single")
    payload = {"job_id": "job", "config": prepared, "artifact_store": store.describe()}
    manager = JobManager(str(tmp_path / "jobs.db"))
    state = worker_entry.WorkerState("job", "train")
    seen = []
    def train(job_id, runtime, manager, *args):
        data = Path(runtime.dataset["source"]["file_path"])
        assert data.read_bytes() == b"[]"
        seen.append(data)
        directory = Path("models/model")
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}")
        manager.update_job(job_id, status="completed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker_entry, "_load_training_config", lambda cfg: SimpleNamespace(**cfg))
    monkeypatch.setitem(sys.modules, "src.training_runner", SimpleNamespace(
        _get_distributed_gpu_count=lambda cfg: 1, run_training_sync=train, run_training_distributed=train))
    worker_entry.run_train_stage(payload, manager, state)
    assert seen and not seen[0].exists()
    assert manager.get_job("job").config["dataset"]["source"]["file_path"].startswith("artifact://")
    assert store.exists("adapter")


def test_upload_train_and_explicit_eval_reach_real_training_pipeline(tmp_path, monkeypatch):
    """Use actual canonical config and loaders, including JSONL and column mapping."""
    from merlina import TrainingConfig
    from dataset_handlers import DatasetPipeline, get_formatter, create_loader_from_config
    # Keep Hugging Face's local dataset cache inside the test workspace.
    import datasets.config
    # Keep the real map/format implementation but disable process spawning;
    # sandboxed tests cannot create the multiprocessing manager's Unix socket.
    from datasets import Dataset
    original_map = Dataset.map
    def serial_map(self, *args, **kwargs):
        kwargs["num_proc"] = None
        return original_map(self, *args, **kwargs)
    monkeypatch.setattr(Dataset, "map", serial_map)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    rows = [{"prompt": f"question {i}", "chosen": f"answer {i}"} for i in range(3)]
    uploads = {
        "train": {"filename": "/private/train.jsonl", "content":
                  "\n".join(json.dumps(row) for row in rows).encode()},
        "eval": {"filename": "/private/eval.jsonl", "content":
                 b'{"question":"held out","response":"expected"}\n'},
    }
    original = TrainingConfig(
        base_model="org/model", output_name="model", training_mode="sft",
        dataset={"source": {"source_type": "upload", "dataset_id": "train"},
                 "eval_source": {"source_type": "upload", "dataset_id": "eval",
                                 "column_mapping": {"question": "prompt", "response": "chosen"}},
                 "format": {"format_type": "raw"}},
    )
    store = LocalArtifactStore(tmp_path / "store")
    prepared = stage_remote_inputs(original.model_dump(), uploads, store)
    assert "dataset_id" not in json.dumps(prepared["dataset"])
    uploads.clear()
    runtime = TrainingConfig.model_validate(restore_remote_inputs(prepared, store, tmp_path / "worker"))
    loader = create_loader_from_config(runtime.dataset.source, uploaded_datasets={})
    eval_loader = create_loader_from_config(runtime.dataset.eval_source, uploaded_datasets={})
    train, eval_data = DatasetPipeline(
        loader=loader, formatter=get_formatter("raw"), training_mode="sft",
        eval_loader=eval_loader, eval_column_mapping=runtime.dataset.eval_source.column_mapping,
        shuffle=False,
    ).prepare()
    assert len(train) == 3
    assert train[0]["prompt"] == "question 0"
    assert len(eval_data) == 1
    assert eval_data[0]["prompt"] == "held out"
    assert eval_data[0]["chosen"] == "expected"


@pytest.mark.parametrize("source", [
    {"source_type": "local_file", "file_path": "/server/private/eval.jsonl"},
    {"source_type": "upload", "dataset_id": "missing"},
    {"source_type": "internal", "uri": "swl://private/customer@rev"},
])
def test_worker_rejects_unstaged_explicit_eval_source(tmp_path, source):
    cfg = config({"source_type": "huggingface", "repo_id": "org/train"})
    cfg["dataset"]["eval_source"] = source
    with pytest.raises(RemoteInputError, match="staged"):
        restore_remote_inputs(cfg, LocalArtifactStore(tmp_path / "store"), tmp_path / "worker")


def test_eval_local_path_and_extraneous_hf_paths_cannot_reach_payload(tmp_path):
    local = tmp_path / "private.jsonl"
    local.write_text('{"prompt":"hi","chosen":"hello"}\n')
    cfg = config({"source_type": "huggingface", "repo_id": "org/train",
                  "file_path": "/server/train-secret", "uri": "secret-token", "dataset_id": "secret-id"})
    cfg["dataset"]["eval_source"] = {"source_type": "local_file", "file_path": str(local)}
    store = LocalArtifactStore(tmp_path / "store")
    prepared = stage_remote_inputs(cfg, {}, store)
    serialized = json.dumps(prepared)
    assert "secret" not in serialized
    assert str(tmp_path) not in serialized
    assert prepared["dataset"]["eval_source"]["file_path"].startswith("artifact://")


@pytest.mark.parametrize("uri", ["swl://private/customer@rev", "swl://west1-incident@rev", "swl://demo"])
def test_remote_plan_validates_explicit_eval_privacy_and_revision(uri):
    cfg = SimpleNamespace(dataset=SimpleNamespace(
        source=SimpleNamespace(source_type="huggingface"), additional_sources=[],
        eval_source=SimpleNamespace(source_type="internal", uri=uri)), training_mode="sft")
    assert check_remote_compatible(cfg)


def test_private_explicit_eval_never_uploads_to_remote_store(tmp_path):
    cfg = config({"source_type": "huggingface", "repo_id": "org/train"})
    cfg["dataset"]["eval_source"] = {"source_type": "internal", "uri": "swl://demo@rev"}
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(RemoteInputError, match="Private"):
        stage_remote_inputs(cfg, {}, store, internal_store=InternalStore(tmp_path, private=True))
    assert list((tmp_path / "store").iterdir()) == []
