"""Remote input handoff tests: snapshot, reconstruct, reject tampering and leaks."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.remote.artifacts import LocalArtifactStore
from src.remote.inputs import RemoteInputError, restore_remote_inputs, stage_remote_inputs
from src.remote.plan import check_remote_compatible


@pytest.fixture(autouse=True)
def operator_storage(monkeypatch, tmp_path):
    """LocalArtifactStore stands in for an operator-owned S3/R2 bucket here."""
    from config import settings
    from src.remote import inputs
    monkeypatch.setattr(inputs, "OPERATOR_STORE_KINDS", {"s3", "local"})
    monkeypatch.setattr(settings, "merlina_artifact_store", "s3")
    monkeypatch.setattr(settings, "remote_local_data_roots", str(tmp_path))


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
    {"source_type": "local_file", "file_path": "data.jsonl"},
    {"source_type": "internal", "uri": "swl://demo@rev"},
])
def test_remote_plan_accepts_stageable_sources(source, tmp_path):
    if "file_path" in source:
        (tmp_path / source["file_path"]).write_text("{}\n")
        source = {**source, "file_path": str(tmp_path / source["file_path"])}
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
        "train": {"filename": "/secret/train.jsonl", "content":
                  "\n".join(json.dumps(row) for row in rows).encode()},
        "eval": {"filename": "/secret/eval.jsonl", "content":
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


def _plan_config(source):
    return SimpleNamespace(dataset=SimpleNamespace(source=SimpleNamespace(**source), additional_sources=[]),
                           training_mode="sft")


def _s3_store():
    from src.remote.s3_artifacts import S3ArtifactStore
    from tests.test_s3_artifacts import FakeS3
    fs = FakeS3()
    return S3ArtifactStore("bucket", "job", fs=fs), fs


def test_local_file_outside_allowlist_is_refused_in_plan_and_staging(tmp_path):
    store, fs = _s3_store()
    hostname = Path("/etc/hostname")
    if not hostname.is_file():
        pytest.skip("no /etc/hostname on this system")
    # A symlink inside the allowlisted root resolves outside it.
    link = tmp_path / "data.json"
    link.symlink_to(hostname)
    for path in (hostname, link, tmp_path / ".." / tmp_path.name / ".." / "x.json"):
        source = {"source_type": "local_file", "file_path": str(path), "file_format": "json"}
        assert check_remote_compatible(_plan_config(source))
        with pytest.raises(RemoteInputError):
            stage_remote_inputs(config(source), {}, store)
    assert fs.objects == {}


def test_empty_allowlist_refuses_every_remote_local_file(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "remote_local_data_roots", "")
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    source = {"source_type": "local_file", "file_path": str(data)}
    assert any("REMOTE_LOCAL_DATA_ROOTS" in p for p in check_remote_compatible(_plan_config(source)))
    with pytest.raises(RemoteInputError, match="REMOTE_LOCAL_DATA_ROOTS"):
        stage_remote_inputs(config(source), {}, _s3_store()[0])


def test_private_named_local_file_and_upload_are_refused(tmp_path):
    store, fs = _s3_store()
    exports = tmp_path / "vps-incident-exports"
    exports.mkdir()
    data = exports / "logs.jsonl"
    data.write_text("{}\n")
    source = {"source_type": "local_file", "file_path": str(data)}
    assert any("Private" in p for p in check_remote_compatible(_plan_config(source)))
    with pytest.raises(RemoteInputError, match="Private"):
        stage_remote_inputs(config(source), {}, store)
    upload = {"source_type": "upload", "dataset_id": "u"}
    with pytest.raises(RemoteInputError, match="Private"):
        stage_remote_inputs(config(upload), {"u": {"filename": "west1-incident.jsonl", "content": b"[]"}}, store)
    assert check_remote_compatible(_plan_config({"source_type": "upload", "dataset_id": "incident_dump"}))
    assert fs.objects == {}


def test_allowlisted_local_file_stages_to_s3_store(tmp_path, monkeypatch):
    from config import settings
    from src.remote import inputs
    monkeypatch.setattr(inputs, "OPERATOR_STORE_KINDS", {"s3"})
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(settings, "remote_local_data_roots", f"/nonexistent-root,{root}")
    data = root / "train.jsonl"
    data.write_text('{"prompt":"hi","chosen":"hello"}\n')
    store, fs = _s3_store()
    source = {"source_type": "local_file", "file_path": str(data)}
    assert check_remote_compatible(_plan_config(source)) == []
    prepared = stage_remote_inputs(config(source), {}, store)
    restored = restore_remote_inputs(prepared, store, tmp_path / "worker")
    assert Path(restored["dataset"]["source"]["file_path"]).read_text() == data.read_text()


def test_non_s3_artifact_store_refuses_operator_data(tmp_path, monkeypatch):
    from config import settings
    from src.remote.artifacts import HFHubArtifactStore
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    store = HFHubArtifactStore("someone/merlina-run")
    store.push_dir = lambda *a: pytest.fail("operator data reached the HF store")
    sources = [{"source_type": "local_file", "file_path": str(data)},
               {"source_type": "upload", "dataset_id": "u"},
               {"source_type": "internal", "uri": "swl://demo@rev"}]
    monkeypatch.setattr(settings, "artifact_store", "hf_hub")
    for source in sources:
        assert any("operator-controlled S3" in p for p in check_remote_compatible(_plan_config(source)))
        with pytest.raises(RemoteInputError, match="operator-controlled S3"):
            stage_remote_inputs(config(source), {"u": {"filename": "d.json", "content": b"[]"}}, store,
                                internal_store=InternalStore(tmp_path))
    # HF sources need no operator store.
    assert check_remote_compatible(_plan_config({"source_type": "huggingface", "repo_id": "org/d"})) == []


def test_input_snapshot_manifests_omit_upload_ids(monkeypatch):
    from config import settings
    from src.remote.artifacts import store_for_job
    from src.remote.s3_artifacts import S3ArtifactStore
    from tests.test_s3_artifacts import FakeS3
    monkeypatch.setattr(settings, "merlina_artifact_s3_bucket", "bucket")
    upload_id = "upload_1760000000_0123456789abcdef0123456789abcdef"
    original = config({"source_type": "upload", "dataset_id": upload_id},
                      [{"source_type": "local_file", "file_path": "/srv/data/customer.json"}])
    store = store_for_job(SimpleNamespace(hf_token=None, model_dump=lambda: original), "job")
    fs = FakeS3()
    store._fs = fs
    stage_remote_inputs(config({"source_type": "upload", "dataset_id": upload_id}),
                        {upload_id: {"filename": "d.json", "content": b"[]"}}, store)
    manifests = [v for k, v in fs.objects.items() if k.endswith("/manifest.json")]
    assert manifests
    for raw in manifests:
        assert upload_id not in raw.decode()
        assert "customer" not in raw.decode()
    assert original["dataset"]["source"]["dataset_id"] == upload_id
