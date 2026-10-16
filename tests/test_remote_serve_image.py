"""
Guards for Dockerfile.serve, the slim vLLM-based serve worker image.

The image copies an explicit list of Merlina modules instead of the whole
repo, so a new import in the serve worker would only fail on a rented GPU.
These tests import the serve worker in a clean interpreter, collect every
repo module it loads, and check each one is copied — and that the training
stack never comes along.

Run standalone: python -m pytest tests/test_remote_serve_image.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCKERFILE = ROOT / "Dockerfile.serve"

# Everything the serve worker can touch on the instance, including the S3
# artifact store (s3fs + config) used to pull training-job artifacts.
_PROBE = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import src.remote.worker_entry, src.remote.serve_worker
import src.remote.s3_artifacts, src.remote.artifact_capabilities, config
root = os.getcwd()
mods = sorted(os.path.relpath(m.__file__, root) for m in list(sys.modules.values())
              if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(root + os.sep))
heavy = [m for m in ("torch", "transformers", "peft", "datasets", "grimoire", "vllm") if m in sys.modules]
print(json.dumps({"modules": mods, "heavy": heavy}))
"""


def _probe():
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _copied_files():
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    copied = set()
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != "COPY" or any(p.startswith("--from") for p in parts):
            continue
        *sources, dest = parts[1:]
        for src in sources:
            copied.add(src)
    return copied


def test_serve_worker_import_surface_is_copied():
    probe = _probe()
    missing = [m for m in probe["modules"] if m not in _copied_files()]
    assert not missing, f"Dockerfile.serve does not COPY modules the serve worker imports: {missing}"


def test_serve_worker_never_imports_the_training_stack():
    assert _probe()["heavy"] == []


def test_base_image_is_an_exact_vllm_release():
    froms = re.findall(r"^FROM\s+(\S+)", DOCKERFILE.read_text(), re.M)
    assert len(froms) == 1
    assert re.fullmatch(r"vllm/vllm-openai:v\d+\.\d+\.\d+", froms[0]), froms[0]


def test_entrypoint_runs_the_worker():
    text = DOCKERFILE.read_text()
    assert "ENTRYPOINT []" in text
    assert 'CMD ["python3", "-m", "src.remote.worker_entry"]' in text
