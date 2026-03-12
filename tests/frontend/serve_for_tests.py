#!/usr/bin/env python3
"""
Lightweight Merlina server for frontend integration tests.

Mocks GPU-dependent imports (torch, transformers, etc.) so the FastAPI
server can start on CI runners without ML dependencies. The API endpoints
respond normally — only actual training is unavailable.

Usage:
    python tests/frontend/serve_for_tests.py
"""

import sys
from unittest.mock import MagicMock, Mock
from pathlib import Path

# ── Mock ML dependencies before any imports ──────────────────────────────────

class FakeTensor:
    pass

class FakeModule:
    pass

class FakeTokenizerBase:
    pass

mock_torch = MagicMock()
mock_torch.__spec__ = MagicMock()
mock_torch.Tensor = FakeTensor
mock_torch.nn = MagicMock()
mock_torch.nn.Module = FakeModule
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.device_count.return_value = 0
mock_torch.cuda.empty_cache = Mock()
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch.cuda
sys.modules['torch.nn'] = mock_torch.nn

mock_transformers = MagicMock()
mock_transformers.PreTrainedTokenizerBase = FakeTokenizerBase
sys.modules['transformers'] = mock_transformers

for module in ['trl', 'peft', 'accelerate', 'bitsandbytes', 'wandb', 'psutil', 'pynvml']:
    sys.modules[module] = MagicMock()

# ── Start the actual server ──────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from merlina import app  # noqa: E402

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, log_level='warning')
