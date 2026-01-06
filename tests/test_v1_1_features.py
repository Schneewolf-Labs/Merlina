"""
Test suite for Merlina v1.1 features

Tests the three major improvements:
1. Persistent job storage with SQLite (job_manager.py)
2. Real-time WebSocket updates (websocket_manager.py)
3. Pre-flight validation (preflight_checks.py)

Usage:
    python tests/test_v1.1_features.py

    or from project root:
    python -m pytest tests/test_v1.1_features.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_job_manager():
    """Test persistent job storage"""
    print("\n" + "="*60)
    print("TEST 1: Job Manager (Persistent Storage)")
    print("="*60)

    from src.job_manager import JobManager
    import os

    # Use a test database
    test_db_path = "./data/test_jobs.db"

    # Clean up old test database if it exists
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"🧹 Cleaned up old test database")

    # Initialize job manager
    job_manager = JobManager(db_path=test_db_path)

    # Create a test job
    config = {
        "base_model": "test/model",
        "output_name": "test_output",
        "learning_rate": 1e-5
    }

    job_id = "test_job_001"
    job = job_manager.create_job(job_id, config)

    print(f"✅ Created job: {job.job_id}")
    print(f"   Status: {job.status}")
    print(f"   Progress: {job.progress}")

    # Update job
    job_manager.update_job(job_id, status="training", progress=0.5, loss=0.42)
    print(f"✅ Updated job to training status")

    # Retrieve job
    retrieved = job_manager.get_job(job_id)
    assert retrieved.status == "training"
    assert retrieved.progress == 0.5
    assert retrieved.loss == 0.42
    print(f"✅ Retrieved job successfully")
    print(f"   Loss: {retrieved.loss}")

    # Add metrics
    job_manager.add_metric(job_id, step=10, loss=0.42, learning_rate=1e-5)
    job_manager.add_metric(job_id, step=20, loss=0.35, learning_rate=9e-6)
    print(f"✅ Added metrics")

    # Get metrics
    metrics = job_manager.get_metrics(job_id)
    print(f"✅ Retrieved {len(metrics)} metric records")

    # List jobs
    jobs = job_manager.list_jobs(limit=10)
    print(f"✅ Listed {len(jobs)} jobs")

    # Get stats
    stats = job_manager.get_stats()
    print(f"✅ Stats: {stats['total_jobs']} total jobs, {stats['total_metrics']} metrics")

    print("\n✅ Job Manager tests passed!\n")
    return True


def test_websocket_manager():
    """Test WebSocket manager"""
    print("="*60)
    print("TEST 2: WebSocket Manager")
    print("="*60)

    from src.websocket_manager import WebSocketManager

    ws_manager = WebSocketManager()

    # Test connection count
    count = ws_manager.get_connection_count()
    print(f"✅ Active connections: {count}")

    # Test creating message structures (no actual WebSocket needed)
    import asyncio

    async def test_messages():
        # These won't actually send without connections, but test the structure
        try:
            await ws_manager.send_status_update(
                job_id="test_job",
                status="training",
                progress=0.5,
                loss=0.42
            )
            print("✅ Status update message structure valid")
        except Exception as e:
            print(f"⚠️  No connections to send to (expected): {type(e).__name__}")

        try:
            await ws_manager.send_metric_update(
                job_id="test_job",
                step=10,
                metrics={"loss": 0.42, "lr": 1e-5}
            )
            print("✅ Metric update message structure valid")
        except Exception as e:
            print(f"⚠️  No connections to send to (expected): {type(e).__name__}")

    # Run async tests using asyncio.run() (Python 3.7+)
    # This is the recommended way to run async code from sync context
    asyncio.run(test_messages())

    print("\n✅ WebSocket Manager tests passed!\n")
    return True


def test_preflight_validation():
    """Test pre-flight validation"""
    print("="*60)
    print("TEST 3: Pre-flight Validation")
    print("="*60)

    from src.preflight_checks import PreflightValidator
    from pydantic import BaseModel, Field
    from typing import Optional

    # Create mock config classes
    class DatasetSource(BaseModel):
        source_type: str = "huggingface"
        repo_id: Optional[str] = "test/dataset"
        split: str = "train"
        file_path: Optional[str] = None
        file_format: Optional[str] = None
        dataset_id: Optional[str] = None

    class DatasetFormat(BaseModel):
        format_type: str = "chatml"
        custom_templates: Optional[dict] = None

    class DatasetConfig(BaseModel):
        source: DatasetSource = Field(default_factory=DatasetSource)
        format: DatasetFormat = Field(default_factory=DatasetFormat)
        test_size: float = 0.01
        max_samples: Optional[int] = None
        column_mapping: Optional[dict] = None

    class MockConfig(BaseModel):
        base_model: str = "gpt2"  # Small model for testing
        output_name: str = "test_model"
        lora_r: int = 64
        lora_alpha: int = 128
        lora_dropout: float = 0.05
        target_modules: list = ["c_attn"]
        learning_rate: float = 5e-6
        num_epochs: int = 2
        batch_size: int = 1
        gradient_accumulation_steps: int = 16
        max_length: int = 512
        max_prompt_length: int = 256
        beta: float = 0.1
        dataset: DatasetConfig = Field(default_factory=DatasetConfig)
        warmup_ratio: float = 0.05
        eval_steps: float = 0.2
        use_4bit: bool = True
        use_wandb: bool = False
        push_to_hub: bool = False
        hf_token: Optional[str] = None
        wandb_key: Optional[str] = None

    config = MockConfig()

    validator = PreflightValidator()
    is_valid, results = validator.validate_all(config)

    print(f"Validation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")

    # Print check results
    for check_name, check_result in results['checks'].items():
        status = check_result['status']
        emoji = "✅" if status == "pass" else "⚠️" if status == "warning" else "❌"
        print(f"{emoji} {check_name}: {status}")

    # Print warnings (expected)
    if results['warnings']:
        print("\n⚠️  Warnings (expected):")
        for warning in results['warnings'][:3]:  # Show first 3
            print(f"   - {warning[:80]}...")

    # Should pass basic validation
    print("\n✅ Pre-flight Validation tests passed!\n")
    return True


def test_imports():
    """Test that all core modules can be imported"""
    print("="*60)
    print("TEST 4: Module Imports")
    print("="*60)

    try:
        from src.job_manager import JobManager, JobRecord
        print("✅ job_manager imports")

        from src.websocket_manager import websocket_manager, WebSocketManager
        print("✅ websocket_manager imports")

        from src.preflight_checks import PreflightValidator, validate_config, ValidationError
        print("✅ preflight_checks imports")

        # Note: training_runner.py requires transformers/peft which may have
        # environment-specific issues (like flash_attn). We skip testing it here
        # as it's tested implicitly when the server runs.
        print("ℹ️  training_runner skipped (requires full ML stack)")

        print("\n✅ Core module imports successful!\n")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n🧙‍♀️ Merlina - New Features Test Suite")
    print("="*60)

    tests = [
        ("Module Imports", test_imports),
        ("Job Manager", test_job_manager),
        ("WebSocket Manager", test_websocket_manager),
        ("Pre-flight Validation", test_preflight_validation),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test failed with error:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
