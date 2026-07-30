#!/usr/bin/env python3
"""
Test script for job queue system

Tests:
1. Queue initialization
2. Job submission with different priorities
3. Queue position tracking
4. Job cancellation (queued and running)
5. Concurrent job execution
6. Queue statistics
"""

import sys
import time
import threading
from pathlib import Path

# Add merlina to path
sys.path.insert(0, str(Path(__file__).parent))

from src.job_queue import JobQueue, JobPriority
from src.job_manager import JobManager


def simple_task(job_id: str, config: dict):
    """Simple test task that sleeps for a bit."""
    print(f"[{job_id}] Starting task with config: {config}")
    duration = config.get("duration", 2)

    for i in range(duration):
        print(f"[{job_id}] Working... ({i+1}/{duration})")
        time.sleep(1)

    print(f"[{job_id}] Task completed!")


def wait_for_job_start(queue, max_wait: float = 2.0, poll_interval: float = 0.1) -> bool:
    """
    Wait for at least one job to start running.

    Args:
        queue: JobQueue instance
        max_wait: Maximum time to wait in seconds
        poll_interval: Time between checks in seconds

    Returns:
        True if a job started, False if timeout reached
    """
    elapsed = 0.0
    while elapsed < max_wait:
        running = queue.list_running_jobs()
        if running:
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    return False


def test_basic_queue():
    """Test basic queue functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic Queue Functionality")
    print("="*60)

    # Create queue with 1 worker
    queue = JobQueue(max_concurrent_jobs=1)

    # Submit jobs
    print("\nSubmitting 3 jobs...")
    pos1 = queue.submit("job1", {"duration": 2}, simple_task, JobPriority.NORMAL)
    pos2 = queue.submit("job2", {"duration": 2}, simple_task, JobPriority.NORMAL)
    pos3 = queue.submit("job3", {"duration": 2}, simple_task, JobPriority.NORMAL)

    print(f"Job 1 position: {pos1}")
    print(f"Job 2 position: {pos2}")
    print(f"Job 3 position: {pos3}")

    # Check queue stats
    stats = queue.get_queue_stats()
    print(f"\nQueue stats: {stats}")

    # Wait for completion
    print("\nWaiting for all jobs to complete...")
    queue.wait_for_completion()

    print("\nAll jobs completed!")
    queue.shutdown(wait=True)
    print("✓ Test 1 passed!")


def test_priority_queue():
    """Test priority-based job ordering"""
    print("\n" + "="*60)
    print("TEST 2: Priority Queue")
    print("="*60)

    queue = JobQueue(max_concurrent_jobs=1)

    # Submit jobs with different priorities
    print("\nSubmitting jobs with different priorities...")
    queue.submit("low-1", {"duration": 1}, simple_task, JobPriority.LOW)
    queue.submit("normal-1", {"duration": 1}, simple_task, JobPriority.NORMAL)
    queue.submit("high-1", {"duration": 1}, simple_task, JobPriority.HIGH)
    queue.submit("low-2", {"duration": 1}, simple_task, JobPriority.LOW)

    # Wait for first job to start (with timeout instead of fixed sleep)
    wait_for_job_start(queue)

    # Check queued jobs order
    queued = queue.list_queued_jobs()
    print("\nQueued jobs (should be ordered by priority):")
    for job in queued:
        print(f"  Position {job['position']}: {job['job_id']} (priority: {job['priority']})")

    # Wait for completion
    queue.wait_for_completion()
    queue.shutdown(wait=True)
    print("\n✓ Test 2 passed!")


def test_job_cancellation():
    """Test job cancellation"""
    print("\n" + "="*60)
    print("TEST 3: Job Cancellation")
    print("="*60)

    queue = JobQueue(max_concurrent_jobs=1)

    # Submit jobs
    print("\nSubmitting jobs...")
    queue.submit("cancel-1", {"duration": 3}, simple_task, JobPriority.NORMAL)
    queue.submit("cancel-2", {"duration": 3}, simple_task, JobPriority.NORMAL)
    queue.submit("cancel-3", {"duration": 3}, simple_task, JobPriority.NORMAL)

    # Wait for first job to start
    wait_for_job_start(queue)

    # Cancel queued job
    print("\nCancelling cancel-2 (should be queued)...")
    success = queue.cancel("cancel-2")
    print(f"Cancellation result: {success}")

    # Check queue
    queued = queue.list_queued_jobs()
    print(f"\nQueued jobs after cancellation: {[j['job_id'] for j in queued]}")

    # Wait for completion
    queue.wait_for_completion()
    queue.shutdown(wait=True)
    print("\n✓ Test 3 passed!")


def test_concurrent_execution():
    """Test concurrent job execution"""
    print("\n" + "="*60)
    print("TEST 4: Concurrent Execution (2 workers)")
    print("="*60)

    queue = JobQueue(max_concurrent_jobs=2)

    # Submit jobs
    print("\nSubmitting 4 jobs (2 should run concurrently)...")
    for i in range(4):
        queue.submit(f"concurrent-{i+1}", {"duration": 2}, simple_task, JobPriority.NORMAL)

    # Wait for jobs to start
    wait_for_job_start(queue)

    # Check running jobs
    running = queue.list_running_jobs()
    print(f"\nCurrently running: {len(running)} jobs")
    for job in running:
        print(f"  - {job['job_id']} on {job['worker']}")

    # Wait for completion
    queue.wait_for_completion()
    queue.shutdown(wait=True)
    print("\n✓ Test 4 passed!")


def test_with_job_manager():
    """Test integration with JobManager"""
    print("\n" + "="*60)
    print("TEST 5: Integration with JobManager")
    print("="*60)

    # Create temp database
    db_path = "/tmp/test_queue_jobs.db"
    job_manager = JobManager(db_path=db_path)

    # Create queue with job manager
    queue = JobQueue(max_concurrent_jobs=1, job_manager=job_manager)

    # Submit job
    print("\nSubmitting job with job manager integration...")
    job_id = "test-job-1"
    job_manager.create_job(job_id, {"test": "config"})
    queue.submit(job_id, {"duration": 2}, simple_task, JobPriority.NORMAL)

    # Wait for job to start running
    wait_for_job_start(queue)
    job = job_manager.get_job(job_id)
    print(f"\nJob status from database: {job.status}")

    # Wait for completion
    queue.wait_for_completion()

    # Check final status
    job = job_manager.get_job(job_id)
    print(f"Final job status: {job.status}")

    queue.shutdown(wait=True)

    # Cleanup
    import os
    os.remove(db_path)

    print("\n✓ Test 5 passed!")


def test_wait_for_completion_timeout():
    """Test that wait_for_completion honors its timeout"""
    print("\n" + "="*60)
    print("TEST 6: wait_for_completion timeout")
    print("="*60)

    queue = JobQueue(max_concurrent_jobs=1)

    # A job longer than the timeout: wait should give up and return False
    queue.submit("slow-job", {"duration": 3}, simple_task, JobPriority.NORMAL)

    start = time.monotonic()
    finished = queue.wait_for_completion(timeout=0.5)
    elapsed = time.monotonic() - start
    print(f"\nwait_for_completion(timeout=0.5) -> {finished} after {elapsed:.2f}s")
    assert finished is False, "expected timeout before the job finished"
    assert elapsed < 2.5, f"timeout did not fire promptly (took {elapsed:.2f}s)"

    # With a generous timeout the job finishes and we get True
    finished = queue.wait_for_completion(timeout=30)
    print(f"wait_for_completion(timeout=30) -> {finished}")
    assert finished is True, "expected completion within the timeout"

    queue.shutdown(wait=True)
    print("\n✓ Test 6 passed!")


def test_submit_rollback_on_db_failure():
    """Test that a failed DB write during submit leaves no stale queue state"""
    print("\n" + "="*60)
    print("TEST 7: submit() rollback on DB failure")
    print("="*60)

    class FailingJobManager:
        def update_job(self, job_id, **kwargs):
            raise RuntimeError("database is locked")

    queue = JobQueue(max_concurrent_jobs=1, job_manager=FailingJobManager())

    raised = False
    try:
        queue.submit("doomed-job", {"duration": 1}, simple_task, JobPriority.NORMAL)
    except RuntimeError:
        raised = True
    assert raised, "expected submit() to propagate the DB error"

    # No stale tracking entry: the job must not report as queued
    status = queue.get_status("doomed-job")
    print(f"Status after failed submit: {status}")
    assert status["state"] == "unknown", f"stale queue state: {status}"
    assert queue.get_queue_stats()["queued"] == 0

    queue._shutdown = True
    print("\n✓ Test 7 passed!")


def test_no_stale_queued_state_after_run():
    """Test that a job picked up instantly by a worker leaves no 'queued' ghost"""
    print("\n" + "="*60)
    print("TEST 8: no stale queued state after immediate pickup")
    print("="*60)

    queue = JobQueue(max_concurrent_jobs=1)

    # Idle worker: the job is grabbed the moment it hits the queue, which
    # previously could race submit()'s own bookkeeping.
    queue.submit("instant-job", {"duration": 1}, simple_task, JobPriority.NORMAL)
    queue.wait_for_completion(timeout=30)

    status = queue.get_status("instant-job")
    print(f"Status after completion: {status}")
    assert status["state"] == "unknown", f"job still tracked after completion: {status}"
    assert queue.get_queue_stats()["queued"] == 0

    queue.shutdown(wait=True)
    print("\n✓ Test 8 passed!")


def _blocking_pair():
    """Build a (release_event, task, executed_list) trio for occupancy tests.

    The task records every job it runs and blocks until the event is set, so a
    single-worker queue can be pinned while other jobs pile up behind it.
    """
    release = threading.Event()
    executed = []
    lock = threading.Lock()

    def task(job_id, config):
        with lock:
            executed.append(job_id)
        release.wait(timeout=10)

    return release, task, executed


def test_deleted_job_is_not_executed():
    """Test that a job whose record was deleted never runs"""
    print("\n" + "="*60)
    print("TEST 9: deleted job is not executed")
    print("="*60)

    import os
    db_path = "/tmp/test_queue_deleted_job.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    job_manager = JobManager(db_path=db_path)
    queue = JobQueue(max_concurrent_jobs=1, job_manager=job_manager)

    release, task, executed = _blocking_pair()

    # Pin the only worker so the second job is guaranteed to sit in the queue
    job_manager.create_job("blocker", {"test": "config"})
    queue.submit("blocker", {}, task, JobPriority.NORMAL)
    assert wait_for_job_start(queue), "blocker job never started"

    job_manager.create_job("doomed", {"test": "config"})
    queue.submit("doomed", {}, task, JobPriority.NORMAL)

    # Job record deleted while the job is still waiting in the queue
    assert job_manager.delete_job("doomed"), "expected the job record to be deleted"

    release.set()
    assert queue.wait_for_completion(timeout=30), "queue did not drain"

    print(f"Jobs executed: {executed}")
    assert "doomed" not in executed, "deleted job was executed"
    assert "blocker" in executed, "blocker job should still have run"

    queue.shutdown(wait=True)
    os.remove(db_path)
    print("\n✓ Test 9 passed!")


def test_remove_clears_queued_entry_and_config():
    """Test that remove() drops a queued job's entry and releases its config"""
    print("\n" + "="*60)
    print("TEST 10: remove() clears queue entry and config")
    print("="*60)

    queue = JobQueue(max_concurrent_jobs=1)

    release, task, executed = _blocking_pair()

    queue.submit("blocker", {}, task, JobPriority.NORMAL)
    assert wait_for_job_start(queue), "blocker job never started"

    queue.submit("deleted", {"hf_token": "secret"}, task, JobPriority.NORMAL)
    tracked = queue._queued_jobs["deleted"]

    assert queue.remove("deleted") is True, "expected the queued job to be removed"
    assert queue.remove("deleted") is False, "removing twice should report nothing to remove"

    status = queue.get_status("deleted")
    print(f"Status after removal: {status}")
    assert status["state"] == "unknown", f"job still tracked after removal: {status}"
    assert queue.get_queue_stats()["queued"] == 0
    assert [j["job_id"] for j in queue.list_queued_jobs()] == []

    # The config (which can hold API tokens) must not be kept alive by the
    # stale PriorityQueue item
    assert tracked.config is None, "config still referenced after removal"
    assert tracked.callback is None, "callback still referenced after removal"

    release.set()
    assert queue.wait_for_completion(timeout=30), "queue did not drain"
    print(f"Jobs executed: {executed}")
    assert "deleted" not in executed, "removed job was executed"

    queue.shutdown(wait=True)
    print("\n✓ Test 10 passed!")


def test_remove_all_queued():
    """Test that remove_all_queued() empties the queue without touching runners"""
    print("\n" + "="*60)
    print("TEST 11: remove_all_queued()")
    print("="*60)

    queue = JobQueue(max_concurrent_jobs=1)

    release, task, executed = _blocking_pair()

    queue.submit("blocker", {}, task, JobPriority.NORMAL)
    assert wait_for_job_start(queue), "blocker job never started"

    for i in range(3):
        queue.submit(f"pending-{i}", {"duration": 1}, task, JobPriority.NORMAL)

    removed = queue.remove_all_queued()
    print(f"Removed {removed} queued job(s)")
    assert removed == 3, f"expected 3 removals, got {removed}"
    assert queue.get_queue_stats()["queued"] == 0
    assert queue.get_queue_stats()["running"] == 1, "running job should be untouched"

    release.set()
    assert queue.wait_for_completion(timeout=30), "queue did not drain"

    print(f"Jobs executed: {executed}")
    assert executed == ["blocker"], f"removed jobs were executed: {executed}"

    queue.shutdown(wait=True)
    print("\n✓ Test 11 passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("MERLINA JOB QUEUE TEST SUITE")
    print("🧪 "*20)

    try:
        test_basic_queue()
        test_priority_queue()
        test_job_cancellation()
        test_concurrent_execution()
        test_with_job_manager()
        test_wait_for_completion_timeout()
        test_submit_rollback_on_db_failure()
        test_no_stale_queued_state_after_run()
        test_deleted_job_is_not_executed()
        test_remove_clears_queued_entry_and_config()
        test_remove_all_queued()

        print("\n" + "✅ "*20)
        print("ALL TESTS PASSED!")
        print("✅ "*20 + "\n")

    except Exception as e:
        print("\n" + "❌ "*20)
        print(f"TEST FAILED: {e}")
        print("❌ "*20 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
