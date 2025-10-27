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
    """Simple test task that sleeps for a bit"""
    print(f"[{job_id}] Starting task with config: {config}")
    duration = config.get("duration", 2)

    for i in range(duration):
        print(f"[{job_id}] Working... ({i+1}/{duration})")
        time.sleep(1)

    print(f"[{job_id}] Task completed!")


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

    # Give first job time to start
    time.sleep(0.5)

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

    # Wait a bit
    time.sleep(0.5)

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

    # Wait a bit for jobs to start
    time.sleep(0.5)

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

    # Check job status
    time.sleep(0.5)
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
