"""
Example: View and analyze job history
Demonstrates persistent job storage and metrics retrieval
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from datetime import datetime
from typing import List, Dict, Any


API_URL = "http://localhost:8000"


def get_job_history(status: str = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get job history from database.

    Args:
        status: Filter by status (completed, failed, training, etc.)
        limit: Maximum number of jobs to retrieve

    Returns:
        List of job records
    """
    params = {"limit": limit}
    if status:
        params["status"] = status

    response = requests.get(f"{API_URL}/jobs/history", params=params)

    if response.status_code != 200:
        print(f"❌ Failed to get job history: {response.status_code}")
        return []

    data = response.json()
    return data.get('jobs', [])


def get_job_metrics(job_id: str) -> List[Dict[str, Any]]:
    """
    Get detailed metrics for a job.

    Args:
        job_id: Job identifier

    Returns:
        List of metric records
    """
    response = requests.get(f"{API_URL}/jobs/{job_id}/metrics")

    if response.status_code != 200:
        print(f"❌ Failed to get metrics for {job_id}: {response.status_code}")
        return []

    data = response.json()
    return data.get('metrics', [])


def get_stats() -> Dict[str, Any]:
    """Get database and system statistics"""
    response = requests.get(f"{API_URL}/stats")

    if response.status_code != 200:
        print(f"❌ Failed to get stats: {response.status_code}")
        return {}

    return response.json()


def format_datetime(iso_string: str) -> str:
    """Format ISO datetime string for display"""
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return iso_string


def print_job_summary(jobs: List[Dict[str, Any]]):
    """Print formatted job summary table"""
    if not jobs:
        print("No jobs found.")
        return

    print("\n📋 Job History")
    print("=" * 120)
    print(f"{'Job ID':<25} {'Status':<12} {'Progress':<10} {'Created':<20} {'Output':<30}")
    print("-" * 120)

    for job in jobs:
        job_id = job['job_id']
        status = job['status']
        progress = f"{job['progress']*100:.1f}%"
        created = format_datetime(job['created_at'])
        output = job.get('output_dir', '-')[:28]

        # Color code status
        status_emoji = {
            'completed': '✅',
            'failed': '❌',
            'training': '🔄',
            'started': '🚀',
            'initializing': '⚙️'
        }.get(status, '📝')

        print(f"{job_id:<25} {status_emoji} {status:<10} {progress:<10} {created:<20} {output:<30}")

    print("=" * 120 + "\n")


def print_job_details(job_id: str):
    """Print detailed information about a specific job"""
    # Get job from history
    jobs = get_job_history(limit=1000)  # Get all to find specific job
    job = next((j for j in jobs if j['job_id'] == job_id), None)

    if not job:
        print(f"❌ Job {job_id} not found")
        return

    print(f"\n🔍 Job Details: {job_id}")
    print("=" * 80)
    print(f"Status:      {job['status']}")
    print(f"Progress:    {job['progress']*100:.1f}%")
    print(f"Created:     {format_datetime(job['created_at'])}")
    print(f"Updated:     {format_datetime(job['updated_at'])}")

    if job.get('output_dir'):
        print(f"Output:      {job['output_dir']}")

    if job.get('error'):
        print(f"Error:       {job['error']}")

    # Get metrics
    metrics = get_job_metrics(job_id)

    if metrics:
        print(f"\n📊 Training Metrics ({len(metrics)} records)")
        print("-" * 80)

        # Show first, middle, and last metrics
        key_metrics = []
        if len(metrics) > 0:
            key_metrics.append(("First", metrics[0]))
        if len(metrics) > 2:
            key_metrics.append(("Middle", metrics[len(metrics)//2]))
        if len(metrics) > 1:
            key_metrics.append(("Last", metrics[-1]))

        for label, metric in key_metrics:
            step = metric.get('step', 0)
            loss = metric.get('loss', 0)
            eval_loss = metric.get('eval_loss')
            lr = metric.get('learning_rate')
            gpu = metric.get('gpu_memory_used')

            line = f"  {label:8} - Step {step:4}"
            if loss:
                line += f" | Loss: {loss:.4f}"
            if eval_loss:
                line += f" | Eval: {eval_loss:.4f}"
            if lr:
                line += f" | LR: {lr:.2e}"
            if gpu:
                line += f" | GPU: {gpu:.1f}GB"

            print(line)

        # Show loss improvement
        if len(metrics) >= 2:
            first_loss = metrics[0].get('loss')
            last_loss = metrics[-1].get('loss')
            if first_loss and last_loss:
                improvement = ((first_loss - last_loss) / first_loss) * 100
                print(f"\n  📈 Loss Improvement: {improvement:.1f}%")

    print("=" * 80 + "\n")


def print_statistics():
    """Print overall statistics"""
    stats = get_stats()

    if not stats:
        return

    db_stats = stats.get('database', {})
    ws_stats = stats.get('websockets', {})

    print("\n📊 System Statistics")
    print("=" * 60)

    print("\nDatabase:")
    print(f"  Total Jobs: {db_stats.get('total_jobs', 0)}")
    print(f"  Total Metrics: {db_stats.get('total_metrics', 0)}")

    by_status = db_stats.get('by_status', {})
    if by_status:
        print("\nJobs by Status:")
        for status, count in by_status.items():
            print(f"  {status}: {count}")

    print("\nWebSockets:")
    print(f"  Active Connections: {ws_stats.get('total_connections', 0)}")

    print("=" * 60 + "\n")


def main():
    """Main execution"""
    import sys

    print("🧙‍♀️ Merlina - Job History Viewer")
    print("=" * 80)

    if len(sys.argv) > 1:
        # Show specific job details
        job_id = sys.argv[1]
        print_job_details(job_id)
    else:
        # Show overall statistics
        print_statistics()

        # Show all completed jobs
        print("\n✅ Completed Jobs:")
        completed = get_job_history(status="completed", limit=20)
        print_job_summary(completed)

        # Show failed jobs
        print("\n❌ Failed Jobs:")
        failed = get_job_history(status="failed", limit=10)
        print_job_summary(failed)

        # Show recent jobs
        print("\n📝 Recent Jobs:")
        recent = get_job_history(limit=10)
        print_job_summary(recent)

        print("\nUsage:")
        print("  python job_history.py              # Show overview")
        print("  python job_history.py <job_id>     # Show job details")


if __name__ == "__main__":
    main()
