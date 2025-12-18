"""
Example: Monitor training with WebSocket
Demonstrates real-time training updates via WebSocket connection
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import websockets
import json
from datetime import datetime


async def monitor_training(job_id, api_url="ws://localhost:8000"):
    """
    Monitor training job via WebSocket.

    Args:
        job_id: Job identifier to monitor
        api_url: WebSocket API URL
    """
    uri = f"{api_url}/ws/{job_id}"

    print(f"🔌 Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to job {job_id}")
            print("📡 Listening for updates...\n")
            print("=" * 80)

            # Send heartbeat periodically
            async def send_heartbeat():
                while True:
                    await asyncio.sleep(30)
                    try:
                        await websocket.send(json.dumps({"type": "heartbeat"}))
                    except (websockets.exceptions.WebSocketException, ConnectionError):
                        break

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(send_heartbeat())

            # Listen for messages
            try:
                async for message in websocket:
                    data = json.loads(message)
                    handle_message(data)

                    # Stop if training completed or failed
                    if data.get('type') in ['completed', 'error']:
                        break

            finally:
                heartbeat_task.cancel()

    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")


def handle_message(data):
    """Handle incoming WebSocket message"""
    msg_type = data.get('type')
    timestamp = datetime.now().strftime("%H:%M:%S")

    if msg_type == 'status_update':
        handle_status_update(data, timestamp)

    elif msg_type == 'metrics':
        handle_metrics(data, timestamp)

    elif msg_type == 'completed':
        handle_completion(data, timestamp)

    elif msg_type == 'error':
        handle_error(data, timestamp)

    elif msg_type == 'heartbeat':
        # Heartbeat ack - don't print
        pass

    else:
        print(f"[{timestamp}] Unknown message type: {msg_type}")


def handle_status_update(data, timestamp):
    """Handle status update message"""
    status = data.get('status', 'unknown')
    progress = data.get('progress', 0) * 100

    # Build status line
    line = f"[{timestamp}] [{status.upper()}] Progress: {progress:.1f}%"

    # Add step info if available
    current_step = data.get('current_step')
    total_steps = data.get('total_steps')
    if current_step is not None and total_steps is not None:
        line += f" | Step: {current_step}/{total_steps}"

    # Add loss if available
    loss = data.get('loss')
    if loss is not None:
        line += f" | Loss: {loss:.4f}"

    # Add eval loss if available
    eval_loss = data.get('eval_loss')
    if eval_loss is not None:
        line += f" | Eval Loss: {eval_loss:.4f}"

    # Add learning rate if available
    lr = data.get('learning_rate')
    if lr is not None:
        line += f" | LR: {lr:.2e}"

    # Add GPU memory if available
    gpu_memory = data.get('gpu_memory')
    if gpu_memory is not None:
        line += f" | GPU: {gpu_memory:.1f}GB"

    print(line)


def handle_metrics(data, timestamp):
    """Handle metrics message"""
    step = data.get('step')
    metrics = data.get('metrics', {})

    print(f"\n[{timestamp}] 📊 Metrics at step {step}:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()


def handle_completion(data, timestamp):
    """Handle completion message"""
    output_dir = data.get('output_dir', 'unknown')
    final_metrics = data.get('final_metrics', {})

    print("\n" + "=" * 80)
    print(f"[{timestamp}] ✅ TRAINING COMPLETED!")
    print(f"📁 Model saved to: {output_dir}")

    if final_metrics:
        print("\n📊 Final Metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")

    print("=" * 80 + "\n")


def handle_error(data, timestamp):
    """Handle error message"""
    error = data.get('error', 'Unknown error')

    print("\n" + "=" * 80)
    print(f"[{timestamp}] ❌ TRAINING FAILED!")
    print(f"Error: {error}")
    print("=" * 80 + "\n")


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python websocket_monitor.py <job_id>")
        print("\nExample:")
        print("  python websocket_monitor.py job_20251020_143022")
        sys.exit(1)

    job_id = sys.argv[1]

    print("🧙‍♀️ Merlina - WebSocket Training Monitor")
    print("=" * 80)
    print(f"Job ID: {job_id}")
    print("Press Ctrl+C to stop monitoring\n")

    # Run async monitor
    asyncio.run(monitor_training(job_id))


if __name__ == "__main__":
    main()
