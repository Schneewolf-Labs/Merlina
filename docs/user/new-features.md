# New Features Guide

## 🎉 What's New in Merlina v1.1

Merlina now includes three major improvements that make training more reliable, transparent, and user-friendly!

### 1. Persistent Job Storage 💾

**What it does:** All training jobs are now saved to a SQLite database, so your job history persists across server restarts.

**Benefits:**
- Resume monitoring jobs after server restart
- View complete training history
- Track metrics over time
- Never lose job data

**API Endpoints:**

```bash
# Get job history with pagination
GET /jobs/history?limit=50&offset=0&status=completed

# Get detailed metrics for a specific job
GET /jobs/{job_id}/metrics

# Get database statistics
GET /stats
```

**Example Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_20251020_143022",
      "status": "completed",
      "progress": 1.0,
      "created_at": "2025-10-20T14:30:22",
      "updated_at": "2025-10-20T15:45:10",
      "output_dir": "./models/my_model",
      "error": null
    }
  ],
  "count": 1
}
```

### 2. Real-time WebSocket Updates 📡

**What it does:** Connect via WebSocket to receive live training updates without polling.

**Benefits:**
- Real-time loss curves
- Live GPU memory monitoring
- Instant error notifications
- Step-by-step progress tracking

**How to Connect:**

```javascript
// Connect to WebSocket for a specific job
const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'status_update':
      console.log(`Progress: ${data.progress * 100}%`);
      console.log(`Step: ${data.current_step}/${data.total_steps}`);
      console.log(`Loss: ${data.loss}`);
      console.log(`GPU Memory: ${data.gpu_memory}GB`);
      break;

    case 'metrics':
      console.log(`Metrics at step ${data.step}:`, data.metrics);
      break;

    case 'completed':
      console.log('Training completed!', data.output_dir);
      break;

    case 'error':
      console.error('Training failed:', data.error);
      break;
  }
};

// Send heartbeat to keep connection alive
setInterval(() => {
  ws.send(JSON.stringify({ type: 'heartbeat' }));
}, 30000);
```

**Message Types:**

| Type | When Sent | Fields |
|------|-----------|--------|
| `status_update` | Every training step | status, progress, current_step, total_steps, loss, eval_loss, learning_rate, gpu_memory |
| `metrics` | After each evaluation | step, metrics |
| `completed` | Training finishes | output_dir, final_metrics |
| `error` | Training fails | error |
| `heartbeat` | Client sends periodically | status: "ok" |

### 3. Pre-flight Validation ✅

**What it does:** Validates your configuration before starting training to catch issues early.

**Checks Performed:**

1. **GPU Availability**
   - CUDA available
   - Compute capability for Flash Attention
   - Number of GPUs

2. **VRAM Estimation**
   - Estimates memory needed for your model size
   - Checks available VRAM
   - Warns about quantization settings

3. **Disk Space**
   - Ensures enough space for checkpoints
   - Estimates based on model size

4. **Model Access**
   - Checks for gated models (Llama, Mixtral, etc.)
   - Validates HF token if needed

5. **Dataset Configuration**
   - Validates source configuration
   - Checks file existence for local datasets
   - Validates format settings

6. **Training Configuration**
   - LoRA parameter sanity checks
   - Batch size warnings
   - Learning rate recommendations
   - Sequence length validation

7. **API Tokens**
   - W&B token if logging enabled
   - HF token if push_to_hub enabled

8. **Dependencies**
   - Flash Attention availability
   - xformers availability

**API Endpoint:**

```bash
# Validate configuration before training
POST /validate
Content-Type: application/json

{
  "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "output_name": "my_model",
  "lora_r": 64,
  ...
}
```

**Example Response:**

```json
{
  "valid": true,
  "results": {
    "warnings": [
      "Flash Attention not installed but GPU supports it. Install with: pip install flash-attn --no-build-isolation",
      "Learning rate (0.0001) is high for fine-tuning. Consider using 1e-5 to 1e-4 for LoRA fine-tuning."
    ],
    "errors": [],
    "checks": {
      "GPU": {
        "status": "pass",
        "details": {
          "available": true,
          "count": 1,
          "devices": [
            {
              "id": 0,
              "name": "NVIDIA GeForce RTX 4090",
              "compute_capability": "8.9",
              "total_memory_gb": 24.0
            }
          ]
        }
      },
      "VRAM": {
        "status": "pass",
        "details": {
          "available_gb": 24.0,
          "estimated_required_gb": 10,
          "sufficient": true
        }
      },
      "Disk Space": {
        "status": "pass",
        "details": {
          "available_gb": 500.0,
          "estimated_required_gb": 20,
          "sufficient": true
        }
      }
    }
  }
}
```

**Automatic Validation:**

The `/train` endpoint now automatically runs validation before starting training. If validation fails, you'll get a detailed error response:

```json
{
  "detail": {
    "message": "Training configuration validation failed",
    "errors": [
      "Insufficient VRAM: Model requires ~16GB, but only 8GB available. Consider using a smaller model or reducing batch size/sequence length."
    ],
    "warnings": []
  }
}
```

## Usage Examples

### Example 1: Monitor Training with WebSocket

```python
import websockets
import asyncio
import json

async def monitor_training(job_id):
    uri = f"ws://localhost:8000/ws/{job_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Connected to job {job_id}")

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data['type'] == 'status_update':
                print(f"[{data['status']}] Progress: {data['progress']*100:.1f}%")
                if data.get('loss'):
                    print(f"  Loss: {data['loss']:.4f}")
                if data.get('learning_rate'):
                    print(f"  LR: {data['learning_rate']:.2e}")

            elif data['type'] == 'completed':
                print(f"✅ Training complete! Model saved to: {data['output_dir']}")
                break

            elif data['type'] == 'error':
                print(f"❌ Training failed: {data['error']}")
                break

# Run
asyncio.run(monitor_training("job_20251020_143022"))
```

### Example 2: Validate Before Training

```python
import requests

# Your training config
config = {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "output_name": "llama3-custom",
    "lora_r": 64,
    "lora_alpha": 128,
    "learning_rate": 5e-6,
    "num_epochs": 2,
    "use_4bit": True,
    # ... other config
}

# Validate first
response = requests.post("http://localhost:8000/validate", json=config)
result = response.json()

if result['valid']:
    print("✅ Configuration is valid!")

    # Show warnings if any
    if result['results']['warnings']:
        print("\n⚠️  Warnings:")
        for warning in result['results']['warnings']:
            print(f"  - {warning}")

    # Start training
    print("\n🚀 Starting training...")
    train_response = requests.post("http://localhost:8000/train", json=config)
    job_id = train_response.json()['job_id']
    print(f"Job ID: {job_id}")

else:
    print("❌ Configuration is invalid!")
    print("\nErrors:")
    for error in result['results']['errors']:
        print(f"  - {error}")
```

### Example 3: View Job History

```python
import requests
import pandas as pd

# Get all completed jobs
response = requests.get("http://localhost:8000/jobs/history?status=completed&limit=100")
data = response.json()

# Convert to DataFrame for analysis
df = pd.DataFrame(data['jobs'])
print(df[['job_id', 'created_at', 'output_dir']])

# Get metrics for a specific job
job_id = df.iloc[0]['job_id']
metrics_response = requests.get(f"http://localhost:8000/jobs/{job_id}/metrics")
metrics = metrics_response.json()['metrics']

# Plot loss curve
metrics_df = pd.DataFrame(metrics)
metrics_df.plot(x='step', y='loss', title='Training Loss')
```

## Migration Guide

### Existing Code

Your existing code will continue to work! The changes are backwards compatible.

**Old endpoints still work:**
- `POST /train` - Now with automatic validation
- `GET /status/{job_id}` - Returns same format
- `GET /jobs` - Returns same format

**New endpoints are additions:**
- `POST /validate` - New
- `GET /jobs/history` - New
- `GET /jobs/{job_id}/metrics` - New
- `WebSocket /ws/{job_id}` - New
- `GET /stats` - New

### Recommended Upgrades

1. **Add pre-flight validation** to your workflow:
   ```python
   # Before
   response = requests.post("/train", json=config)

   # After (recommended)
   validate = requests.post("/validate", json=config)
   if validate.json()['valid']:
       response = requests.post("/train", json=config)
   ```

2. **Switch from polling to WebSockets** for real-time updates:
   ```python
   # Before
   while True:
       status = requests.get(f"/status/{job_id}").json()
       print(status['progress'])
       time.sleep(10)

   # After (recommended)
   async with websockets.connect(f"ws://localhost:8000/ws/{job_id}") as ws:
       async for message in ws:
           data = json.loads(message)
           print(data['progress'])
   ```

3. **Use job history** to track experiments:
   ```python
   # Get all jobs
   history = requests.get("/jobs/history").json()

   # Find best performing job
   for job in history['jobs']:
       if job['status'] == 'completed':
           metrics = requests.get(f"/jobs/{job['job_id']}/metrics").json()
           # Analyze metrics
   ```

## Database Location

Jobs are stored in `./data/jobs.db` (SQLite database).

**Backup:**
```bash
cp ./data/jobs.db ./data/jobs_backup.db
```

**Reset:**
```bash
rm ./data/jobs.db  # Database will be recreated on next start
```

## Performance Notes

- **WebSocket Overhead:** Minimal (<1% GPU time)
- **Database Writes:** Async, doesn't block training
- **Validation Time:** <1 second for most configs

## Troubleshooting

**WebSocket won't connect:**
- Check firewall settings
- Ensure job_id exists
- Try `ws://` instead of `wss://` locally

**Validation false positives:**
- VRAM estimates are conservative
- You can still start training if you disagree with warnings

**Database locked error:**
- Only one Merlina instance per database
- Close other instances or use different `db_path`

## Next Steps

- Check out the [API Examples](../examples/) directory
- Read the [Developer Docs](../dev/) for implementation details
- Join our community for support!
