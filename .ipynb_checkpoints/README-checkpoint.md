# 🧙‍♀️ Merlina - Magical Model Training

Train LLMs with ORPO using a delightful web interface powered by magic ✨

![Merlina Banner](https://img.shields.io/badge/Merlina-Magical%20Training-c042ff?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==)

## Features

- 🎨 **Beautiful Web Interface** - Cute wizard-themed UI with animations
- 🚀 **ORPO Training** - State-of-the-art preference optimization
- 🗜️ **4-bit Quantization** - Train large models on consumer GPUs
- 📊 **Real-time Monitoring** - Track progress, loss, and metrics
- 🤗 **HuggingFace Integration** - Push models directly to the Hub
- 📈 **W&B Logging** - Detailed experiment tracking

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository (or download the files)
git clone https://github.com/schneewolflabs/merlina.git
cd merlina

# Install dependencies
pip install -r requirements.txt
```

### 2. Directory Structure

```
merlina/
├── merlina.py           # Backend server
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── frontend/           # Web interface
    ├── index.html
    ├── styles.css
    └── script.js
```

### 3. Run Merlina

```bash
python merlina.py
```

Visit http://localhost:8000 and start training! 🎉

## Requirements

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
transformers>=4.36.0
trl>=0.7.0
datasets>=2.14.0
peft>=0.6.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
torch>=2.0.0
wandb>=0.15.0
huggingface-hub>=0.19.0
```

## Configuration

Merlina uses the `schneewolflabs/Athanor-DPO` dataset by default. The interface lets you configure:

- **Base Model** - Choose from popular models or enter custom
- **LoRA Settings** - Rank, alpha, dropout
- **Training Parameters** - Learning rate, epochs, batch size
- **ORPO Beta** - Control preference optimization strength
- **Options** - 4-bit quantization, W&B logging, HF Hub upload

## API Endpoints

- `GET /` - Web interface
- `POST /train` - Start training job
- `GET /status/{job_id}` - Get job status
- `GET /jobs` - List all jobs
- `GET /api/docs` - OpenAPI documentation

## GPU Memory Requirements

With 4-bit quantization enabled (default):

| Model Size | VRAM Required |
|------------|---------------|
| 3B params  | ~6 GB         |
| 7B params  | ~10 GB        |
| 13B params | ~16 GB        |

## Environment Variables

Optional environment variables:

```bash
export WANDB_API_KEY=your_wandb_key
export HF_TOKEN=your_huggingface_token
export CUDA_VISIBLE_DEVICES=0  # Select GPU
```

## Docker Support

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.10 python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python3", "merlina.py"]
```

Build and run:
```bash
docker build -t merlina .
docker run --gpus all -p 8000:8000 merlina
```

## Tips

- 🎯 Start with 1-2 epochs for testing
- 💾 Models are saved to `./models/{output_name}`
- 📊 Enable W&B for detailed metrics
- 🔧 Adjust LoRA rank based on your GPU memory
- ✨ Try the Konami code in the UI for a surprise!

## Credits

Created with 💜 by Schneewolf Labs

## License

MIT License - feel free to use for your magical model training!