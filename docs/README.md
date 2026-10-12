# Merlina Documentation

Welcome to the Merlina documentation! Choose your path:

## ✨ Merlina 3 — Wishcraft

- **[Development Plan](dev/merlina-3.md)** - Product direction, current capabilities,
  implementation milestones, and the initial backlog for the planned SWL.ai service

## 📖 For Users

Start here if you want to use Merlina to train models:

- **[Quick Start](user/quick-start.md)** - Get up and running in minutes
- **[Dataset Guide](user/dataset-guide.md)** - Configure datasets (HuggingFace, local files, uploads)
- **[Tokenizer Format Guide](user/tokenizer-format.md)** - Use automatic chat formatting (recommended!)
- **[MCP Server Guide](user/mcp.md)** - Drive Merlina from an LLM agent via Model Context Protocol
- **[Remote Training](user/remote-training.md)** - Current self-hosted RunPod workflow
- **[Internal Dataset Store](internal-dataset-store.md)** - Existing S3/MinIO/R2 dataset support

## 🔧 For Developers

Start here if you want to understand or contribute to Merlina:

- **[Dataset Implementation](dev/dataset-implementation.md)** - How the dataset system works
- **[Tokenizer Implementation](dev/tokenizer-implementation.md)** - Tokenizer formatter details
- **[Implementation Notes](dev/implementation-notes.md)** - General implementation notes
- **[Preview Feature](dev/preview-feature.md)** - Preview endpoint documentation

## 💡 Quick Links

- **Main README**: [../README.md](../README.md)
- **Examples**: [../examples/](../examples/)
- **Tests**: [../tests/](../tests/)

## 📚 Documentation Structure

```
docs/
├── user/                    # User-facing documentation
│   ├── quick-start.md      # Getting started guide
│   ├── dataset-guide.md    # Complete dataset configuration
│   └── tokenizer-format.md # Tokenizer format feature
│
└── dev/                     # Developer documentation
    ├── merlina-3.md          # Wishcraft product and implementation plan
    ├── dataset-implementation.md
    ├── tokenizer-implementation.md
    ├── implementation-notes.md
    └── preview-feature.md
```
