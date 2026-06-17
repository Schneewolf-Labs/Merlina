# Merlina Documentation

Welcome to the Merlina documentation! Choose your path:

## 📖 For Users

Start here if you want to use Merlina to train models:

- **[Quick Start](user/quick-start.md)** - Get up and running in minutes
- **[Cookbook](user/cookbook.md)** - Battle-tested end-to-end recipes (SFT, ORPO, VLM) + the non-obvious gotchas
- **[Dataset Guide](user/dataset-guide.md)** - Configure datasets (HuggingFace, local files, uploads)
- **[Tokenizer Format Guide](user/tokenizer-format.md)** - Use automatic chat formatting (recommended!)
- **[MCP Server Guide](user/mcp.md)** - Drive Merlina from an LLM agent via Model Context Protocol

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
│   ├── cookbook.md         # End-to-end recipes + gotchas
│   ├── dataset-guide.md    # Complete dataset configuration
│   └── tokenizer-format.md # Tokenizer format feature
│
└── dev/                     # Developer documentation
    ├── dataset-implementation.md
    ├── tokenizer-implementation.md
    ├── implementation-notes.md
    └── preview-feature.md
```
