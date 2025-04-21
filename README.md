# 🚀 FloTorch-core

**FloTorch-core** is a modular and extensible Python framework for building LLM-powered RAG (Retrieval-Augmented Generation) pipelines. It offers plug-and-play components for embeddings, chunking, retrieval, gateway-based LLM calls, and RAG evaluation.

---

## ✨ Features

- 🔌 Unified LLM Gateway (OpenAI, Bedrock, Ollama, etc.)
- 💻 Bedrock/sagemaker/gateway inferencer
- 🧠 Embedding Models (Titan, Cohere, Bedrock)
- 🧩 Text Chunking (Fixed-size, Hierarchical)
- 🔍 Document Retrieval (OpenSearch + Vector Storage)
- 📏 RAG Evaluation (RAGAS Metrics)
- ☁️ AWS Integration (S3, DynamoDB, Lambda)
- 🧢 Built-in Testing Support

---

## 📆 Installation

```bash
pip install FloTorch-core
```

To install development dependencies:

```bash
pip install FloTorch-core[dev]
```

---

## 📂 Project Structure

```
flotorch/
├── gateway/            # LLM gateway interface
├── embedding/          # Embedding models
├── chunking/           # Text chunking logic
├── retriever/          # Retrieval pipeline
├── evaluation/         # RAG evaluation (RAGAS)
├── storage/            # Vector DB, S3, DynamoDB
├── util/               # Utilities and helpers
```

---

## 📬 Maintainer

**Shiva Krishna**  
📧 Email: shiva.krishnaah@gmail.com

**Adil Raza**  
📧 Email: adilraza.12345@gmail.com

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌐 Links

- GitHub: [https://github.com/FissionAI/flotorch-core](https://github.com/FissionAI/flotorch-core)

