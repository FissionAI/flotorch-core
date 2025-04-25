# 🚀 FloTorch-core

**FloTorch-core** is a modular and extensible Python framework for building LLM-powered RAG (Retrieval-Augmented Generation) pipelines. It offers plug-and-play components for embeddings, chunking, retrieval, gateway-based LLM calls, and RAG evaluation.

---

## ✨ Features

- 🧩 Text Chunking (Fixed-size, Hierarchical)
- 🧠 Embedding Models (Titan, Cohere, Bedrock)
- 🔍 Document Retrieval (OpenSearch + Vector Storage)
- 💻 Bedrock/sagemaker/gateway inferencer
- 🔌 Unified LLM Gateway (OpenAI, Bedrock, Ollama, etc.)
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
├── inferencer/         # LLM gateway/bedrock/sagemaker interface
├── embedding/          # Embedding models
├── chunking/           # Text chunking logic
├── evaluator/          # RAG evaluation (RAGAS)
├── storage/            # Vector DB, S3, DynamoDB
├── util/               # Utilities and helpers
├── rerank/             # Ranking documents
├── guardrails/         # Enabling guardrails
├── reader/             # reader for json/pdf
```

---

## 📖 Usage Example

### Reader

```
from flotorch_core.reader.json_reader import JSONReader
from flotorch_core.storage.s3_storage import S3StorageProvider

json_reader = JSONReader(S3StorageProvider(<S3 bucket>))
json_reader.read(<path>)
```

### Embedding
```
from flotorch_core.embedding.embedding_registry import embedding_registry

embedding_class = embedding_registry.get_model(<model id>)

# model id example: amazon.titan-text-express-v1, amazon.titan-embed-text-v2:0, cohere.embed-multilingual-v3
```

### Vector storage (opensearch)
```
from flotorch_core.storage.db.vector.open_search import OpenSearchClient

vector_storage_object = OpenSearchClient(
    <opensearch_host>, 
    <opensearch_port>, 
    <opensearch_username>, 
    <opensearch_password>, 
    <index_id>, 
    <embedding object>
)
```

### Vector storage (bedrock knowledgebase)
```
from flotorch_core.storage.db.vector.bedrock_knowledgebase_storage import BedrockKnowledgeBaseStorage

vector_storage_object = BedrockKnowledgeBaseStorage(
    knowledge_base_id=<knowledge_base_id>,
    region=<aws_region>
)
```

### Guardrails over vector storage
```
from flotorch_core.storage.db.vector.guardrails_vector_storage import GuardRailsVectorStorage

base_guardrails = BedrockGuardrail(<guardrail_id>, <guardrail_version>, <aws_region>)            
vector_storage_object = GuardRailsVectorStorage(
    vector_storage_object, 
    base_guardrails,
    <enable_prompt_guardrails(True/False)>,
    <enable_context_guardrails(True/False)>
)
```

### Inferencer
```
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer
from flotorch_core.inferencer.gateway_inferencer import GatewayInferencer
from flotorch_core.inferencer.sagemaker_inferencer import SageMakerInferencer

inferencer = BedrockInferencer(
    <model_id>, 
    <region>, 
    <number of n_shot_prompts>, 
    <temperature>, 
    <n_shot_prompt_guide_obj>
)

inferencer = GatewayInferencer(
    model_id=<model_id>, 
    api_key=<api_key>, 
    base_url=<base_url>, 
    n_shot_prompts=<n_shot_prompts>, 
    n_shot_prompt_guide_obj=<n_shot_prompt_guide_obj>
)

inferencer = SageMakerInferencer(
    <model_id>, 
    <region>, 
    <arn_role>, 
    <n_shot_prompts>, 
    <temperature>, 
    <n_shot_prompt_guide_obj>
)
```

### GuardRail over inferencer

```
from flotorch_core.inferencer.guardrails.guardrails_inferencer import GuardRailsInferencer

inferencer = GuardRailsInferencer(inferencer, base_guardrails)
```

### Run unit test coverage and generate html report locally

- First, install required packages for calculating the unit test coverage:

    `pip install pytest pytest-cov coverage`

- From the project root directory, run:

    `coverage run -m pytest`

- To see the report on the terminal, run:

    `coverage report -m`

- To generate html report, run:

    `coverage html`
    
  open htmlcov/index.html

---


## 📬 Maintainer

**Shiva Krishna**  
📧 Email: shiva.krishnaah@gmail.com

**Adil Raza**  
📧 Email: adilraza.9752@gmail.com

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌐 Links

- GitHub: [https://github.com/FissionAI/flotorch-core](https://github.com/FissionAI/flotorch-core)

- Use **snake_case** for:
    - names
    - Configuration variables
    - python file names
      Example: `example_snake_case`



