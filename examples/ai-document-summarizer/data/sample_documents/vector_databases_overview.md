# Vector Databases: A Technical Overview

## Introduction

Vector databases are specialized database systems designed to store, index, and retrieve high-dimensional vectors efficiently. They are a cornerstone of modern AI applications, enabling similarity search, semantic search, and nearest-neighbor retrieval at scale.

Unlike traditional relational databases that match exact values, vector databases operate on mathematical representations (embeddings) of data — whether text, images, audio, or any other modality — and find items that are "close" in meaning or content.

## Why Vector Databases?

The rise of large language models (LLMs) and embedding models has created a massive demand for vector storage. When you embed a sentence using a model like all-MiniLM-L6-v2 or OpenAI's text-embedding-ada-002, you get a dense array of floating-point numbers (a vector) that encodes the semantic meaning of that text.

Storing these vectors and finding the most similar ones for a given query is the fundamental operation behind:

- **Retrieval-Augmented Generation (RAG)**: LLMs retrieve relevant chunks from a knowledge base before generating answers
- **Semantic Search**: Finding documents by meaning, not just keyword overlap
- **Recommendation Systems**: Matching users to products, or articles to readers
- **Image Search**: Finding visually similar images
- **Anomaly Detection**: Identifying vectors that deviate significantly from the norm

## Key Concepts

### Embeddings
Embeddings are dense numerical representations of data. For text, models like sentence-transformers encode sentences into vectors of fixed dimensionality (e.g., 384 or 1536 dimensions).

### Similarity Metrics
Vector databases support several distance/similarity metrics:
- **Cosine Similarity**: Measures the angle between two vectors. Best for semantic text matching.
- **Euclidean Distance (L2)**: Measures straight-line distance. Common in image embedding spaces.
- **Dot Product**: Unnormalized inner product, often used in recommendation systems.

### Approximate Nearest Neighbor (ANN) Search
Exact nearest-neighbor search becomes computationally infeasible for millions of vectors. ANN algorithms like HNSW (Hierarchical Navigable Small World graphs) and IVF (Inverted File Index) enable fast, approximate searches with configurable recall-speed tradeoffs.

## Endee Vector Database

Endee is a high-performance, open-source vector database designed to handle up to 1 billion vectors on a single node. It uses optimized indexing strategies and supports both dense and sparse vector retrieval.

### Key Features
- **Python SDK**: Install with `pip install endee`
- **Multiple precision levels**: INT8, FP16, FP32 for storage/speed tradeoffs
- **Cosine, Euclidean, and Dot Product** space types
- **Metadata filtering**: Store and filter by payload fields alongside vectors
- **Docker deployment**: Runs as a lightweight container on port 8080
- **REST API**: Direct HTTP access for any language

### Example: Creating and Querying an Index

```python
from endee import Endee, Precision

client = Endee()
client.create_index(
    name="documents",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8,
)

index = client.get_index("documents")

# Upsert vectors
index.upsert([
    {"id": "chunk_1", "vector": [...384 floats...], "meta": {"text": "Hello world"}}
])

# Query
results = index.query(vector=[...384 floats...], top_k=5)
for r in results:
    print(r.id, r.similarity, r.meta)
```

## RAG with Vector Databases

The Retrieval-Augmented Generation pattern combines vector search with LLM generation:

1. **Indexing time**: Embed document chunks → store in vector DB
2. **Query time**: Embed query → search vector DB → retrieve top-k chunks → pass to LLM
3. **Generation**: LLM generates a grounded, context-aware answer

This pattern dramatically reduces hallucinations compared to raw LLM inference, because the model has access to relevant, verified context from your document corpus.

## Conclusion

Vector databases have become the memory layer of AI applications. As embedding models continue to improve and the volume of unstructured data grows, efficient vector retrieval will remain at the heart of intelligent search and generation systems.
