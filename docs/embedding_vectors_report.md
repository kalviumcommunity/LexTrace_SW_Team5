# LexTrace Text Embedding Vectors & Corpus Retrieval Report

## Executive Summary
Keyword search engines (e.g. BM25, TF-IDF) index exact string tokens. When a user query uses synonyms, paraphrased phrasing, or different terminology from the underlying document, keyword search fails completely.

This module delivers the **API-Based Embedding Generator & Corpus Vector Store Engine** (`EmbeddingGenerator`), which converts prepared document chunks into 1536-dimensional dense numerical vectors using an OpenAI-compatible embeddings API (`text-embedding-3-small`). By projecting text into a continuous semantic vector space and storing vectors alongside source text and metadata, LexTrace enables precise conceptual retrieval for RAG pipelines.

---

## 1. What Embedding Vectors Represent & Dimensionality (Task 1 & Task 4)

An **embedding vector** is a dense array of floating-point numbers (e.g., 1,536 dimensions for `text-embedding-3-small`) representing text meaning in a high-dimensional geometric space:

$$\vec{v} \in \mathbb{R}^{1536}$$

- **Not Random Database IDs**: Database primary keys are arbitrary identifiers with zero semantic relationships.
- **Not Sparse Keyword Counts**: TF-IDF vectors store word counts across a discrete vocabulary (mostly zeros). Embeddings are dense continuous vectors where every dimension encodes latent semantic features.
- **Latent Feature Representation**: Dimensions capture abstract conceptual attributes such as topic domain, sentiment, intent, technical specificity, and grammatical structure.

### Empirical Dimensionality Audit:
- **Embedding Model**: `text-embedding-3-small`
- **Total Chunks Embedded**: `15`
- **Vector Dimension Length**: `1536` components per vector
- **Dimensionality Status**: `[UNIFORM_DIMENSIONS]` (100% of corpus chunks produced identical 1536-dimensional vectors)

---

## 2. Why the Same Embedding Model Must Be Used for Documents and Queries

In vector search, document chunks and user queries are compared by calculating geometric distance (such as cosine similarity) in vector space.

- **Unique Coordinate Space**: Every embedding model projects text into its own proprietary latent space with specific vector dimensions (e.g., OpenAI `text-embedding-3-small` uses 1536 dimensions, whereas `bge-small-en-v1.5` uses 384 dimensions).
- **Dimensionality Mismatch**: Comparing vectors across different models results in mathematical dimension mismatches ($\mathbb{R}^{1536}$ vs $\mathbb{R}^{384}$), breaking vector math.
- **Semantic Axis Mismatch**: Even if two models use identical dimension lengths, each model maps semantic features to different axes. Vector dimension index #42 in Model A does not correspond to dimension #42 in Model B.
- **Retrieval Requirement**: Therefore, **the exact same embedding model used during document ingestion must be used to embed incoming user queries**.

---

## 3. How a Text Chunk Becomes a Vector in Code (Task 1 & Task 3)

The embedding pipeline converts plain text chunks into vectors via environment-configured API calls:

1. **Environment-Based Configuration (Task 3)**:
   API parameters (`OPENAI_API_KEY`, `EMBEDDING_MODEL`, `OPENAI_API_BASE`) are read strictly from environment variables or `.env` files via `src/config.py`:
   ```python
   EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
   OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
   OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
   ```
2. **API Embedding Request**:
   Text chunks are passed to the `EmbeddingGenerator.generate_embedding()` method, which invokes the OpenAI client:
   ```python
   response = self.client.embeddings.create(
       input=text,
       model=self.model
   )
   vector = response.data[0].embedding
   ```
3. **Verification**:
   The returned array of 1536 float values represents the dense embedding vector.

---

## 4. How Embeddings Are Stored with Source Text & Metadata for Retrieval (Task 2 & Task 5)

For a RAG system to generate accurate citations and context windows, embedding vectors cannot be stored in isolation. Each vector must be permanently coupled with its source text and metadata.

### Storage JSON Schema (`outputs/embedded_corpus_vectors.json`):
```json
{
  "chunk_id": "data/sample_corpus/company_policy.pdf#chunk-001",
  "source_id": "data/sample_corpus/company_policy.pdf",
  "chunk_index": 0,
  "section": "1. Scope and Applicability",
  "page_number": 1,
  "file_type": "pdf",
  "source_text": "LexTrace Governance & Compliance Policy\n1. Scope and Applicability\nThis policy applies to all full-time employees...",
  "metadata": {
    "source_id": "data/sample_corpus/company_policy.pdf",
    "section": "1. Scope and Applicability",
    "page_number": 1,
    "chunk_index": 0,
    "start_char": 0,
    "end_char": 365,
    "file_type": "pdf",
    "char_count": 365,
    "token_count": 70,
    "strategy": "recursive_semantic"
  },
  "vector_length": 1536,
  "trimmed_vector": [-0.038988, 0.016522, -0.01919, 0.018269, -0.039385],
  "embedding": [-0.0389875, 0.0165217, ...]
}
```

This structure ensures that when vector search finds the top-$k$ most similar vectors, the RAG pipeline immediately retrieves the exact source text and citation metadata (`source_id`, `section`, `page_number`, `chunk_index`).

---

## 5. Cosine Similarity & Semantic Distance Evaluation

The semantic angle between two vectors $\vec{u}$ and $\vec{v}$ is calculated using **Cosine Similarity**:

$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

### Benchmark Evaluation Results:

| Text Pair | Category | Cosine Similarity | Semantic Distance | Result |
|---|---|---|---|---|
| **Text A vs Text B** | **SIMILAR** (Security Policies) | **0.8961** (89.6%) | Very Close | High Similarity |
| **Text A vs Text C** | **DISSIMILAR** (Security vs Cafeteria) | **-0.0016** (-0.2%) | Far | Orthogonal |
| **Text B vs Text C** | **DISSIMILAR** (Security vs Cafeteria) | **0.0002** (0.0%) | Far | Orthogonal |

$$\text{Proof}: \text{CosineSim}(\text{Similar Pair}) = 0.8961 \gg \text{CosineSim}(\text{Dissimilar Pair}) = -0.0016 \quad [\text{PASSED}]$$

---

## 6. Follow-up: What Happens to Cost and Latency as the Corpus Grows?

As a document corpus scales from 15 chunks to 100,000+ chunks, system characteristics evolve significantly:

### 1. Ingestion Cost ($O(N)$ Tokens):
- **API Token Billing**: API embedding costs scale linearly with total tokens in the corpus ($N$). For OpenAI `text-embedding-3-small` ($0.02 per 1M tokens), embedding 1M tokens costs \$0.02.
- **Optimization Strategy**: Batching API calls (up to 2,048 texts per API request) minimizes HTTP connection overhead. Caching vector outputs using hash fingerprints prevents re-embedding unchanged document chunks.

### 2. Retrieval Latency ($O(N)$ vs $O(\log N)$ Search):
- **Exact Flat Search ($O(N)$)**: Brute-force cosine distance comparison against all $N$ stored vectors scales linearly. At 1M chunks, comparing a query against 1M 1536-dim vectors takes hundreds of milliseconds.
- **Approximate Nearest Neighbor (ANN) Indexing ($O(\log N)$)**: Vector databases (e.g. ChromaDB, Qdrant, Pinecone) build ANN indexes like **HNSW (Hierarchical Navigable Small World)** or **IVF-PQ (Inverted File with Product Quantization)**. HNSW reduces search latency from $O(N)$ to $O(\log N)$, allowing sub-10ms retrieval across millions of vectors.

---

## 7. Artifacts & Verification Summary (Task 4 & Task 5)

- **API Embedding Script**: [`src/embedding_generator.py`](file:///c:/Users/Priyamtha/OneDrive/Desktop/WI%20PROJECT/LexTrace_SW_Team5/src/embedding_generator.py)
- **CLI Runner**: [`src/run_embedding_demo.py`](file:///c:/Users/Priyamtha/OneDrive/Desktop/WI%20PROJECT/LexTrace_SW_Team5/src/run_embedding_demo.py)
- **Stored Corpus Vectors**: [`outputs/embedded_corpus_vectors.json`](file:///c:/Users/Priyamtha/OneDrive/Desktop/WI%20PROJECT/LexTrace_SW_Team5/outputs/embedded_corpus_vectors.json)
- **Benchmark Summary Output**: [`outputs/embedding_demonstration_results.json`](file:///c:/Users/Priyamtha/OneDrive/Desktop/WI%20PROJECT/LexTrace_SW_Team5/outputs/embedding_demonstration_results.json)
- **Execution Evidence Log**: [`outputs/embedding_vector_analysis.log`](file:///c:/Users/Priyamtha/OneDrive/Desktop/WI%20PROJECT/LexTrace_SW_Team5/outputs/embedding_vector_analysis.log)
