# LexTrace RAG Architecture & Ingestion Guidelines

## Overview
The LexTrace Retrieval-Augmented Generation (RAG) system ingests heterogeneous internal documents, indexes them into vector storage, and provides precise context-grounded answers for internal team queries.

## Component Specifications

### 1. Document Ingestion Layer
- **Multi-format Support**: Parses `.pdf`, `.html`, `.md`, and `.txt` files.
- **Normalization**: Extracts raw text, strips HTML DOM tags, and standardizes white space.
- **Source Identification**: Attaches relative file paths to retain auditability and citation lineage.

### 2. Vector Embedding & Retrieval
- **Embedding Model**: `text-embedding-3-small` (1536 dimensions).
- **Vector Store**: ChromaDB instance running in persistence mode.
- **Chunking Strategy**: Recursive character splitter with 500-token chunk size and 50-token overlap.

### 3. Context Window Management
- Enforces strict token budgets using `tiktoken` (`cl100k_base` / `o200k_base`).
- Prunes oldest messages dynamically when conversation context exceeds system bounds.
