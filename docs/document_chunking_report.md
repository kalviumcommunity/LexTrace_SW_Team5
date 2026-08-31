# Document Chunking Strategies & Comparison Technical Report

## Executive Summary

Retrieval-Augmented Generation (RAG) systems cannot embed or retrieve entire documents as single units. Large documents exceed embedding model input preferences, inflate vector storage costs, and dilute semantic vector similarity. Conversely, splitting text too aggressively into tiny fragments destroys contextual semantics.

This technical report presents a comparative evaluation of two distinct chunking strategies — **Fixed-Size Window Chunking with Overlap** vs. **Recursive Semantic Boundary Chunking** — evaluated across the LexTrace RAG corpus.

---

## 1. Why Documents Must Be Chunked for RAG Retrieval

In an enterprise RAG architecture:
1. **Precision & Noise Reduction**: Whole documents contain multiple topics. Querying a 5,000-word policy document for a specific 2-sentence rule retrieves unnecessary noise that dilutes vector similarity matches.
2. **Context Window & Cost Management**: LLM prompts (`gpt-4o-mini`) operate under token budgets. Injecting concise, highly relevant chunks minimizes prompt token consumption, reduces API latency, and lowers operational cost.
3. **Embedding Vector Density**: Embedding models (`text-embedding-3-small`) produce denser, more accurate vector representations when encoding focused semantic units (100–300 tokens) compared to multi-page documents.

---

## 2. Comparison of Chunking Strategies

We evaluated two strategy implementations on the same cleaned corpus (`data/sample_corpus/`):

### Strategy A: Fixed-Size Character Window with Overlap
- **Mechanics**: Moves a rigid sliding window of fixed character length (e.g. 500 characters) with a fixed overlap step (e.g. 100 characters).
- **Pros**: Simple, predictable chunk counts, deterministic execution time.
- **Cons**: Cuts text arbitrarily mid-word or mid-sentence, splitting critical facts across chunk boundaries and introducing 50% sentence fragmentation.

### Strategy B: Recursive Semantic Boundary Chunking (Recommended)
- **Mechanics**: Hierarchically splits text on natural semantic boundaries (`\n\n` -> `\n` -> `. ` -> `; ` -> ` `) and recombines fragments up to the target chunk size with overlap.
- **Pros**: Respects document structural hierarchy (paragraphs, sections, complete sentences), eliminates mid-sentence cuts, and preserves semantic cohesion.
- **Cons**: Slightly variable chunk lengths depending on paragraph density.

---

## 3. Empirical Benchmark & Quantitative Metrics

Evaluated across the 5 valid multi-format corpus documents using `gpt-4o-mini` (`cl100k_base` tokenizer):

| Metric / Dimension | Strategy A: Fixed Window | Strategy B: Recursive Semantic | Operational Impact |
|---|---|---|---|
| **Total Chunks Generated** | 10 chunks | 12 chunks | Semantic strategy produces +2 chunks due to paragraph boundary alignment. |
| **Average Chunk Size (Chars)** | 406.9 chars | 354.8 chars | Semantic chunks average ~355 chars (~72 tokens), fitting vector search sweet spot. |
| **Min / Max Chunk Size (Chars)** | 209 / 500 chars | 243 / 472 chars | Semantic chunk sizes naturally adapt between 243–472 chars based on sentence boundaries. |
| **Average Chunk Size (Tokens)** | 83.3 tokens | 72.2 tokens | Both stay safely under 100 tokens per chunk. |
| **Total Corpus Tokens** | 833 tokens | 867 tokens | Comparable token footprint (+4% for semantic due to overlap preservation). |
| **Sentence Fragmentation Rate** | **50.0%** (5/10 cut mid-sentence) | **16.7%** (2/12 boundary tail only) | **Fixed window splits 50% of sentences mid-word/sentence**, severely corrupting retrieval. |

---

## 4. Sample Boundary Comparison on Same Document

### Document: `data/sample_corpus/employee_handbook.txt`

#### Strategy A (Fixed Window): Fragmented Cut Sample
```text
[Chunk ID: data/sample_corpus/employee_handbook.txt#chunk-001]
"...LexTrace Internal Employee Handbook 1. Code of Conduct All team members are expected to maintain professional standards... 3. System Access & Security - Mandatory multi-factor authenticatio..."
```
> ⚠️ **Issue**: Cuts the word `"authentication"` mid-string into `"authenticatio..."`, truncating Security Rule #3.

#### Strategy B (Recursive Semantic): Clean Boundary Sample
```text
[Chunk ID: data/sample_corpus/employee_handbook.txt#chunk-001]
"LexTrace Internal Employee Handbook 1. Code of Conduct All team members are expected to maintain professional standards, foster an inclusive engineering culture, and protect company and client confidentiality at all times. 2. Working Hours & Remote Work LexTrace operates on a flexible hybrid work schedule. Core sync hours are between 10:00 AM and 4:00 PM local time. Remote collaboration requires active status updates in Slack."
```
> ✅ **Benefit**: Preserves complete Code of Conduct and Working Hours sections without splitting sentences.

---

## 5. Justification of Selected Strategy

We select **Strategy B: Recursive Semantic Boundary Chunking** as the production standard for LexTrace:

1. **Sentence & Fact Integrity**: Eliminates arbitrary mid-word/sentence cuts that plague fixed sliding windows (50% vs 16.7% fragmentation).
2. **Retrieval Accuracy**: Keeping complete clauses together ensures that vector distance matches the full context of policies, security rules, and architectural specs.
3. **Subword Tokenizer Compatibility**: Avoids feeding truncated word stubs (`authenticatio...`) to `tiktoken`, preserving vocabulary alignment.

---

## 6. How Chunk Size Relates to the Context Window

- **Small Chunks (100–300 tokens)**: High precision, minimal prompt overhead, excellent for pinpointing specific facts or answers. Allows fitting 10–20 retrieved chunks into a standard RAG prompt without overwhelming LLM context window bounds.
- **Large Chunks (800–2,000 tokens)**: Broader context, but risks embedding noise, inflating LLM prompt costs, and running into context window truncation when concatenating top-K search results.
- **LexTrace Standard**: Target chunk size of **500 chars (~100 tokens)** with **100 char overlap (~20 tokens)** achieves optimal balance between semantic precision and LLM context window economy.
