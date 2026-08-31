import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger("DocumentChunker")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken package not installed. Token counts will be estimated (1 token ≈ 4 chars).")


@dataclass
class DocumentChunk:
    """
    Standardized chunk representing a retrieved context unit in RAG applications.
    Retains source lineage metadata and exact text boundaries.
    """
    chunk_id: str             # e.g. "data/sample_corpus/employee_handbook.txt#chunk-001"
    source_id: str            # Original document relative path
    chunk_index: int          # 0-indexed position within parent document
    text: str                 # Extracted chunk text content
    char_count: int           # Total character count
    word_count: int           # Total word count
    token_count: int          # Exact or estimated tiktoken count
    strategy: str             # Strategy used: "fixed_window" or "recursive_semantic"
    start_char: int           # Start character index in parent document
    end_char: int             # End character index in parent document

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk metadata to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "token_count": self.token_count,
            "strategy": self.strategy,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "snippet": self.text[:120] + ("..." if len(self.text) > 120 else "")
        }


class DocumentChunker:
    """
    Multi-strategy text chunker for RAG context preparation.
    Supports Fixed-Size Sliding Window Chunking and Recursive Semantic Boundary Chunking.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Calculate exact token count using tiktoken or fallback estimation."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback estimation: ~4 chars per token
        return max(1, len(text) // 4)

    def chunk_fixed_window(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
        source_id: str = "doc"
    ) -> List[DocumentChunk]:
        """
        Strategy A: Fixed-size character window with overlapping sliding step (Task 1).

        Args:
            text: Normalized document text.
            chunk_size: Fixed target character length per chunk.
            overlap: Overlapping character count between consecutive chunks.
            source_id: Origin document identifier for citation.

        Returns:
            List of DocumentChunk objects.
        """
        if not text:
            return []

        chunks = []
        step = max(1, chunk_size - overlap)
        chunk_idx = 0
        i = 0
        text_len = len(text)

        while i < text_len:
            end = min(i + chunk_size, text_len)
            chunk_text = text[i:end].strip()

            if chunk_text:
                chunk_id = f"{source_id}#chunk-{chunk_idx + 1:03d}"
                c = DocumentChunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    token_count=self.count_tokens(chunk_text),
                    strategy="fixed_window",
                    start_char=i,
                    end_char=end
                )
                chunks.append(c)
                chunk_idx += 1

            if end == text_len:
                break
            i += step

        return chunks

    def chunk_recursive_semantic(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
        source_id: str = "doc",
        separators: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """
        Strategy B: Recursive semantic boundary chunking preserving paragraphs & sentences (Task 1).

        Args:
            text: Normalized document text.
            chunk_size: Maximum target character size per chunk.
            overlap: Overlap target in characters.
            source_id: Origin document identifier for citation.
            separators: Hierarchical separator list.

        Returns:
            List of DocumentChunk objects.
        """
        if not text:
            return []

        seps = separators or self.DEFAULT_SEPARATORS
        raw_splits = self._recursive_split(text, chunk_size, seps)

        # Merge split fragments into coherent chunks up to chunk_size with overlap
        chunks = []
        current_chunk_text = ""
        current_start = 0
        chunk_idx = 0

        for fragment in raw_splits:
            if not fragment:
                continue

            if not current_chunk_text:
                current_chunk_text = fragment
            elif len(current_chunk_text) + len(fragment) + 1 <= chunk_size:
                current_chunk_text += (" " if not current_chunk_text.endswith("\n") else "") + fragment
            else:
                # Emit current chunk
                cleaned = current_chunk_text.strip()
                if cleaned:
                    chunk_id = f"{source_id}#chunk-{chunk_idx + 1:03d}"
                    c = DocumentChunk(
                        chunk_id=chunk_id,
                        source_id=source_id,
                        chunk_index=chunk_idx,
                        text=cleaned,
                        char_count=len(cleaned),
                        word_count=len(cleaned.split()),
                        token_count=self.count_tokens(cleaned),
                        strategy="recursive_semantic",
                        start_char=current_start,
                        end_char=current_start + len(cleaned)
                    )
                    chunks.append(c)
                    chunk_idx += 1

                # Retain overlap tail from current chunk for next chunk
                overlap_text = current_chunk_text[-overlap:] if len(current_chunk_text) > overlap else ""
                current_start = current_start + len(current_chunk_text) - len(overlap_text)
                current_chunk_text = (overlap_text + " " + fragment).strip()

        # Emit trailing chunk
        cleaned = current_chunk_text.strip()
        if cleaned:
            chunk_id = f"{source_id}#chunk-{chunk_idx + 1:03d}"
            c = DocumentChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                chunk_index=chunk_idx,
                text=cleaned,
                char_count=len(cleaned),
                word_count=len(cleaned.split()),
                token_count=self.count_tokens(cleaned),
                strategy="recursive_semantic",
                start_char=current_start,
                end_char=current_start + len(cleaned)
            )
            chunks.append(c)

        return chunks

    def _recursive_split(self, text: str, max_size: int, separators: List[str]) -> List[str]:
        """Hierarchically split text on the first matching separator."""
        if len(text) <= max_size or not separators:
            return [text]

        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            # Character fallback
            return [text[i:i + max_size] for i in range(0, len(text), max_size)]

        parts = text.split(sep)
        result = []
        for part in parts:
            if not part:
                continue
            if len(part) > max_size and next_seps:
                result.extend(self._recursive_split(part, max_size, next_seps))
            else:
                result.append(part)

        return result

    def compare_strategies(
        self,
        docs: List[Any],
        chunk_size: int = 500,
        overlap: int = 100
    ) -> Dict[str, Any]:
        """
        Run both chunking strategies across a list of LoadedDocument objects and return comparison metrics (Tasks 2 & 3).

        Returns:
            Dict containing detailed metrics for both strategies.
        """
        valid_docs = [d for d in docs if d.status == "SUCCESS"]

        fixed_chunks: List[DocumentChunk] = []
        semantic_chunks: List[DocumentChunk] = []

        for doc in valid_docs:
            f_c = self.chunk_fixed_window(doc.text_content, chunk_size, overlap, doc.source_id)
            s_c = self.chunk_recursive_semantic(doc.text_content, chunk_size, overlap, doc.source_id)
            fixed_chunks.extend(f_c)
            semantic_chunks.extend(s_c)

        def compute_stats(chunks: List[DocumentChunk], name: str) -> Dict[str, Any]:
            if not chunks:
                return {"name": name, "total_chunks": 0}

            char_lens = [c.char_count for c in chunks]
            token_lens = [c.token_count for c in chunks]

            # Detect sentence fragmentation (chunk ends mid-sentence without terminal punctuation)
            fragmented = sum(
                1 for c in chunks if not c.text.strip().endswith((".", "!", "?", ":", '"'))
            )
            frag_pct = round((fragmented / len(chunks) * 100), 1)

            return {
                "name": name,
                "total_chunks": len(chunks),
                "avg_char_size": round(sum(char_lens) / len(chunks), 1),
                "min_char_size": min(char_lens),
                "max_char_size": max(char_lens),
                "avg_token_size": round(sum(token_lens) / len(chunks), 1),
                "min_token_size": min(token_lens),
                "max_token_size": max(token_lens),
                "total_tokens": sum(token_lens),
                "fragmented_chunks": fragmented,
                "fragmentation_pct": frag_pct
            }

        return {
            "total_documents": len(valid_docs),
            "target_chunk_size": chunk_size,
            "target_overlap": overlap,
            "fixed_window_stats": compute_stats(fixed_chunks, "Fixed Window with Overlap"),
            "recursive_semantic_stats": compute_stats(semantic_chunks, "Recursive Semantic Boundary"),
            "fixed_chunks": fixed_chunks,
            "semantic_chunks": semantic_chunks
        }
