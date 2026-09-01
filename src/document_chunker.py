import re
from dataclasses import dataclass, field
from datetime import datetime
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
class ChunkMetadata:
    """
    Standardized, consistent metadata structure attached to every chunk across the corpus.
    Guarantees key uniformity for downstream vector indexing, citation tracing, and metadata filtering.
    """
    source_id: str                          # Original document relative path / identifier (Task 1)
    section: Optional[str] = None           # Active section heading or document header (Task 2)
    page_number: Optional[int] = 1          # Document page number for multi-page formats (Task 2)
    chunk_index: int = 0                    # 0-indexed position within parent document (Task 2)
    start_char: int = 0                     # Character offset start position in parent document (Task 2)
    end_char: int = 0                       # Character offset end position in parent document (Task 2)
    file_type: str = "txt"                  # Extension / format tag: pdf, html, md, txt
    char_count: int = 0                     # Total character count in chunk
    token_count: int = 0                    # Exact or estimated tiktoken count
    strategy: str = "recursive_semantic"    # Chunking strategy used: fixed_window or recursive_semantic
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk metadata to a consistent dictionary structure."""
        return {
            "source_id": self.source_id,
            "section": self.section,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "file_type": self.file_type,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "strategy": self.strategy,
            "created_at": self.created_at
        }


@dataclass
class DocumentChunk:
    """
    Standardized chunk representing a retrieved context unit in RAG applications.
    Retains source lineage metadata and exact text boundaries in a consistent structure.
    """
    chunk_id: str             # e.g. "data/sample_corpus/employee_handbook.txt#chunk-001"
    text: str                 # Extracted chunk text content
    metadata: ChunkMetadata   # Unified metadata object containing standard fields

    # Backward-compatible convenience properties
    @property
    def source_id(self) -> str:
        return self.metadata.source_id

    @property
    def chunk_index(self) -> int:
        return self.metadata.chunk_index

    @property
    def char_count(self) -> int:
        return self.metadata.char_count

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def token_count(self) -> int:
        return self.metadata.token_count

    @property
    def strategy(self) -> str:
        return self.metadata.strategy

    @property
    def start_char(self) -> int:
        return self.metadata.start_char

    @property
    def end_char(self) -> int:
        return self.metadata.end_char

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk metadata to a consistent serializable dictionary."""
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
            "metadata": self.metadata.to_dict(),
            "snippet": self.text[:120] + ("..." if len(self.text) > 120 else "")
        }


class ChunkTracebackEngine:
    """
    Engine to trace a retrieved RAG document chunk back to its exact origin in the source document (Task 4).
    Verifies character range offsets, checks text match precision, extracts surrounding context,
    and formats formal citation strings for LLM responses.
    """

    @staticmethod
    def trace_chunk(
        chunk: DocumentChunk,
        parent_doc: Any,
        context_window: int = 120
    ) -> Dict[str, Any]:
        """
        Trace a chunk back to its parent document using its metadata.

        Args:
            chunk: The DocumentChunk object to trace.
            parent_doc: The LoadedDocument (or document object) containing full text_content.
            context_window: Characters of surrounding context to include before & after.

        Returns:
            Dict containing traceback details, citation string, match verification, and context window.
        """
        source_id = chunk.metadata.source_id
        start_char = chunk.metadata.start_char
        end_char = chunk.metadata.end_char
        doc_text = getattr(parent_doc, "text_content", str(parent_doc))

        # Check source ID match
        doc_source_id = getattr(parent_doc, "source_id", "unknown")
        id_matched = (source_id == doc_source_id) or (source_id in doc_source_id) or (doc_source_id in source_id)

        # Slice parent document text by character position metadata
        extracted_slice = doc_text[start_char:end_char]
        
        # Normalize whitespace for position verification across line breaks
        norm_extracted = re.sub(r'\s+', ' ', extracted_slice).strip()
        norm_chunk = re.sub(r'\s+', ' ', chunk.text).strip()
        exact_text_match = (norm_extracted == norm_chunk) or (norm_chunk in norm_extracted) or (norm_extracted in norm_chunk)

        # Substring fallback verification if character offsets slightly shifted
        if not exact_text_match and chunk.text[:40].strip() in doc_text:
            alt_start = doc_text.find(chunk.text[:40].strip())
            position_verified = True
        else:
            position_verified = exact_text_match or (chunk.text[:30].strip() in extracted_slice)

        # Extract surrounding context window
        prefix_start = max(0, start_char - context_window)
        suffix_end = min(len(doc_text), end_char + context_window)
        prefix_context = doc_text[prefix_start:start_char]
        suffix_context = doc_text[end_char:suffix_end]

        # Format standardized RAG Citation string
        sec_str = f" | Section: '{chunk.metadata.section}'" if chunk.metadata.section else ""
        page_str = f" | Page: {chunk.metadata.page_number}" if chunk.metadata.page_number else ""
        citation = f"[Source: {source_id}{sec_str}{page_str} | Pos: #{chunk.metadata.chunk_index + 1} | Chars: {start_char}-{end_char}]"

        return {
            "chunk_id": chunk.chunk_id,
            "source_id": source_id,
            "traceability_status": "VERIFIED" if (id_matched and position_verified) else "MISMATCH",
            "source_id_matched": id_matched,
            "position_verified": position_verified,
            "exact_text_match": exact_text_match,
            "start_char": start_char,
            "end_char": end_char,
            "citation_string": citation,
            "chunk_text": chunk.text,
            "surrounding_context": {
                "prefix": prefix_context,
                "target_chunk": chunk.text,
                "suffix": suffix_context
            },
            "metadata_summary": chunk.metadata.to_dict()
        }


class DocumentChunker:
    """
    Multi-strategy text chunker for RAG context preparation with metadata tagging & source traceability.
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

    def extract_section_header(self, text: str, start_char: int, file_type: str = "txt") -> Optional[str]:
        """
        Extract the active section or heading title in document text preceding start_char.
        Supports Markdown (# ## ###), HTML tags (<h1> <h2> <h3>), numbered sections (1. 2. Section 3),
        Q&A markers (Q1: Q2:), and capital title lines.
        """
        if not text or start_char < 0:
            return None

        # Inspect document text up to current chunk location
        preceding = text[:min(len(text), start_char + 150)]
        lines = preceding.splitlines()

        last_header = None
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Markdown headings (# Heading)
            md_match = re.match(r'^(#{1,6})\s+(.+)$', line_str)
            if md_match:
                last_header = md_match.group(2).strip()
                continue

            # HTML headings (<h1>Heading</h1>)
            html_match = re.search(r'<h[1-6][^>]*>(.*?)</h[1-6]>', line_str, re.IGNORECASE)
            if html_match:
                last_header = html_match.group(1).strip()
                continue

            # Q&A questions (Q1: ..., Q2: ...)
            qa_match = re.match(r'^(Q\d+:?\s+[^\n]+)', line_str, re.IGNORECASE)
            if qa_match:
                last_header = qa_match.group(1).strip()
                continue

            # Numbered sections (e.g. "1. Code of Conduct", "2. Password & Key Management", "Section 3.1: ...")
            sec_match = re.match(r'^(?:Section\s+)?\d+(?:\.\d+)*[\:\.]?\s+([A-Z][^\n]+)', line_str)
            if sec_match:
                last_header = line_str
                continue

            # Standalone Header Title
            if len(line_str) <= 60 and line_str[0].isupper() and not line_str.endswith((".", ",", ";")):
                keywords = ["Policy", "Architecture", "Handbook", "Guide", "Overview", "Specification", "Guidelines", "Standards"]
                if any(kw in line_str for kw in keywords):
                    last_header = line_str

        return last_header

    def extract_page_number(self, text: str, start_char: int) -> int:
        """
        Extract the active page number in document text preceding start_char.
        Searches for 'Page X of Y' or '- Page X -' markers. Defaults to 1.
        """
        if not text:
            return 1
        preceding = text[:start_char + 50]
        page_matches = re.findall(r'(?:Page|- Page)\s*(\d+)', preceding, re.IGNORECASE)
        if page_matches:
            return int(page_matches[-1])
        return 1

    def chunk_fixed_window(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
        source_id: str = "doc",
        file_type: str = "txt",
        section: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Strategy A: Fixed-size character window with overlapping sliding step and metadata tagging.
        """
        if not text:
            return []

        chunks = []
        step = max(1, chunk_size - overlap)
        chunk_idx = 0
        i = 0
        text_len = len(text)
        search_cursor = 0

        while i < text_len:
            end = min(i + chunk_size, text_len)
            chunk_text = text[i:end].strip()

            if chunk_text:
                start_offset = text.find(chunk_text, search_cursor)
                if start_offset == -1:
                    start_offset = i
                end_offset = start_offset + len(chunk_text)
                search_cursor = max(search_cursor, start_offset + 1)

                chunk_id = f"{source_id}#chunk-{chunk_idx + 1:03d}"
                active_sec = section if section is not None else self.extract_section_header(text, start_offset, file_type)
                active_page = page_number if page_number is not None else self.extract_page_number(text, start_offset)
                tok_cnt = self.count_tokens(chunk_text)

                meta = ChunkMetadata(
                    source_id=source_id,
                    section=active_sec,
                    page_number=active_page,
                    chunk_index=chunk_idx,
                    start_char=start_offset,
                    end_char=end_offset,
                    file_type=file_type,
                    char_count=len(chunk_text),
                    token_count=tok_cnt,
                    strategy="fixed_window"
                )

                c = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=meta
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
        file_type: str = "txt",
        section: Optional[str] = None,
        page_number: Optional[int] = None,
        separators: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """
        Strategy B: Recursive semantic boundary chunking with metadata tagging (Task 1 & Task 2).
        """
        if not text:
            return []

        seps = separators or self.DEFAULT_SEPARATORS
        raw_splits = self._recursive_split(text, chunk_size, seps)

        chunks = []
        current_chunk_text = ""
        chunk_idx = 0
        search_cursor = 0

        for fragment in raw_splits:
            if not fragment:
                continue

            if not current_chunk_text:
                current_chunk_text = fragment
            elif len(current_chunk_text) + len(fragment) + 1 <= chunk_size:
                current_chunk_text += (" " if not current_chunk_text.endswith("\n") else "") + fragment
            else:
                cleaned = current_chunk_text.strip()
                if cleaned:
                    start_offset = text.find(cleaned, search_cursor)
                    if start_offset == -1:
                        start_offset = search_cursor
                    end_offset = start_offset + len(cleaned)
                    search_cursor = max(search_cursor, start_offset + 1)

                    chunk_id = f"{source_id}#chunk-{chunk_idx + 1:03d}"
                    active_sec = section if section is not None else self.extract_section_header(text, start_offset, file_type)
                    active_page = page_number if page_number is not None else self.extract_page_number(text, start_offset)
                    tok_cnt = self.count_tokens(cleaned)

                    meta = ChunkMetadata(
                        source_id=source_id,
                        section=active_sec,
                        page_number=active_page,
                        chunk_index=chunk_idx,
                        start_char=start_offset,
                        end_char=end_offset,
                        file_type=file_type,
                        char_count=len(cleaned),
                        token_count=tok_cnt,
                        strategy="recursive_semantic"
                    )

                    c = DocumentChunk(
                        chunk_id=chunk_id,
                        text=cleaned,
                        metadata=meta
                    )
                    chunks.append(c)
                    chunk_idx += 1

                overlap_text = current_chunk_text[-overlap:] if len(current_chunk_text) > overlap else ""
                current_chunk_text = (overlap_text + " " + fragment).strip()

        cleaned = current_chunk_text.strip()
        if cleaned:
            start_offset = text.find(cleaned, search_cursor)
            if start_offset == -1:
                start_offset = search_cursor
            end_offset = start_offset + len(cleaned)

            chunk_id = f"{source_id}#chunk-{chunk_idx + 1:03d}"
            active_sec = section if section is not None else self.extract_section_header(text, start_offset, file_type)
            active_page = page_number if page_number is not None else self.extract_page_number(text, start_offset)
            tok_cnt = self.count_tokens(cleaned)

            meta = ChunkMetadata(
                source_id=source_id,
                section=active_sec,
                page_number=active_page,
                chunk_index=chunk_idx,
                start_char=start_offset,
                end_char=end_offset,
                file_type=file_type,
                char_count=len(cleaned),
                token_count=tok_cnt,
                strategy="recursive_semantic"
            )

            c = DocumentChunk(
                chunk_id=chunk_id,
                text=cleaned,
                metadata=meta
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
        Run both chunking strategies across LoadedDocument objects with metadata tagging.
        """
        valid_docs = [d for d in docs if getattr(d, "status", "SUCCESS") == "SUCCESS"]

        fixed_chunks: List[DocumentChunk] = []
        semantic_chunks: List[DocumentChunk] = []

        for doc in valid_docs:
            src_id = getattr(doc, "source_id", "doc")
            ftype = getattr(doc, "file_type", "txt")
            txt = getattr(doc, "text_content", "")

            f_c = self.chunk_fixed_window(txt, chunk_size, overlap, source_id=src_id, file_type=ftype)
            s_c = self.chunk_recursive_semantic(txt, chunk_size, overlap, source_id=src_id, file_type=ftype)
            fixed_chunks.extend(f_c)
            semantic_chunks.extend(s_c)

        def compute_stats(chunks: List[DocumentChunk], name: str) -> Dict[str, Any]:
            if not chunks:
                return {"name": name, "total_chunks": 0}

            char_lens = [c.char_count for c in chunks]
            token_lens = [c.token_count for c in chunks]

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

