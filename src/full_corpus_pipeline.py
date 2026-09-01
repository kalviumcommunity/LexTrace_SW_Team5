import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FullCorpusPipeline")

from src.config import Config
from src.document_loader import DocumentLoader, LoadedDocument
from src.document_chunker import DocumentChunker, DocumentChunk, ChunkTracebackEngine

class CompletenessAuditError(Exception):
    """Raised when the document count reconciliation audit fails."""
    pass

class FullCorpusPipeline:
    """
    End-to-End Full-Corpus Ingestion Pipeline with Completeness Reconciliation Auditing.
    Processes documents through 6 sequential stages:
      Stage 1: File Discovery
      Stage 2: Format Ingestion & Parser Boundary Isolation
      Stage 3: Text Cleaning & Normalization
      Stage 4: Token-Aware & Semantic Chunking
      Stage 5: Metadata Tagging & Lineage Preservation
      Stage 6: Completeness Reconciliation Audit & Verification
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None, model_name: str = "gpt-4o-mini"):
        self.base_dir = Path(base_dir).resolve() if base_dir else Config.BASE_DIR
        self.loader = DocumentLoader(base_dir=self.base_dir, clean_text=True)
        self.chunker = DocumentChunker(model_name=model_name)

    def run_pipeline(
        self,
        corpus_dir: Union[str, Path],
        max_tokens_per_chunk: int = 200,
        overlap_tokens: int = 40,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Run the end-to-end ingestion pipeline over an entire document directory.

        Args:
            corpus_dir: Target directory containing corpus documents.
            max_tokens_per_chunk: Maximum tokens per chunk.
            overlap_tokens: Overlap tokens between adjacent chunks.
            recursive: If True, recursively scan subdirectories.

        Returns:
            Dict containing ingestion metrics, audit status, failure logs, and sample chunks.
        """
        folder = Path(corpus_dir).resolve()
        start_time = datetime.utcnow()

        logger.info(f"=== Starting Full-Corpus Ingestion Pipeline on '{folder.name}' ===")

        # ---------------------------------------------------------
        # STAGE 1: File Discovery (Task 1)
        # ---------------------------------------------------------
        pattern = "**/*" if recursive else "*"
        all_discovered_paths = [p for p in folder.glob(pattern) if p.is_file()]
        total_discovered = len(all_discovered_paths)

        logger.info(f"[Stage 1: File Discovery] Discovered {total_discovered} total file(s) in corpus directory.")

        # ---------------------------------------------------------
        # STAGE 2 & 3: Multi-Format Ingestion & Text Cleaning (Task 1)
        # ---------------------------------------------------------
        loaded_docs: List[LoadedDocument] = self.loader.load_directory(folder, recursive=recursive)

        successful_docs = [d for d in loaded_docs if d.status == "SUCCESS"]
        failed_docs = [d for d in loaded_docs if d.status == "FAILED"]

        logger.info(f"[Stage 2 & 3: Ingestion & Cleaning] Loaded {len(successful_docs)} successful, {len(failed_docs)} failed file(s).")

        # ---------------------------------------------------------
        # STAGE 4 & 5: Chunking & Metadata Tagging (Task 1 & Task 4)
        # ---------------------------------------------------------
        all_chunks: List[DocumentChunk] = []

        for doc in successful_docs:
            chunks = self.chunker.chunk_token_aware(
                text=doc.text_content,
                max_tokens=max_tokens_per_chunk,
                overlap_tokens=overlap_tokens,
                source_id=doc.source_id,
                file_type=doc.file_type
            )
            all_chunks.extend(chunks)

        logger.info(f"[Stage 4 & 5: Chunking & Tagging] Generated {len(all_chunks)} chunks with standardized metadata.")

        # ---------------------------------------------------------
        # STAGE 6: Completeness Reconciliation Audit (Task 3)
        # ---------------------------------------------------------
        total_processed_documents = len(loaded_docs)
        successful_count = len(successful_docs)
        failed_count = len(failed_docs)

        # Reconciliation Equation: Discovered == Successful + Recorded Failures
        reconciliation_check = (total_discovered == (successful_count + failed_count))

        if not reconciliation_check:
            msg = (
                f"AUDIT FAILURE: Document count mismatch! "
                f"Discovered ({total_discovered}) != Ingested ({successful_count}) + Failed ({failed_count})"
            )
            logger.error(msg)
            raise CompletenessAuditError(msg)

        logger.info(f"[Stage 6: Completeness Audit] [PASSED] Reconciliation check verified: {total_discovered} = {successful_count} + {failed_count}.")

        # ---------------------------------------------------------
        # Sample Chunk Traceability Verification (Task 4)
        # ---------------------------------------------------------
        sample_tracebacks = []
        if all_chunks:
            # Select representative sample chunks across different source files
            sample_indices = [0, len(all_chunks) // 2, len(all_chunks) - 1]
            for idx in set(sample_indices):
                chunk = all_chunks[idx]
                target_doc = next((d for d in successful_docs if d.source_id == chunk.metadata.source_id), None)
                if target_doc:
                    trace_info = ChunkTracebackEngine.trace_chunk(chunk, target_doc, context_window=80)
                    sample_tracebacks.append(trace_info)

        end_time = datetime.utcnow()
        duration_sec = round((end_time - start_time).total_seconds(), 3)

        # ---------------------------------------------------------
        # Format Ingestion Summary Report (Task 2 & Task 4)
        # ---------------------------------------------------------
        summary_payload = {
            "execution_metadata": {
                "timestamp": start_time.isoformat() + "Z",
                "duration_seconds": duration_sec,
                "target_directory": str(folder),
                "model_name": self.chunker.model_name
            },
            "ingestion_reconciliation_audit": {
                "total_discovered_files": total_discovered,
                "successfully_ingested_files": successful_count,
                "recorded_failures_or_skipped": failed_count,
                "total_processed_files": total_processed_documents,
                "reconciliation_equation": f"{total_discovered} == {successful_count} + {failed_count}",
                "reconciliation_status": "VERIFIED_NO_SILENT_DROPS" if reconciliation_check else "FAILED"
            },
            "chunking_summary": {
                "total_chunks_created": len(all_chunks),
                "avg_tokens_per_chunk": round(sum(c.token_count for c in all_chunks) / len(all_chunks), 1) if all_chunks else 0,
                "avg_chars_per_chunk": round(sum(c.char_count for c in all_chunks) / len(all_chunks), 1) if all_chunks else 0,
                "max_tokens_per_chunk_setting": max_tokens_per_chunk,
                "overlap_tokens_setting": overlap_tokens
            },
            "recorded_failures": [
                {
                    "source_id": d.source_id,
                    "file_type": d.file_type,
                    "error_message": d.error_message
                }
                for d in failed_docs
            ],
            "sample_chunks_inspection": [c.to_dict() for c in all_chunks[:5]],
            "traceability_verifications": sample_tracebacks
        }

        return summary_payload
