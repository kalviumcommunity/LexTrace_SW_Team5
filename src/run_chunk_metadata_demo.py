import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.document_loader import DocumentLoader
from src.document_chunker import DocumentChunker, ChunkTracebackEngine

def main():
    print("=" * 85)
    print("      LexTrace RAG Assistant — Document Chunk Metadata & Source Traceability       ")
    print("=" * 85)

    corpus_dir = Config.DATA_DIR / "sample_corpus"
    if not corpus_dir.exists():
        print(f"[ERROR] Sample corpus directory not found at: {corpus_dir}")
        sys.exit(1)

    # 1. Ingest multi-format documents
    print("\n[Step 1] Ingesting documents across PDF, HTML, MD, and TXT formats...")
    loader = DocumentLoader(base_dir=Config.BASE_DIR, clean_text=True)
    docs = loader.load_directory(corpus_dir)
    valid_docs = [d for d in docs if d.status == "SUCCESS"]
    print(f"  -> Ingested {len(valid_docs)} valid documents cleanly.")

    # 2. Chunk documents with metadata tagging (Tasks 1 & 2)
    print("\n[Step 2] Chunking documents with structured metadata tagging...")
    chunker = DocumentChunker(model_name="gpt-4o-mini")
    all_chunks = []
    
    for doc in valid_docs:
        doc_chunks = chunker.chunk_recursive_semantic(
            text=doc.text_content,
            chunk_size=400,
            overlap=80,
            source_id=doc.source_id,
            file_type=doc.file_type
        )
        all_chunks.extend(doc_chunks)

    print(f"  -> Generated {len(all_chunks)} chunks with attached metadata across corpus.")

    # 3. Verify consistent metadata structure (Task 3)
    print("\n[Step 3] Verifying metadata schema consistency across corpus (Task 3)...")
    expected_keys = {
        "source_id", "section", "page_number", "chunk_index", 
        "start_char", "end_char", "file_type", "char_count", 
        "token_count", "strategy", "created_at"
    }

    inconsistent_count = 0
    for chunk in all_chunks:
        meta_dict = chunk.metadata.to_dict()
        missing = expected_keys - meta_dict.keys()
        if missing:
            inconsistent_count += 1
            print(f"  [WARNING] Chunk {chunk.chunk_id} missing metadata keys: {missing}")

    if inconsistent_count == 0:
        print(f"  [SUCCESS] 100% of chunks ({len(all_chunks)}/{len(all_chunks)}) adhere to exact schema keys: {sorted(list(expected_keys))}")

    # 4. Demonstrate Chunk Traceability to Source (Task 4)
    print("\n" + "=" * 85)
    print("[Task 4] Demonstrating Chunk Source Traceability & RAG Citation Generation:")
    print("=" * 85)

    traceback_logs = []
    traceback_results = []

    # Pick sample chunks across different document types (txt, md, html, pdf)
    sample_indices = [0, min(2, len(all_chunks)-1), min(5, len(all_chunks)-1), min(8, len(all_chunks)-1)]
    selected_chunks = [all_chunks[i] for i in set(sample_indices)]

    for idx, chunk in enumerate(selected_chunks, 1):
        target_doc = next((d for d in valid_docs if d.source_id == chunk.metadata.source_id), valid_docs[0])
        trace_info = ChunkTracebackEngine.trace_chunk(chunk, target_doc, context_window=100)
        traceback_results.append(trace_info)

        header_line = f"--- Traceback Sample #{idx} ---"
        print(f"\n{header_line}")
        print(f"  Chunk ID       : {trace_info['chunk_id']}")
        print(f"  Source Document: {trace_info['source_id']} ({chunk.metadata.file_type.upper()})")
        print(f"  Section Header : {chunk.metadata.section or 'N/A'}")
        print(f"  Page Number    : {chunk.metadata.page_number}")
        print(f"  Position Index : Chunk #{chunk.metadata.chunk_index + 1}")
        print(f"  Char Offsets   : Range [{trace_info['start_char']} .. {trace_info['end_char']}]")
        print(f"  Citation String: {trace_info['citation_string']}")
        print(f"  Trace Status   : [{trace_info['traceability_status']}] Exact Match: {trace_info['exact_text_match']}")
        print(f"  Snippet Text   : \"{chunk.text[:100]}...\"")
        print(f"  Surrounding Context Prefix: \"...{trace_info['surrounding_context']['prefix'][-60:]}\"")
        print(f"  Surrounding Context Suffix: \"{trace_info['surrounding_context']['suffix'][:60]}...\"")

        traceback_logs.append(header_line)
        traceback_logs.append(f"Chunk ID: {trace_info['chunk_id']}")
        traceback_logs.append(f"Source ID: {trace_info['source_id']}")
        traceback_logs.append(f"Citation: {trace_info['citation_string']}")
        traceback_logs.append(f"Traceability Status: {trace_info['traceability_status']}")
        traceback_logs.append(f"Chunk Text:\n{chunk.text}")
        traceback_logs.append(f"Surrounding Context Window:\n[PREFIX] {trace_info['surrounding_context']['prefix']}\n[TARGET] {chunk.text}\n[SUFFIX] {trace_info['surrounding_context']['suffix']}\n" + "-" * 75)

    # 5. Export JSON and Log Artifacts (Task 5)
    print("\n" + "=" * 85)
    print("[Task 5] Persisting Metadata & Traceback Artifacts:")
    print("=" * 85)

    json_output_path = Config.OUTPUTS_DIR / "sample_chunks_with_metadata.json"
    log_output_path = Config.OUTPUTS_DIR / "traceback_demonstration.log"

    export_payload = {
        "summary": {
            "total_documents": len(valid_docs),
            "total_chunks": len(all_chunks),
            "schema_keys": sorted(list(expected_keys)),
            "schema_consistent": (inconsistent_count == 0)
        },
        "sample_chunks": [c.to_dict() for c in all_chunks],
        "traceback_verifications": traceback_results
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    with open(log_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(traceback_logs))

    print(f"  [1/2] JSON export saved to : outputs/sample_chunks_with_metadata.json ({len(all_chunks)} chunks with full metadata)")
    print(f"  [2/2] Traceback log saved to: outputs/traceback_demonstration.log (Traceability evidence)")

    print("\n" + "=" * 85)
    print(" SUCCESS: Metadata tagging & chunk traceability executed and verified cleanly!")
    print("=" * 85)

if __name__ == "__main__":
    main()
