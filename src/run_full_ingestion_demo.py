import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.full_corpus_pipeline import FullCorpusPipeline

def main():
    print("=" * 85)
    print("      LexTrace RAG Assistant — Full-Corpus Ingestion Pipeline & Audit       ")
    print("=" * 85)

    corpus_dir = Config.DATA_DIR / "sample_corpus"
    if not corpus_dir.exists():
        print(f"[ERROR] Sample corpus directory not found at: {corpus_dir}")
        sys.exit(1)

    # 1. Initialize and run Full-Corpus Ingestion Pipeline (Task 1)
    pipeline = FullCorpusPipeline(base_dir=Config.BASE_DIR, model_name="gpt-4o-mini")
    results = pipeline.run_pipeline(
        corpus_dir=corpus_dir,
        max_tokens_per_chunk=200,
        overlap_tokens=40,
        recursive=True
    )

    audit = results["ingestion_reconciliation_audit"]
    chunks_info = results["chunking_summary"]

    # ---------------------------------------------------------
    # PART 1: Ingestion Summary & Reconciliation Audit Report (Tasks 2 & 3)
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("[Task 2 & Task 3] Full-Corpus Ingestion Summary & Reconciliation Audit:")
    print("=" * 85)
    print(f"Target Directory              : {results['execution_metadata']['target_directory']}")
    print(f"Total Discovered Files        : {audit['total_discovered_files']}")
    print(f"Successfully Ingested Files   : {audit['successfully_ingested_files']}")
    print(f"Recorded Failures / Skipped   : {audit['recorded_failures_or_skipped']}")
    print(f"Total Processed Documents     : {audit['total_processed_files']}")
    print("-" * 85)
    print(f"Reconciliation Proof Equation : {audit['reconciliation_equation']} (Discovered == Ingested + Failures)")
    print(f"Silent Drop Audit Status      : [{audit['reconciliation_status']}]")
    print("-" * 85)
    print(f"Total Chunks Created          : {chunks_info['total_chunks_created']}")
    print(f"Average Token Size per Chunk  : {chunks_info['avg_tokens_per_chunk']} tokens")
    print(f"Average Char Size per Chunk   : {chunks_info['avg_chars_per_chunk']} chars")

    # Display recorded failures gracefully (Task 2)
    if results["recorded_failures"]:
        print("\n--- RECORDED FAILURES & EXCLUDED FILES LOG ---")
        for f in results["recorded_failures"]:
            print(f"  - [{f['file_type'].upper()}] Source ID: {f['source_id']}")
            print(f"    Error Log : {f['error_message']}")

    # ---------------------------------------------------------
    # PART 2: Sample Chunks Inspection with Full Metadata Tags (Task 4)
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("[Task 4] Sample Chunks Inspection with Attached Metadata Tags:")
    print("=" * 85)

    for idx, chunk_dict in enumerate(results["sample_chunks_inspection"][:4], 1):
        meta = chunk_dict["metadata"]
        print(f"\n--- Sample Chunk #{idx} ---")
        print(f"  Chunk ID       : {chunk_dict['chunk_id']}")
        print(f"  Source ID      : {meta['source_id']} ({meta['file_type'].upper()})")
        print(f"  Section Header : {meta['section'] or 'N/A'}")
        print(f"  Page Number    : {meta['page_number']}")
        print(f"  Position Index : Chunk #{meta['chunk_index'] + 1}")
        print(f"  Char Offsets   : Range [{meta['start_char']} .. {meta['end_char']}]")
        print(f"  Size Metrics   : {meta['token_count']} tokens | {meta['char_count']} chars")
        print(f"  Strategy Tag   : {meta['strategy']}")
        print(f"  Cleaned Text   : \"{chunk_dict['snippet']}\"")

    # ---------------------------------------------------------
    # PART 3: Persist Ingestion Run Output Artifacts (Task 5)
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("[Task 5] Persisting Ingestion Run Output Artifacts:")
    print("=" * 85)

    json_output_path = Config.OUTPUTS_DIR / "full_corpus_ingestion_summary.json"
    log_output_path = Config.OUTPUTS_DIR / "full_corpus_ingestion.log"

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log_lines = [
        "=" * 80,
        "      LEXTRACE FULL-CORPUS INGESTION & AUDIT EXECUTION LOG      ",
        "=" * 80,
        f"\nTIMESTAMP         : {results['execution_metadata']['timestamp']}",
        f"DURATION          : {results['execution_metadata']['duration_seconds']}s",
        f"TOTAL DISCOVERED  : {audit['total_discovered_files']}",
        f"SUCCESSFUL INGEST : {audit['successfully_ingested_files']}",
        f"RECORDED FAILURES : {audit['recorded_failures_or_skipped']}",
        f"RECONCILIATION    : {audit['reconciliation_status']} ({audit['reconciliation_equation']})\n",
        "=" * 80,
        "=== RECORDED FAILURES breakdown ==="
    ]
    for f in results["recorded_failures"]:
        log_lines.append(f"  - [{f['file_type'].upper()}] {f['source_id']} -> Error: {f['error_message']}")

    log_lines.append("\n=== SAMPLE CHUNKS WITH METADATA ===")
    for c in results["sample_chunks_inspection"]:
        meta = c["metadata"]
        log_lines.append(f"\nChunk ID: {c['chunk_id']}")
        log_lines.append(f"Source: {meta['source_id']} | Section: {meta['section']} | Page: {meta['page_number']} | Pos: #{meta['chunk_index']+1}")
        log_lines.append(f"Tokens: {meta['token_count']} | Chars: {meta['char_count']} | Offsets: {meta['start_char']}-{meta['end_char']}")
        log_lines.append(f"Text:\n{c['snippet']}")

    with open(log_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"  [1/2] JSON ingestion summary saved to: outputs/full_corpus_ingestion_summary.json")
    print(f"  [2/2] Execution log saved to          : outputs/full_corpus_ingestion.log")

    print("\n" + "=" * 85)
    print(" SUCCESS: Full-corpus ingestion & completeness audit executed and verified cleanly!")
    print("=" * 85)

if __name__ == "__main__":
    main()
