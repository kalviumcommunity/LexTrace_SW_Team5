import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.document_loader import DocumentLoader, LoadedDocument

def main():
    print("=" * 80)
    print("      LexTrace Internal RAG Assistant - Document Loader Demonstration     ")
    print("=" * 80)
    
    corpus_dir = Config.DATA_DIR / "sample_corpus"
    if not corpus_dir.exists():
        print(f"[ERROR] Sample corpus directory not found at: {corpus_dir}")
        sys.exit(1)

    print(f"\n[Step 1] Initializing DocumentLoader for target directory:")
    print(f"  - Target Directory: {corpus_dir}")

    loader = DocumentLoader(base_dir=Config.BASE_DIR)

    # 1. Discover and load corpus directory
    print("\n[Step 2] Ingesting Sample Corpus (PDF, HTML, Markdown, Plain Text)...")
    loaded_docs = loader.load_directory(corpus_dir)

    # 2. Add an explicit missing file test case to demonstrate Task 2 resilience
    missing_file_path = corpus_dir / "missing_audit_report.txt"
    print(f"\n[Step 3] Simulating loading of missing file: {missing_file_path.name}")
    missing_doc = loader.load_file(missing_file_path)
    loaded_docs.append(missing_doc)

    # 3. Output Intake Confirmation (Task 4)
    print("\n[Step 4] Confirming Document Intake & Outputting Sample Previews:")
    summary = loader.confirm_intake(loaded_docs, sample_chars=140)

    # 4. Detailed Format Breakdown Report
    print("\n[Step 5] Document Ingestion Summary Breakdown:")
    format_stats = {}
    for doc in loaded_docs:
        fmt = doc.file_type.upper()
        if fmt not in format_stats:
            format_stats[fmt] = {"success": 0, "failed": 0}
        if doc.status == "SUCCESS":
            format_stats[fmt]["success"] += 1
        else:
            format_stats[fmt]["failed"] += 1

    print("  Format Breakdown:")
    for fmt, counts in format_stats.items():
        print(f"   - Format {fmt:6s} | Success: {counts['success']} | Failed/Skipped: {counts['failed']}")

    print(f"\n  Total Characters Extracted : {summary['total_characters']:,} chars")
    print(f"  Total Words Extracted      : {summary['total_words']:,} words")
    
    # Save log output to outputs/document_loader_results.json
    output_json = Config.OUTPUTS_DIR / "document_loader_results.json"
    output_data = [doc.to_dict() for doc in loaded_docs]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[Step 6] Execution results persisted cleanly to: outputs/document_loader_results.json")

    print("\n" + "=" * 80)
    print(" SUCCESS: Multi-format document loader executed with complete fault tolerance!")
    print("=" * 80)

if __name__ == "__main__":
    main()
