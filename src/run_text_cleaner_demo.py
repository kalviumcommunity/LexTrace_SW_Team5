import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.text_cleaner import TextCleaner
from src.document_loader import DocumentLoader

def main():
    print("=" * 80)
    print("       LexTrace Internal RAG Assistant - Text Cleaning Pipeline Demo       ")
    print("=" * 80)

    corpus_dir = Config.DATA_DIR / "sample_corpus"
    if not corpus_dir.exists():
        print(f"[ERROR] Sample corpus directory not found at: {corpus_dir}")
        sys.exit(1)

    cleaner = TextCleaner()

    # ---------------------------------------------------------
    # PART 1: Detailed Before / After Demonstration (Task 4)
    # ---------------------------------------------------------
    print("\n[PART 1] Individual Document Before & After Cleaning Demonstration:")
    noisy_file = corpus_dir / "noisy_policy_document.txt"
    if noisy_file.exists():
        raw_text = noisy_file.read_text(encoding="utf-8")
        cleaned_text = cleaner.clean_text(raw_text)
        stats = cleaner.compare(raw_text, cleaned_text)

        print("\n" + "-" * 75)
        print(f"FILE: {noisy_file.name}")
        print("-" * 75)
        print(">>> RAW EXTRACTED TEXT (BEFORE CLEANING):")
        print("-" * 40)
        print(raw_text)
        print("-" * 40)

        print("\n>>> CLEANED RETRIEVAL-READY TEXT (AFTER CLEANING):")
        print("-" * 40)
        print(cleaned_text)
        print("-" * 40)

        print("\n>>> CLEANING STATS & ARTIFACT RECOVERY:")
        print(f"  - Raw Character Count    : {stats['raw_char_count']:,} chars")
        print(f"  - Cleaned Character Count: {stats['cleaned_char_count']:,} chars")
        print(f"  - Noise Reduced          : {stats['char_reduction_count']} chars ({stats['char_reduction_pct']}%)")
        print("  - Artifacts Stripped / Repaired:")
        for artifact in stats['artifacts_detected']:
            print(f"     [CLEANED] {artifact}")


    # ---------------------------------------------------------
    # PART 2: Consistent Application Across Whole Corpus (Task 3)
    # ---------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("[PART 2] Consistent Cleaning Pipeline Applied Across Entire Corpus:")
    print("=" * 80)

    # Ingest corpus both raw and cleaned to produce side-by-side evidence
    loader_raw = DocumentLoader(base_dir=Config.BASE_DIR, clean_text=False)
    loader_cleaned = DocumentLoader(base_dir=Config.BASE_DIR, clean_text=True)

    raw_docs = loader_raw.load_directory(corpus_dir)
    cleaned_docs = loader_cleaned.load_directory(corpus_dir)

    print(f"\nDiscovered and processed {len(cleaned_docs)} documents across PDF, HTML, MD, TXT.")
    print("-" * 80)
    print(f"{'Source ID':<45} | {'Raw Chars':<10} | {'Clean Chars':<11} | {'Noise Reduction'}")
    print("-" * 80)

    corpus_summary = []
    log_output_lines = []
    log_output_lines.append("=" * 80)
    log_output_lines.append("         LEXTRACE TEXT CLEANING PIPELINE BEFORE/AFTER EVIDENCE LOG         ")
    log_output_lines.append("=" * 80 + "\n")

    for raw_doc, clean_doc in zip(raw_docs, cleaned_docs):
        if clean_doc.status == "SUCCESS":
            stats = cleaner.compare(raw_doc.text_content, clean_doc.text_content)
            print(f"{clean_doc.source_id:<45} | {stats['raw_char_count']:<10,} | {stats['cleaned_char_count']:<11,} | {stats['char_reduction_pct']}% reduction")
            
            doc_record = {
                "source_id": clean_doc.source_id,
                "file_type": clean_doc.file_type,
                "raw_char_count": stats['raw_char_count'],
                "cleaned_char_count": stats['cleaned_char_count'],
                "char_reduction_pct": stats['char_reduction_pct'],
                "artifacts_repaired": stats['artifacts_detected'],
                "before_snippet": raw_doc.text_content[:200].replace("\n", " "),
                "after_snippet": clean_doc.text_content[:200].replace("\n", " ")
            }
            corpus_summary.append(doc_record)

            log_output_lines.append(f"DOCUMENT: {clean_doc.source_id}")
            log_output_lines.append(f"FORMAT  : {clean_doc.file_type.upper()}")
            log_output_lines.append(f"RAW CHARS: {stats['raw_char_count']} | CLEAN CHARS: {stats['cleaned_char_count']} | REDUCTION: {stats['char_reduction_pct']}%")
            log_output_lines.append("BEFORE:\n" + raw_doc.text_content)
            log_output_lines.append("\nAFTER:\n" + clean_doc.text_content)
            log_output_lines.append("\n" + "=" * 80 + "\n")
        else:
            print(f"{clean_doc.source_id:<45} | {'SKIPPED':<10} | {'SKIPPED':<11} | Failure: {clean_doc.error_message}")

    print("-" * 80)

    # ---------------------------------------------------------
    # PART 3: Persist Execution Outputs (Task 5)
    # ---------------------------------------------------------
    results_json = Config.OUTPUTS_DIR / "text_cleaning_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(corpus_summary, f, indent=2)

    log_file = Config.OUTPUTS_DIR / "text_cleaning_before_after.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_output_lines))

    print(f"\n[Step 3] Persisted cleaning results JSON to : outputs/text_cleaning_results.json")
    print(f"[Step 4] Persisted before/after evidence log to: outputs/text_cleaning_before_after.log")

    print("\n" + "=" * 80)
    print(" SUCCESS: Text cleaning pipeline executed consistently across corpus!")
    print("=" * 80)

if __name__ == "__main__":
    main()
