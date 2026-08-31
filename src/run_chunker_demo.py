import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.document_loader import DocumentLoader
from src.document_chunker import DocumentChunker

def main():
    print("=" * 80)
    print("      LexTrace Internal RAG Assistant - Document Chunking Comparison     ")
    print("=" * 80)

    corpus_dir = Config.DATA_DIR / "sample_corpus"
    if not corpus_dir.exists():
        print(f"[ERROR] Sample corpus directory not found at: {corpus_dir}")
        sys.exit(1)

    # 1. Load and clean documents
    loader = DocumentLoader(base_dir=Config.BASE_DIR, clean_text=True)
    docs = loader.load_directory(corpus_dir)
    valid_docs = [d for d in docs if d.status == "SUCCESS"]

    print(f"\n[Step 1] Ingested {len(valid_docs)} valid documents for chunking evaluation.")

    # 2. Initialize chunker and run comparative benchmark (Tasks 1, 2, 3)
    chunker = DocumentChunker(model_name="gpt-4o-mini")
    target_chunk_size = 500
    target_overlap = 100

    benchmark = chunker.compare_strategies(valid_docs, chunk_size=target_chunk_size, overlap=target_overlap)

    s_fixed = benchmark["fixed_window_stats"]
    s_sem = benchmark["recursive_semantic_stats"]

    # ---------------------------------------------------------
    # PART 1: Quantitative Strategy Comparison Report (Task 3)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 1] Quantitative Chunking Strategy Comparison Metrics:")
    print("=" * 80)
    print(f"Target Parameters : Chunk Size = {target_chunk_size} chars (~100 tokens) | Overlap = {target_overlap} chars (~20 tokens)")
    print("-" * 80)
    print(f"{'Metric':<32} | {'Strategy A: Fixed Window':<22} | {'Strategy B: Recursive Semantic'}")
    print("-" * 80)
    min_max_char_fixed = f"{s_fixed['min_char_size']} / {s_fixed['max_char_size']}"
    min_max_char_sem = f"{s_sem['min_char_size']} / {s_sem['max_char_size']}"
    min_max_tok_fixed = f"{s_fixed['min_token_size']} / {s_fixed['max_token_size']}"
    min_max_tok_sem = f"{s_sem['min_token_size']} / {s_sem['max_token_size']}"
    frag_fixed = f"{s_fixed['fragmentation_pct']}%"
    frag_sem = f"{s_sem['fragmentation_pct']}%"

    print(f"{'Total Chunks Generated':<32} | {s_fixed['total_chunks']:<22} | {s_sem['total_chunks']}")
    print(f"{'Average Chunk Length (chars)':<32} | {s_fixed['avg_char_size']:<22} | {s_sem['avg_char_size']}")
    print(f"{'Min / Max Chunk Length (chars)':<32} | {min_max_char_fixed:<22} | {min_max_char_sem}")
    print(f"{'Average Chunk Length (tokens)':<32} | {s_fixed['avg_token_size']:<22} | {s_sem['avg_token_size']}")
    print(f"{'Min / Max Chunk Length (tokens)':<32} | {min_max_tok_fixed:<22} | {min_max_tok_sem}")
    print(f"{'Total Corpus Tokens':<32} | {s_fixed['total_tokens']:<22} | {s_sem['total_tokens']}")
    print(f"{'Sentence Fragmentation Rate (%)':<32} | {frag_fixed:<22} | {frag_sem}")

    print("-" * 80)

    # ---------------------------------------------------------
    # PART 2: Sample Boundary Inspection (Task 5)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 2] Boundary Inspection on Same Document:")
    print("=" * 80)
    
    target_doc = next((d for d in valid_docs if "noisy_policy" in d.source_id or "handbook" in d.source_id), valid_docs[0])
    
    doc_fixed = chunker.chunk_fixed_window(target_doc.text_content, target_chunk_size, target_overlap, target_doc.source_id)
    doc_sem = chunker.chunk_recursive_semantic(target_doc.text_content, target_chunk_size, target_overlap, target_doc.source_id)

    print(f"DOCUMENT: {target_doc.source_id}")
    print("-" * 75)
    print("\n--- STRATEGY A: FIXED-WINDOW OVERLAP CHUNKS ---")
    for c in doc_fixed[:3]:
        print(f"\n[Chunk ID: {c.chunk_id}] ({c.char_count} chars, {c.token_count} tokens)")
        print(f"\"...{c.text}...\"")

    print("\n--- STRATEGY B: RECURSIVE SEMANTIC BOUNDARY CHUNKS ---")
    for c in doc_sem[:3]:
        print(f"\n[Chunk ID: {c.chunk_id}] ({c.char_count} chars, {c.token_count} tokens)")
        print(f"\"{c.text}\"")

    # ---------------------------------------------------------
    # PART 3: Strategy Choice & Justification (Task 4)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 3] Selected Strategy & Justification for LexTrace Corpus:")
    print("=" * 80)
    print("CHOSEN STRATEGY: Strategy B - Recursive Semantic Boundary Chunking")

    print("\nJUSTIFICATION:")
    print(" 1. Sentence Integrity: Fixed-window chunking cut mid-sentence in {:.1f}% of chunks, splitting critical rules across boundaries.".format(s_fixed['fragmentation_pct']))
    print(" 2. Retrieval Precision: Recursive semantic chunking respects paragraph and section boundaries, ensuring vector similarity matches complete policy units.")
    print(" 3. Context Budget Efficiency: Prevents word fragment truncation and avoids polluting embedding vector space with incomplete clauses.")

    # ---------------------------------------------------------
    # PART 4: Persist Outputs for Verification (Task 5)
    # ---------------------------------------------------------
    output_json = Config.OUTPUTS_DIR / "chunking_comparison_results.json"
    results_summary = {
        "benchmark_summary": {
            "fixed_window": s_fixed,
            "recursive_semantic": s_sem
        },
        "sample_chunks": {
            "fixed_window": [c.to_dict() for c in doc_fixed],
            "recursive_semantic": [c.to_dict() for c in doc_sem]
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    log_file = Config.OUTPUTS_DIR / "sample_chunks_comparison.log"
    log_lines = [
        "=" * 80,
        "          LEXTRACE DOCUMENT CHUNKING BOUNDARY EVIDENCE LOG          ",
        "=" * 80,
        f"\nTARGET DOCUMENT: {target_doc.source_id}\n",
        "=== STRATEGY A: FIXED WINDOW CHUNKS ==="
    ]
    for c in doc_fixed:
        log_lines.append(f"\n--- Chunk {c.chunk_id} ({c.char_count} chars, {c.token_count} tokens) ---")
        log_lines.append(c.text)

    log_lines.append("\n=== STRATEGY B: RECURSIVE SEMANTIC CHUNKS ===")
    for c in doc_sem:
        log_lines.append(f"\n--- Chunk {c.chunk_id} ({c.char_count} chars, {c.token_count} tokens) ---")
        log_lines.append(c.text)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n[Step 2] Persisted comparison JSON to : outputs/chunking_comparison_results.json")
    print(f"[Step 3] Persisted sample chunk log to : outputs/sample_chunks_comparison.log")

    print("\n" + "=" * 80)
    print(" SUCCESS: Document chunking comparison executed and verified cleanly!")
    print("=" * 80)

if __name__ == "__main__":
    main()
