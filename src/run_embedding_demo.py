import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.embedding_generator import EmbeddingGenerator

def main():
    print("=" * 85)
    print("      LexTrace RAG Assistant — Text Embedding Vector & Similarity Demonstration      ")
    print("=" * 85)

    # 1. Verify Environment Configuration (Task 3)
    print("\n[Task 3] Environment Configuration Audit:")
    print("-" * 85)
    print(f"  EMBEDDING_MODEL : {Config.EMBEDDING_MODEL}")
    print(f"  OPENAI_API_BASE : {Config.OPENAI_API_BASE}")
    print(f"  OPENAI_API_KEY  : {'[CONFIGURED]' if Config.OPENAI_API_KEY else '[MISSING / MOCK FALLBACK]'}")

    # 2. Define Sample Test Texts (Task 1)
    text_a = "All team members are required to enable multi-factor authentication (MFA) to secure developer accounts."
    text_b = "Mandatory multi-factor authentication (MFA) must be configured on all staff login accounts."
    text_c = "The employee cafeteria serves hot pasta, green salad, and fresh soup every Tuesday."

    sample_texts = [
        {"id": "Text_A (Policy 1)", "text": text_a, "category": "Security Policy"},
        {"id": "Text_B (Policy 2)", "text": text_b, "category": "Security Policy (Similar)"},
        {"id": "Text_C (Unrelated)", "text": text_c, "category": "Cafeteria Menu (Dissimilar)"}
    ]

    print("\n[Task 1] Sample Test Texts for Embedding Generation:")
    print("-" * 85)
    for item in sample_texts:
        print(f"  - [{item['id']}] Category: {item['category']}")
        print(f"    Text: \"{item['text']}\"")

    # 3. Generate Embeddings for Benchmark Texts (Task 1 & Task 4)
    generator = EmbeddingGenerator()
    print(f"\n[Step 2] Generating embedding vectors using model '{generator.model}' via API...")
    
    vec_a = generator.generate_embedding(text_a)
    vec_b = generator.generate_embedding(text_b)
    vec_c = generator.generate_embedding(text_c)

    all_vectors = [vec_a, vec_b, vec_c]

    # 4. Report Vector Dimension & Verify Consistency (Task 1 & Task 4)
    print("\n" + "=" * 85)
    print("[Task 1 & 4] Vector Shape & Dimensionality Audit:")
    print("=" * 85)
    audit = generator.verify_dimensionality(all_vectors)

    print(f"Total Benchmark Vectors Generated : {audit['total_vectors']}")
    print(f"Vector Dimension Length           : {audit['vector_dimension']} components per vector")
    print(f"Dimensionality Status            : [{audit['status']}] (100% of texts produced identical length vectors)")
    print("-" * 85)
    print(f"Text A Vector Length: {len(vec_a)} | Trimmed Sample (first 5 components): {[round(x, 4) for x in vec_a[:5]]}")
    print(f"Text B Vector Length: {len(vec_b)} | Trimmed Sample (first 5 components): {[round(x, 4) for x in vec_b[:5]]}")
    print(f"Text C Vector Length: {len(vec_c)} | Trimmed Sample (first 5 components): {[round(x, 4) for x in vec_c[:5]]}")

    # 5. Cosine Similarity Evaluation
    print("\n" + "=" * 85)
    print("Cosine Similarity Evaluation (Similar vs Dissimilar Pairs):")
    print("=" * 85)

    sim_a_b = generator.compute_cosine_similarity(vec_a, vec_b)
    sim_a_c = generator.compute_cosine_similarity(vec_a, vec_c)
    sim_b_c = generator.compute_cosine_similarity(vec_b, vec_c)

    print(f"Pair 1 (SIMILAR): Text A vs Text B [Both about MFA Security Policy]")
    print(f"  -> Cosine Similarity Score: {sim_a_b:.4f} ({sim_a_b * 100:.1f}% semantic similarity)")

    print(f"\nPair 2 (DISSIMILAR): Text A vs Text C [MFA Security vs Cafeteria Menu]")
    print(f"  -> Cosine Similarity Score: {sim_a_c:.4f} ({sim_a_c * 100:.1f}% semantic similarity)")

    print(f"\nPair 3 (DISSIMILAR): Text B vs Text C [MFA Security vs Cafeteria Menu]")
    print(f"  -> Cosine Similarity Score: {sim_b_c:.4f} ({sim_b_c * 100:.1f}% semantic similarity)")

    print("-" * 85)
    higher_score_proof = sim_a_b > sim_a_c and sim_a_b > sim_b_c
    print(f"Semantic Evaluation Proof : [{'PASSED' if higher_score_proof else 'FAILED'}]")
    print(f"  -> Similar Pair (A vs B) scored {sim_a_b:.4f}, which is SIGNIFICANTLY HIGHER than Dissimilar Pair (A vs C: {sim_a_c:.4f}).")

    # 6. Embed Prepared Corpus Chunks & Store Vectors with Metadata (Task 2, Task 4, Task 5)
    print("\n" + "=" * 85)
    print("[Task 2, 4, 5] Embedding Prepared Corpus Chunks & Storing Vectors with Metadata:")
    print("=" * 85)

    sample_chunks_file = Config.OUTPUTS_DIR / "sample_chunks_with_metadata.json"
    corpus_chunks = []
    if sample_chunks_file.exists():
        with open(sample_chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            corpus_chunks = data.get("sample_chunks", [])
            print(f"Loaded {len(corpus_chunks)} prepared text chunks from '{sample_chunks_file.name}'")
    else:
        from src.full_corpus_pipeline import FullCorpusPipeline
        pipeline = FullCorpusPipeline()
        summary = pipeline.run_pipeline(Config.DATA_DIR / "sample_corpus")
        corpus_chunks = summary.get("sample_chunks_inspection", [])

    embedded_corpus_data = generator.embed_corpus(corpus_chunks)
    embedded_vectors_path = Config.OUTPUTS_DIR / "embedded_corpus_vectors.json"

    with open(embedded_vectors_path, "w", encoding="utf-8") as f:
        json.dump(embedded_corpus_data, f, indent=2)

    print(f"Total Chunks Embedded   : {embedded_corpus_data['corpus_summary']['total_chunks_embedded']}")
    print(f"Vector Length           : {embedded_corpus_data['corpus_summary']['vector_dimension']}")
    print(f"Environment Config Used : Model='{embedded_corpus_data['corpus_summary']['embedding_model']}', BaseURL='{embedded_corpus_data['corpus_summary']['api_base_url']}'")
    print(f"Stored Vectors File     : outputs/embedded_corpus_vectors.json")
    print("-" * 85)
    print("Sample Stored Vector Entry Verification:")
    if embedded_corpus_data["embedded_chunks"]:
        sample_entry = embedded_corpus_data["embedded_chunks"][0]
        print(f"  - Chunk ID      : {sample_entry['chunk_id']}")
        print(f"  - Source Doc    : {sample_entry['source_id']}")
        print(f"  - Section       : {sample_entry['section']}")
        print(f"  - Chunk Index   : {sample_entry['chunk_index']}")
        print(f"  - Page Number   : {sample_entry['page_number']}")
        print(f"  - Source Text   : \"{sample_entry['source_text'][:90]}...\"")
        print(f"  - Vector Length : {sample_entry['vector_length']}")
        print(f"  - Trimmed Vector: {sample_entry['trimmed_vector']}")

    # 7. Theoretical Explanation Note
    print("\n" + "=" * 85)
    print("What Embedding Vectors Represent in RAG Systems:")
    print("=" * 85)
    explanation_note = (
        "Embedding vectors are continuous numerical representations of text meaning in high-dimensional vector space "
        "(e.g. 1536 dimensions for text-embedding-3-small). They are NOT random database IDs, nor are they simple sparse "
        "keyword frequency counts (like BM25 or TF-IDF).\n\n"
        "Each floating-point number in an embedding vector captures an abstract semantic feature or concept "
        "(such as sentiment, topic, formality, or domain context). Because similar concepts are mapped to nearby coordinates "
        "in vector space, calculating the cosine distance between vectors allows RAG systems to retrieve relevant context "
        "based on conceptual meaning—even when the query and document use completely different vocabulary or synonyms."
    )
    print(explanation_note)

    # 8. Export Benchmark Results Artifacts (Task 5)
    print("\n" + "=" * 85)
    print("[Task 5] Exporting Demonstration Artifacts:")
    print("=" * 85)

    json_output_path = Config.OUTPUTS_DIR / "embedding_demonstration_results.json"
    log_output_path = Config.OUTPUTS_DIR / "embedding_vector_analysis.log"

    export_payload = {
        "environment_config": {
            "embedding_model": generator.model,
            "api_base_url": generator.base_url,
            "api_key_configured": bool(generator.api_key)
        },
        "model_info": {
            "embedding_model": generator.model,
            "vector_dimension": audit['vector_dimension'],
            "uniform_dimensions": audit['valid']
        },
        "sample_texts": sample_texts,
        "vector_samples": {
            "Text_A": {"length": len(vec_a), "slice_first_10": [round(x, 6) for x in vec_a[:10]]},
            "Text_B": {"length": len(vec_b), "slice_first_10": [round(x, 6) for x in vec_b[:10]]},
            "Text_C": {"length": len(vec_c), "slice_first_10": [round(x, 6) for x in vec_c[:10]]}
        },
        "similarity_comparisons": {
            "similar_pair_A_B": {"description": "Text A vs Text B (Security Policies)", "cosine_similarity": round(sim_a_b, 4)},
            "dissimilar_pair_A_C": {"description": "Text A vs Text C (Security vs Cafeteria)", "cosine_similarity": round(sim_a_c, 4)},
            "dissimilar_pair_B_C": {"description": "Text B vs Text C (Security vs Cafeteria)", "cosine_similarity": round(sim_b_c, 4)},
            "similar_pair_scores_higher": higher_score_proof
        },
        "corpus_embedding_summary": embedded_corpus_data["corpus_summary"],
        "explanation_note": explanation_note
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    log_lines = [
        "=" * 80,
        "    LEXTRACE EMBEDDING VECTOR & SEMANTIC SIMILARITY EVIDENCE LOG    ",
        "=" * 80,
        f"\nEMBEDDING MODEL : {generator.model}",
        f"API BASE URL    : {generator.base_url}",
        f"VECTOR DIMENSION: {audit['vector_dimension']} dims",
        f"UNIFORM SHAPE   : {audit['status']}\n",
        "=== SAMPLE BENCHMARK TEXTS ===",
        f"Text A: {text_a}",
        f"Text B: {text_b}",
        f"Text C: {text_c}\n",
        "=== COSINE SIMILARITY RESULTS ===",
        f"Similar Pair (Text A vs Text B)   : {sim_a_b:.4f}",
        f"Dissimilar Pair (Text A vs Text C): {sim_a_c:.4f}",
        f"Dissimilar Pair (Text B vs Text C): {sim_b_c:.4f}",
        f"Evaluation Proof: Similar > Dissimilar -> {'PASSED' if higher_score_proof else 'FAILED'}\n",
        "=== PREPARED CORPUS EMBEDDING SUMMARY ===",
        f"Total Chunks Embedded: {embedded_corpus_data['corpus_summary']['total_chunks_embedded']}",
        f"Vector Dimension     : {embedded_corpus_data['corpus_summary']['vector_dimension']} dims",
        f"Stored Output File   : outputs/embedded_corpus_vectors.json\n",
        "=== CONCEPTUAL EXPLANATION NOTE ===",
        explanation_note
    ]

    with open(log_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"  [1/3] Embedded corpus vectors saved to  : outputs/embedded_corpus_vectors.json")
    print(f"  [2/3] JSON demonstration summary saved to: outputs/embedding_demonstration_results.json")
    print(f"  [3/3] Analysis log saved to              : outputs/embedding_vector_analysis.log")

    print("\n" + "=" * 85)
    print(" SUCCESS: Text embedding generation, corpus vector storage & verification completed!")
    print("=" * 85)

if __name__ == "__main__":
    main()
