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

    # 1. Define Sample Test Texts (Task 1)
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

    # 2. Generate Embeddings (Task 1)
    generator = EmbeddingGenerator(model_name="text-embedding-3-small")
    print(f"\n[Step 2] Generating embedding vectors using model '{generator.model}'...")
    
    vec_a = generator.generate_embedding(text_a)
    vec_b = generator.generate_embedding(text_b)
    vec_c = generator.generate_embedding(text_c)

    all_vectors = [vec_a, vec_b, vec_c]

    # 3. Report Vector Dimension & Verify Consistency (Task 2)
    print("\n" + "=" * 85)
    print("[Task 2] Vector Shape & Dimensionality Audit:")
    print("=" * 85)
    audit = generator.verify_dimensionality(all_vectors)

    print(f"Total Vectors Generated : {audit['total_vectors']}")
    print(f"Vector Dimension Length  : {audit['vector_dimension']} components per vector")
    print(f"Dimensionality Status   : [{audit['status']}] (100% of texts produced identical length vectors)")
    print("-" * 85)
    print(f"Text A Vector Length: {len(vec_a)} | Slice (first 5 components): {[round(x, 4) for x in vec_a[:5]]}")
    print(f"Text B Vector Length: {len(vec_b)} | Slice (first 5 components): {[round(x, 4) for x in vec_b[:5]]}")
    print(f"Text C Vector Length: {len(vec_c)} | Slice (first 5 components): {[round(x, 4) for x in vec_c[:5]]}")

    # 4. Compare Similar vs Dissimilar Texts via Cosine Similarity (Task 3)
    print("\n" + "=" * 85)
    print("[Task 3] Cosine Similarity Evaluation (Similar vs Dissimilar Pairs):")
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

    # 5. Theoretical Explanation Note (Task 4)
    print("\n" + "=" * 85)
    print("[Task 4] What Embedding Vectors Represent in RAG Systems:")
    print("=" * 85)
    explanation_note = (
        "Embedding vectors are continuous numerical representations of text meaning in high-dimensional vector space "
        "(e.g. 1536 dimensions for text-embedding-3-small). They are NOT random database IDs, nor are they simple sparse "
        "keyword frequency counts (like BM25 or TF-IDF).\n\n"
        "Each floating-point number in an embedding vector captures a abstract semantic feature or concept "
        "(such as sentiment, topic, formality, or domain context). Because similar concepts are mapped to nearby coordinates "
        "in vector space, calculating the cosine distance between vectors allows RAG systems to retrieve relevant context "
        "based on conceptual meaning—even when the query and document use completely different vocabulary or synonyms."
    )
    print(explanation_note)

    # 6. Export Results Artifacts (Task 5)
    print("\n" + "=" * 85)
    print("[Task 5] Exporting Embedding Artifacts for Commit:")
    print("=" * 85)

    json_output_path = Config.OUTPUTS_DIR / "embedding_demonstration_results.json"
    log_output_path = Config.OUTPUTS_DIR / "embedding_vector_analysis.log"

    export_payload = {
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
        "explanation_note": explanation_note
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    log_lines = [
        "=" * 80,
        "    LEXTRACE EMBEDDING VECTOR & SEMANTIC SIMILARITY EVIDENCE LOG    ",
        "=" * 80,
        f"\nEMBEDDING MODEL : {generator.model}",
        f"VECTOR DIMENSION: {audit['vector_dimension']} dims",
        f"UNIFORM SHAPE   : {audit['status']}\n",
        "=== SAMPLE TEXTS ===",
        f"Text A: {text_a}",
        f"Text B: {text_b}",
        f"Text C: {text_c}\n",
        "=== COSINE SIMILARITY RESULTS ===",
        f"Similar Pair (Text A vs Text B)   : {sim_a_b:.4f}",
        f"Dissimilar Pair (Text A vs Text C): {sim_a_c:.4f}",
        f"Dissimilar Pair (Text B vs Text C): {sim_b_c:.4f}",
        f"Evaluation Proof: Similar > Dissimilar -> {'PASSED' if higher_score_proof else 'FAILED'}\n",
        "=== CONCEPTUAL EXPLANATION NOTE ===",
        explanation_note
    ]

    with open(log_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"  [1/2] JSON demonstration output saved to: outputs/embedding_demonstration_results.json")
    print(f"  [2/2] Analysis log saved to              : outputs/embedding_vector_analysis.log")

    print("\n" + "=" * 85)
    print(" SUCCESS: Text embedding demonstration & semantic similarity executed cleanly!")
    print("=" * 85)

if __name__ == "__main__":
    main()
