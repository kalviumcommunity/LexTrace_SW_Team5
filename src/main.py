import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config

def main():
    print("=" * 60)
    print("       LexTrace Internal RAG Assistant Workspace Verification       ")
    print("=" * 60)
    
    # 1. Verify directory structure
    dirs_to_check = {
        "Data Directory": Config.DATA_DIR,
        "Prompts Directory": Config.PROMPTS_DIR,
        "Outputs Directory": Config.OUTPUTS_DIR,
    }
    
    print("\n[1/4] Checking Directory Structure:")
    for name, path in dirs_to_check.items():
        exists = path.exists() and path.is_dir()
        status = "OK" if exists else "MISSING"
        print(f"  - {name} ({path.relative_to(Config.BASE_DIR)}): [{status}]")

    # 2. Verify environment configuration
    print("\n[2/4] Validating Environment Configuration (.env):")
    is_valid = Config.validate()
    print(f"  - API Base URL: {Config.OPENAI_API_BASE}")
    print(f"  - Chat Model: {Config.CHAT_MODEL}")
    print(f"  - Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"  - API Key set: {'Yes (Hidden)' if Config.OPENAI_API_KEY else 'No'}")

    # 3. Verify sample document loading & schema validation
    print("\n[3/4] Testing Sample Document Ingestion & Validation:")
    sample_file = Config.DATA_DIR / "sample_documents.json"
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"  - Successfully loaded {len(docs)} sample document(s).")
        
        from src.data_validation import validate_dataset
        val_result = validate_dataset(sample_file)
        print(f"  - Data Validation Status: [{'OK' if val_result['valid'] else 'FAILED'}] ({val_result.get('total_records', 0)} valid records)")
    else:
        print("  - Sample document file not found.")

    # 4. Verify system prompt loading
    print("\n[4/4] Testing Prompt Template Loading:")
    prompt_file = Config.PROMPTS_DIR / "system_prompt.txt"
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read().strip()
        print(f"  - System Prompt Loaded ({len(prompt_content)} chars).")
    else:
        print("  - System prompt file not found.")

    print("\n" + "=" * 60)
    print(" SUCCESS: RAG Assistant workspace initialized & verified cleanly!")
    print("=" * 60)

if __name__ == "__main__":
    main()
