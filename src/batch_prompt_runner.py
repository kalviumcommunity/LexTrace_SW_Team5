"""
Feature 2: Batch Document Audit CLI Runner for LexTrace RAG Assistant
Reuses shared PromptManager templates to evaluate document records in batch.
Demonstrates multi-feature template reuse (Task 3 & 4).
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.llm_client import ChatCompletionClient
from src.prompt_manager import PromptManager

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("LexTrace.BatchRunner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def run_batch_audit_feature():
    logger = setup_logger()

    logger.info("=" * 80)
    logger.info("  LexTrace RAG Assistant - Feature 2: Batch Document Audit CLI Runner  ")
    logger.info("=" * 80)

    # 1. Load sample dataset
    sample_doc_path = Config.DATA_DIR / "sample_documents.json"
    if not sample_doc_path.exists():
        logger.error(f"Sample dataset not found at {sample_doc_path}")
        return

    records = json.loads(sample_doc_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(records)} document records from dataset.")

    # 2. Re-use shared PromptManager instance
    pm = PromptManager()

    # Render system prompt
    system_prompt = pm.render_prompt(
        "system_prompt",
        role="LexTrace Policy Audit Compliance Bot",
        domain="Document Schema & Content Quality Validation",
        constraints="Flag missing sections, ambiguous dates, or unverified claims."
    )

    client = ChatCompletionClient()
    audit_results: List[Dict[str, Any]] = []

    audit_criteria = (
        "Check if document includes: (1) Clear title, (2) Effective date, (3) Department contact, "
        "and (4) Itemized reimbursement thresholds."
    )

    logger.info("\nReusing 'batch_audit' template across batch document records...")

    for i, doc in enumerate(records, 1):
        doc_id = doc.get("id", f"DOC-{i:03d}")
        title = doc.get("title", "Untitled Document")
        text = doc.get("content", "")

        # Task 2 & 3: Inject dynamic runtime values into shared batch_audit template
        rendered_prompt = pm.render_prompt(
            "batch_audit",
            document_id=doc_id,
            title=title,
            document_text=text,
            audit_criteria=audit_criteria
        )

        logger.info(f"\n--- Batch Audit Target #{i}: [{doc_id}] '{title}' ---")
        logger.info(f"Rendered Prompt Excerpt:\n{rendered_prompt[:180]}...")

        res = client.create_chat_completion(
            user_message=rendered_prompt,
            system_message=system_prompt,
            temperature=0.0
        )

        audit_results.append({
            "document_id": doc_id,
            "title": title,
            "rendered_prompt": rendered_prompt,
            "audit_output": res.get("content", "")
        })

    logger.info("\n" + "=" * 80)
    logger.info(f"Batch audit evaluation completed successfully for {len(audit_results)} documents.")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_batch_audit_feature()
