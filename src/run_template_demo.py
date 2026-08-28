"""
Prompt Templating & Placeholder Validation Demonstration Script
Demonstrates:
1. Task 1: Defining prompt templates with named placeholders ({context}, {question}, {role}, {domain})
2. Task 2: Injecting dynamic runtime values into templates
3. Task 3: Reusing templates across Feature 1 (Interactive RAG Chat) and Feature 2 (Batch CLI Audit)
4. Task 4: Strict separation of template files (prompts/) from application Python code
5. Task 5: Error handling for missing placeholders & saving example renders to outputs/
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.prompt_manager import PromptManager, PromptTemplate

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("LexTrace.TemplateDemo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def run_template_demo():
    output_log = Config.OUTPUTS_DIR / "prompt_template_renders.log"
    output_json = Config.OUTPUTS_DIR / "prompt_template_renders.json"
    logger = setup_logger(output_log)

    logger.info("=" * 85)
    logger.info("       LexTrace RAG Assistant - Prompt Templating & Placeholder Engine       ")
    logger.info("=" * 85)

    pm = PromptManager()
    render_records: List[Dict[str, Any]] = []

    # =========================================================================
    # DEMO 1: Inspecting Template Placeholders
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 1 & 4] DEMO 1: TEMPLATE DISCOVERY & PLACEHOLDER EXTRACTION")
    logger.info("Goal: Load templates from prompts/ directory and discover named placeholders.")
    logger.info("=" * 85)

    templates_to_inspect = ["system_prompt", "rag_query", "batch_audit"]
    for tname in templates_to_inspect:
        tObj = pm.get_template(tname)
        logger.info(f"  - Template Name: '{tObj.name}' (v{tObj.version})")
        logger.info(f"    File Location: {tObj.file_path.name}")
        logger.info(f"    Discovered Placeholders: {sorted(list(tObj.placeholders))}")

    # =========================================================================
    # DEMO 2 (Task 2 & 3): Feature 1 (Interactive RAG Chat Render)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 2 & 3] DEMO 2: FEATURE 1 (INTERACTIVE RAG CHAT RUNTIME INJECTION)")
    logger.info("Goal: Inject dynamic user query and context into shared RAG template.")
    logger.info("=" * 85)

    feat1_context = "LexTrace Remote Work Reimbursement Policy: Dual monitors, internet stipend $50/month."
    feat1_question = "What hardware items qualify for remote reimbursement?"
    feat1_format = "Provide a concise bulleted summary."

    rendered_f1 = pm.render_prompt(
        "rag_query",
        context=feat1_context,
        question=feat1_question,
        format_instructions=feat1_format
    )

    logger.info(f"Rendered Prompt for Feature 1 (Interactive Chat):\n{rendered_f1}")

    render_records.append({
        "feature": "Feature 1: Interactive RAG Chat",
        "template_used": "rag_query_template.txt",
        "injected_values": {
            "context": feat1_context,
            "question": feat1_question,
            "format_instructions": feat1_format
        },
        "rendered_prompt_output": rendered_f1
    })

    # =========================================================================
    # DEMO 3 (Task 2 & 3): Feature 2 (Batch Document Audit CLI Render)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 2 & 3] DEMO 3: FEATURE 2 (BATCH DOCUMENT AUDIT CLI RUNTIME INJECTION)")
    logger.info("Goal: Reuse shared batch_audit template across document batch processing.")
    logger.info("=" * 85)

    feat2_id = "DOC-2026-HR-004"
    feat2_title = "Executive Travel & Expense Reimbursement Policy"
    feat2_text = "All travel expense reports must be filed within 30 days of trip completion with receipts."
    feat2_criteria = "Verify 30-day receipt submission deadline."

    rendered_f2 = pm.render_prompt(
        "batch_audit",
        document_id=feat2_id,
        title=feat2_title,
        document_text=feat2_text,
        audit_criteria=feat2_criteria
    )

    logger.info(f"Rendered Prompt for Feature 2 (Batch Audit CLI):\n{rendered_f2}")

    render_records.append({
        "feature": "Feature 2: Batch Document Audit CLI",
        "template_used": "batch_audit_template.txt",
        "injected_values": {
            "document_id": feat2_id,
            "title": feat2_title,
            "document_text": feat2_text,
            "audit_criteria": feat2_criteria
        },
        "rendered_prompt_output": rendered_f2
    })

    # =========================================================================
    # DEMO 4: Missing Placeholder Error Validation
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 1] DEMO 4: MISSING PLACEHOLDER ERROR VALIDATION")
    logger.info("Goal: Verify that missing required placeholders trigger clean diagnostic errors.")
    logger.info("=" * 85)

    try:
        # Intentionally omit 'format_instructions'
        pm.render_prompt(
            "rag_query",
            context=feat1_context,
            question=feat1_question
        )
    except ValueError as ve:
        logger.info(f"  - Caught Expected Validation Error: {ve}")
        render_records.append({
            "feature": "Error Handling Validation",
            "template_used": "rag_query_template.txt",
            "omitted_variable": "format_instructions",
            "error_caught": str(ve)
        })

    # Save JSON comparison output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(render_records, f, indent=2)

    logger.info("\n" + "=" * 85)
    logger.info(f"SUCCESS: Prompt templating demo completed cleanly.")
    logger.info(f"  - Log output saved to:  {output_log}")
    logger.info(f"  - JSON renders saved to: {output_json}")
    logger.info("=" * 85)

if __name__ == "__main__":
    run_template_demo()
