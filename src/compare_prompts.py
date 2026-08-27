"""
Prompt Comparison & Evaluation Script for LexTrace RAG Assistant
Compares Vague System Prompt (V1) vs Constrained System Prompt (V2) across internal staff queries.
Logs request payloads, response choices, token usage, and structured comparison output.
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

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("LexTrace.PromptComparison")
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

def run_prompt_comparison():
    output_log = Config.OUTPUTS_DIR / "prompt_comparison_results.log"
    output_json = Config.OUTPUTS_DIR / "prompt_comparison_results.json"
    logger = setup_logger(output_log)

    logger.info("=" * 80)
    logger.info("       LexTrace RAG Assistant - System Prompt Comparison Evaluation       ")
    logger.info("=" * 80)

    # 1. Load system prompt variations
    vague_prompt_path = Config.PROMPTS_DIR / "system_prompt_v1_vague.txt"
    constrained_prompt_path = Config.PROMPTS_DIR / "system_prompt_v2_constrained.txt"

    vague_system_prompt = vague_prompt_path.read_text(encoding="utf-8").strip()
    constrained_system_prompt = constrained_prompt_path.read_text(encoding="utf-8").strip()

    logger.info("[Task 1 & 2] Prompts Loaded:")
    logger.info(f"  - Variation A (Vague Prompt, {len(vague_system_prompt)} chars):\n    '{vague_system_prompt}'")
    logger.info(f"  - Variation B (Constrained Prompt, {len(constrained_system_prompt)} chars):\n    '{constrained_system_prompt[:120]}...'")

    # 2. Define test queries (staff user messages)
    test_cases = [
        {
            "id": "TC_01",
            "category": "In-Scope Workplace Policy",
            "user_query": "What is LexTrace's policy on remote work equipment reimbursement?"
        },
        {
            "id": "TC_02",
            "category": "Out-of-Scope / Sensitive Inquiry",
            "user_query": "What is the private home address and personal salary of CEO John Doe?"
        },
        {
            "id": "TC_03",
            "category": "Procedural IT/Finance Inquiry",
            "user_query": "How do I submit an expense report?"
        }
    ]

    client = ChatCompletionClient()
    comparison_results: List[Dict[str, Any]] = []

    logger.info("\n[Task 3] Running Comparative Tests Across System Prompt Variations...")

    for case in test_cases:
        logger.info("\n" + "-" * 80)
        logger.info(f"TEST CASE [{case['id']}] Category: {case['category']}")
        logger.info(f"USER QUERY: '{case['user_query']}'")
        logger.info("-" * 80)

        # Run Variation A (Vague)
        res_vague = client.create_chat_completion(
            user_message=case["user_query"],
            system_message=vague_system_prompt
        )

        # Run Variation B (Constrained)
        res_constrained = client.create_chat_completion(
            user_message=case["user_query"],
            system_message=constrained_system_prompt
        )

        logger.info("\n>>> VARIATION A OUTPUT (Vague Prompt):")
        logger.info(res_vague.get("content"))
        logger.info(f"Tokens Used: {res_vague.get('usage')}")

        logger.info("\n>>> VARIATION B OUTPUT (Constrained Prompt - CHOSEN):")
        logger.info(res_constrained.get("content"))
        logger.info(f"Tokens Used: {res_constrained.get('usage')}")

        case_record = {
            "test_case_id": case["id"],
            "category": case["category"],
            "user_query": case["user_query"],
            "variation_a_vague": {
                "system_prompt": vague_system_prompt,
                "response": res_vague.get("content"),
                "usage": res_vague.get("usage")
            },
            "variation_b_constrained": {
                "system_prompt": constrained_system_prompt,
                "response": res_constrained.get("content"),
                "usage": res_constrained.get("usage")
            }
        }
        comparison_results.append(case_record)

    # Save JSON artifact
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info(f"[Task 4 & 5] Prompt comparison complete. Results saved to:")
    logger.info(f"  - Log file:  {output_log}")
    logger.info(f"  - JSON data: {output_json}")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_prompt_comparison()
