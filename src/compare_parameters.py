"""
LLM Parameter Experiments & Evaluation Script for LexTrace RAG Assistant
Conducts systematic parameter tuning tests across:
1. Temperature variation (Task 1: Grounded/Factual vs Creative/Varied)
2. max_tokens length capping (Task 2: Token limit enforcement & cost impact)
3. Additional parameters: top_p nucleus sampling & stop sequences (Task 3: Response control & truncation)

Saves detailed logs to outputs/parameter_experiments_results.log and structured JSON to outputs/parameter_experiments_results.json.
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
    logger = logging.getLogger("LexTrace.ParameterExperiments")
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

def run_parameter_experiments():
    output_log = Config.OUTPUTS_DIR / "parameter_experiments_results.log"
    output_json = Config.OUTPUTS_DIR / "parameter_experiments_results.json"
    logger = setup_logger(output_log)

    logger.info("=" * 85)
    logger.info("       LexTrace RAG Assistant - LLM Hyperparameter Experiment Suite       ")
    logger.info("=" * 85)

    system_prompt = (
        "You are LexTrace RAG Assistant, an internal AI assistant designed to answer staff questions "
        "accurately using retrieved internal documents from the knowledge base. Be factual, concise, "
        "and stay grounded in official policy."
    )
    prompt_path = Config.PROMPTS_DIR / "system_prompt_v2_constrained.txt"
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    user_query = "What is LexTrace's policy on remote work equipment reimbursement?"

    client = ChatCompletionClient()
    all_experiment_results: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "user_query": user_query,
        "task_1_temperature_experiments": [],
        "task_2_max_tokens_experiments": [],
        "task_3_top_p_experiments": [],
        "task_3_stop_sequence_experiments": []
    }

    # =========================================================================
    # TASK 1: Vary Temperature and Show the Effect
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 1] EXPERIMENT 1: VARYING TEMPERATURE (0.0 vs 0.7 vs 1.5)")
    logger.info("Goal: Demonstrate deterministic, stable/factual output at low temp vs creative/varied output at high temp.")
    logger.info("=" * 85)

    temperatures_to_test = [0.0, 0.7, 1.5]
    runs_per_temp = 3

    for temp in temperatures_to_test:
        logger.info(f"\n--- Testing Temperature = {temp} (Running {runs_per_temp} iterations) ---")
        temp_runs = []
        for run_idx in range(1, runs_per_temp + 1):
            res = client.create_chat_completion(
                user_message=user_query,
                system_message=system_prompt,
                temperature=temp,
                seed=100 + run_idx if temp > 0.0 else None
            )
            content = res.get("content", "")
            usage = res.get("usage", {})
            logger.info(f"  Run #{run_idx} Output ({len(content)} chars, {usage.get('completion_tokens', 0)} tokens):")
            logger.info(f"    \"{content.strip()}\"")
            temp_runs.append({
                "run_id": run_idx,
                "output": content,
                "usage": usage,
                "finish_reason": res.get("finish_reason")
            })

        # Calculate stability / consistency metric across runs
        outputs_set = set(r["output"] for r in temp_runs)
        is_identical = len(outputs_set) == 1
        stability_summary = "STABLE / DETERMINISTIC (100% Identical)" if is_identical else f"VARIED / CREATIVE ({len(outputs_set)} unique variations)"
        logger.info(f"  => Temperature {temp} Result: {stability_summary}")

        all_experiment_results["task_1_temperature_experiments"].append({
            "temperature": temp,
            "stability_summary": stability_summary,
            "runs": temp_runs
        })

    # =========================================================================
    # TASK 2: Cap Length with max_tokens
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 2] EXPERIMENT 2: CAPPING LENGTH WITH max_tokens (15 vs 35 vs Uncapped/150)")
    logger.info("Goal: Show strict token limits capping output length, setting finish_reason='length', and controlling cost.")
    logger.info("=" * 85)

    token_caps = [15, 35, 150]

    for cap in token_caps:
        logger.info(f"\n--- Testing max_tokens = {cap} ---")
        res = client.create_chat_completion(
            user_message=user_query,
            system_message=system_prompt,
            temperature=0.0,
            max_tokens=cap
        )
        content = res.get("content", "")
        usage = res.get("usage", {})
        finish_reason = res.get("finish_reason", "stop")

        logger.info(f"  Output ({len(content)} chars, {usage.get('completion_tokens', 0)} completion tokens):")
        logger.info(f"    \"{content.strip()}\"")
        logger.info(f"  - Finish Reason: '{finish_reason}'")
        logger.info(f"  - Total Tokens Billed: {usage.get('total_tokens', 0)}")

        all_experiment_results["task_2_max_tokens_experiments"].append({
            "max_tokens_setting": cap,
            "output": content,
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "finish_reason": finish_reason,
            "is_truncated": finish_reason == "length"
        })

    # =========================================================================
    # TASK 3: Test Additional Parameters (top_p Nucleus Sampling & stop Sequences)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 3] EXPERIMENT 3A: TESTING top_p NUCLEUS SAMPLING (0.1 vs 0.95)")
    logger.info("Goal: Demonstrate how top_p restricts candidate token pool to top probability mass.")
    logger.info("=" * 85)

    top_p_values = [0.1, 0.95]
    for tp in top_p_values:
        logger.info(f"\n--- Testing top_p = {tp} ---")
        res = client.create_chat_completion(
            user_message=user_query,
            system_message=system_prompt,
            temperature=0.7,
            top_p=tp
        )
        content = res.get("content", "")
        usage = res.get("usage", {})
        logger.info(f"  Output (top_p={tp}):")
        logger.info(f"    \"{content.strip()}\"")
        logger.info(f"  - Tokens Used: {usage}")

        all_experiment_results["task_3_top_p_experiments"].append({
            "top_p_setting": tp,
            "output": content,
            "usage": usage
        })

    logger.info("\n" + "=" * 85)
    logger.info("[TASK 3] EXPERIMENT 3B: TESTING stop SEQUENCES (stop=['Note:'] vs None)")
    logger.info("Goal: Show stop parameter halting generation immediately when encountering a trigger token.")
    logger.info("=" * 85)

    stop_cases = [
        {"name": "No Stop Sequence", "stop": None},
        {"name": "Stop on 'Note:'", "stop": ["Note:"]},
        {"name": "Stop on Linebreak '\\n'", "stop": ["\n"]}
    ]

    for sc in stop_cases:
        logger.info(f"\n--- Testing Stop Case: {sc['name']} ---")
        res = client.create_chat_completion(
            user_message=user_query,
            system_message=system_prompt,
            temperature=0.0,
            stop=sc["stop"]
        )
        content = res.get("content", "")
        usage = res.get("usage", {})
        finish_reason = res.get("finish_reason", "stop")

        logger.info(f"  Output:")
        logger.info(f"    \"{content.strip()}\"")
        logger.info(f"  - Finish Reason: '{finish_reason}'")
        logger.info(f"  - Tokens Generated: {usage.get('completion_tokens', 0)}")

        all_experiment_results["task_3_stop_sequence_experiments"].append({
            "case_name": sc["name"],
            "stop_parameter": sc["stop"],
            "output": content,
            "finish_reason": finish_reason,
            "completion_tokens": usage.get("completion_tokens", 0)
        })

    # Save JSON comparison output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_experiment_results, f, indent=2)

    logger.info("\n" + "=" * 85)
    logger.info(f"SUCCESS: Parameter experiments completed cleanly.")
    logger.info(f"  - Log output saved to:  {output_log}")
    logger.info(f"  - JSON dataset saved to: {output_json}")
    logger.info("=" * 85)

if __name__ == "__main__":
    run_parameter_experiments()
