"""
Structured Output & Resilient JSON Parsing Evaluation Script
Demonstrates:
1. Task 1: Prompting for defined JSON structure with response_format={"type": "json_object"}
2. Task 2: Parsing valid JSON into a usable Python dictionary object
3. Task 3: Gracefully handling malformed JSON (Markdown-wrapped, trailing commas, broken syntax)
4. Task 4: Schema field validation (required 'answer' and 'sources' fields, auto-recovery, and rejection)
5. Task 5: Saving sample evaluation outputs to log and JSON artifacts.
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
from src.structured_parser import StructuredResponseParser

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("LexTrace.StructuredOutputDemo")
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

def run_structured_output_demo():
    output_log = Config.OUTPUTS_DIR / "structured_output_results.log"
    output_json = Config.OUTPUTS_DIR / "structured_output_results.json"
    logger = setup_logger(output_log)

    logger.info("=" * 85)
    logger.info("       LexTrace RAG Assistant - Structured Output & Resilient Parsing Suite       ")
    logger.info("=" * 85)

    system_prompt_json_path = Config.PROMPTS_DIR / "system_prompt_json.txt"
    system_prompt_json = system_prompt_json_path.read_text(encoding="utf-8").strip() if system_prompt_json_path.exists() else "Respond only in valid JSON format."

    user_query = "What is LexTrace's policy on remote work equipment reimbursement?"

    client = ChatCompletionClient()
    parser = StructuredResponseParser()
    evaluation_records: List[Dict[str, Any]] = []

    # =========================================================================
    # SCENARIO 1 (Task 1 & 2): Valid JSON Mode & Object Parsing
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 1 & 2] SCENARIO 1: PROMPTING FOR STRUCTURED JSON & PARSING INTO DICT")
    logger.info("Goal: Request JSON mode response_format and parse into a clean usable Python dictionary.")
    logger.info("=" * 85)

    res_1 = client.create_chat_completion(
        user_message=user_query,
        system_message=system_prompt_json,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    raw_content_1 = res_1.get("content", "")
    logger.info(f"Raw Model Completion Output:\n{raw_content_1}")

    parsed_1 = parser.parse_json_response(raw_content_1)
    validated_1 = parser.validate_and_normalize_schema(parsed_1, required_fields=["answer", "sources"])

    logger.info(f"  - Parsing Tier Used: {parsed_1.get('tier_used')}")
    logger.info(f"  - Schema Validation Status: {validated_1.get('status')}")
    logger.info(f"  - Usable Python Dictionary Object: {validated_1.get('data')}")

    evaluation_records.append({
        "scenario_id": "SCENARIO_01",
        "title": "Valid Structured JSON Output & Parsing",
        "raw_response": raw_content_1,
        "parsed_result": parsed_1,
        "validation_result": validated_1
    })

    # =========================================================================
    # SCENARIO 2 (Task 3): Malformed JSON - Markdown Fence Recovery
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 3] SCENARIO 2: MALFORMED JSON RECOVERY (Markdown Code Fence Wrapping)")
    logger.info("Goal: Demonstrate resilient extraction when model wraps JSON inside ```json ... ``` codeblocks.")
    logger.info("=" * 85)

    malformed_markdown_raw = (
        "Here is the requested policy information in JSON format:\n\n"
        "```json\n"
        "{\n"
        '  "answer": "LexTrace covers pre-approved dual monitors, ergonomic peripherals, and $50/month internet stipend for remote employees.",\n'
        '  "sources": ["LexTrace HR Portal", "Finance Equipment Policy 2026"],\n'
        '  "confidence": 0.96\n'
        "}\n"
        "```\n\n"
        "Please let me know if you need any additional citations."
    )
    logger.info(f"Raw Model Completion Output (Markdown Wrapped):\n{malformed_markdown_raw}")

    parsed_2 = parser.parse_json_response(malformed_markdown_raw)
    validated_2 = parser.validate_and_normalize_schema(parsed_2)

    logger.info(f"  - Parsing Tier Used: {parsed_2.get('tier_used')}")
    logger.info(f"  - Recovered Successfully: {parsed_2.get('parsed_successfully')}")
    logger.info(f"  - Recovered Python Object: {validated_2.get('data')}")

    evaluation_records.append({
        "scenario_id": "SCENARIO_02",
        "title": "Malformed JSON - Markdown Codeblock Extraction & Recovery",
        "raw_response": malformed_markdown_raw,
        "parsed_result": parsed_2,
        "validation_result": validated_2
    })

    # =========================================================================
    # SCENARIO 3 (Task 3): Malformed JSON - Unparseable Syntax Graceful Catch
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 3] SCENARIO 3: MALFORMED JSON GRACEFUL FAILURE HANDLING (Unparseable Syntax)")
    logger.info("Goal: Detect broken JSON syntax without crashing or raising unhandled exceptions.")
    logger.info("=" * 85)

    unparseable_syntax_raw = (
        '{\n'
        '  "answer": "LexTrace Remote Reimbursement Policy is $50 per month...\n'
        '  "sources": ["HR Manual Section 4",\n'
        '  "confidence": UNQUOTED_IDENTIFIER_BROKEN\n'
    )
    logger.info(f"Raw Model Completion Output (Broken Syntax):\n{unparseable_syntax_raw}")

    parsed_3 = parser.parse_json_response(unparseable_syntax_raw)
    validated_3 = parser.validate_and_normalize_schema(parsed_3)

    logger.info(f"  - Parsed Successfully: {parsed_3.get('parsed_successfully')}")
    logger.info(f"  - Error Type: {parsed_3.get('error_type')}")
    logger.info(f"  - Error Message: {parsed_3.get('error_message')}")
    logger.info(f"  - Application Status: Cleanly caught failure without application crash.")

    evaluation_records.append({
        "scenario_id": "SCENARIO_03",
        "title": "Malformed JSON - Unparseable Syntax Graceful Handling",
        "raw_response": unparseable_syntax_raw,
        "parsed_result": parsed_3,
        "validation_result": validated_3
    })

    # =========================================================================
    # SCENARIO 4 (Task 4): Required Field Validation & Auto-Remediation
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 4] SCENARIO 4: REQUIRED FIELD VALIDATION & AUTO-REMEDIATION")
    logger.info("Goal: Detect missing required fields, map singular 'source' -> 'sources', or apply fallbacks.")
    logger.info("=" * 85)

    singular_source_raw = json.dumps({
        "answer": "Employees may claim up to $50/month for remote broadband expenses with itemized invoices.",
        "source": "LexTrace HR Broadband Stipend Guide v3",
        "confidence": 0.92
    }, indent=2)
    logger.info(f"Raw JSON Output (Using Singular 'source' Field Alias):\n{singular_source_raw}")

    parsed_4 = parser.parse_json_response(singular_source_raw)
    validated_4 = parser.validate_and_normalize_schema(parsed_4, required_fields=["answer", "sources"])

    logger.info(f"  - Validation Status: {validated_4.get('status')}")
    logger.info(f"  - Warnings / Auto-Remediations: {validated_4.get('warnings')}")
    logger.info(f"  - Normalized Python Object: {validated_4.get('data')}")

    evaluation_records.append({
        "scenario_id": "SCENARIO_04",
        "title": "Required Field Validation - Alias Normalization & Auto-Remediation",
        "raw_response": singular_source_raw,
        "parsed_result": parsed_4,
        "validation_result": validated_4
    })

    # =========================================================================
    # SCENARIO 5 (Task 4): Critical Missing Required Field Rejection
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info("[TASK 4] SCENARIO 5: CRITICAL MISSING FIELD REJECTION")
    logger.info("Goal: Reject JSON responses missing required 'answer' payload cleanly.")
    logger.info("=" * 85)

    missing_answer_raw = json.dumps({
        "sources": ["LexTrace Security Policy"],
        "confidence": 0.85
    }, indent=2)
    logger.info(f"Raw JSON Output (Missing Required 'answer' Field):\n{missing_answer_raw}")

    parsed_5 = parser.parse_json_response(missing_answer_raw)
    validated_5 = parser.validate_and_normalize_schema(parsed_5, required_fields=["answer", "sources"])

    logger.info(f"  - Validation Status: {validated_5.get('status')}")
    logger.info(f"  - Error Message: {validated_5.get('error')}")
    logger.info(f"  - Missing Fields Detected: {validated_5.get('missing_fields')}")

    evaluation_records.append({
        "scenario_id": "SCENARIO_05",
        "title": "Required Field Validation - Critical Missing Field Rejection",
        "raw_response": missing_answer_raw,
        "parsed_result": parsed_5,
        "validation_result": validated_5
    })

    # Save JSON comparison output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_records, f, indent=2)

    logger.info("\n" + "=" * 85)
    logger.info(f"SUCCESS: Structured output & resilient parsing suite executed cleanly.")
    logger.info(f"  - Log output saved to:  {output_log}")
    logger.info(f"  - JSON dataset saved to: {output_json}")
    logger.info("=" * 85)

if __name__ == "__main__":
    run_structured_output_demo()
