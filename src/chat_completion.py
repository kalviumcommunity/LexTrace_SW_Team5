"""
Executable Script for OpenAI-Compatible Chat Completion API Demonstration
Executes a request, logs payload and response, handles errors cleanly, and writes output to outputs/chat_completion_sample.log
"""
import sys
import os
import logging
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.llm_client import ChatCompletionClient
from src.prompt_manager import PromptManager

def setup_logging(log_file: Path) -> logging.Logger:
    """Configures dual logging to console and sample log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("LexTrace")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def run_chat_completion_demo():
    output_log_file = Config.OUTPUTS_DIR / "chat_completion_sample.log"
    logger = setup_logging(output_log_file)

    logger.info("=" * 70)
    logger.info("       LexTrace RAG Assistant - Interactive Chat Completion Feature      ")
    logger.info("=" * 70)

    # Task 1 - Read config from environment (.env)
    logger.info("[Task 1] Loaded Environment Configuration:")
    logger.info(f"  - Base URL: {Config.OPENAI_API_BASE}")
    logger.info(f"  - Chat Model: {Config.CHAT_MODEL}")
    logger.info(f"  - API Key Configured: {'Yes (Hidden)' if Config.OPENAI_API_KEY else 'No'}")

    # Feature 1 - Render System & RAG Query Prompts via PromptManager
    pm = PromptManager()
    system_prompt = pm.render_prompt(
        "system_prompt",
        role="LexTrace Internal RAG Assistant",
        domain="Legal & Corporate Workplace Operations",
        constraints="Do not hallucinate policy facts. Cite document titles."
    )

    sample_context = (
        "Document: LexTrace Remote Work Reimbursement Policy 2026\n"
        "Coverage: Pre-approved dual monitors, ergonomic accessories, and $50/month internet stipend.\n"
        "Filing Window: Receipts must be uploaded to the LexTrace HR Portal within 30 days."
    )

    user_query_rendered = pm.render_prompt(
        "rag_query",
        context=sample_context,
        question="What is the reimbursement limit and filing deadline for remote work gear?",
        format_instructions="Summarize in 2 bullet points with document citation."
    )

    logger.info("\n[Task 2 & 3] Rendered Prompts via PromptManager Engine:")
    logger.info(f"  - Rendered System Prompt:\n{system_prompt}")
    logger.info(f"  - Rendered RAG User Query:\n{user_query_rendered}")

    # Task 1 & 2 - Initialize client and send chat completion
    client = ChatCompletionClient()
    result = client.create_chat_completion(
        user_message=user_query_rendered,
        system_message=system_prompt,
        allow_mock=True
    )

    if result and result.get("success"):
        # Task 2 - Print Model's Text Reply (choices[0].message.content)
        logger.info("\nMODEL RESPONSE RECEIVED SUCCESSFULLY:")
        print("\n" + "-" * 50)
        print("MODEL REPLY:")
        print(result["content"])
        print("-" * 50 + "\n")

        # Task 3 - Log token usage metrics
        logger.info("[Task 3] Request & Response Payload Metrics:")
        logger.info(f"  - Model: {result.get('model', Config.CHAT_MODEL)}")
        logger.info(f"  - Prompt Tokens: {result.get('usage', {}).get('prompt_tokens', 0)}")
        logger.info(f"  - Completion Tokens: {result.get('usage', {}).get('completion_tokens', 0)}")
        logger.info(f"  - Total Tokens: {result.get('usage', {}).get('total_tokens', 0)}")
    else:
        logger.error(f"CHAT COMPLETION FAILED: {result.get('message')}")

    # Task 4 - Demonstrate Structured Error Handling (HTTP 401 & 429)
    logger.info("\n" + "=" * 70)
    logger.info("[Task 4] Demonstrating Structured Error Handling (Catching 401 / 429 Errors):")
    
    # Test 401 Authentication Failure Simulation
    logger.info("\n--- Testing 401 Unauthorized Handling (Invalid API Key) ---")
    invalid_client = ChatCompletionClient(api_key="sk-invalid-api-key-test-401")
    err_result = invalid_client.create_chat_completion(
        user_message="Test 401 error handling",
        allow_mock=False
    )
    logger.info(f"Error Handler Output: {err_result.get('message')}")

    logger.info("\n" + "=" * 70)
    logger.info(f"[Task 5] Log output successfully saved to: {output_log_file}")
    logger.info("=" * 70)

if __name__ == "__main__":
    run_chat_completion_demo()
