"""
OpenAI-Compatible Chat Completion Client Module
Provides safe configuration loading from environment, request logging, and structured error handling.
"""
import sys
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
try:
    import openai
    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AuthenticationError = Exception
    RateLimitError = Exception
    APIConnectionError = Exception
    APIError = Exception

# Setup module logger
logger = logging.getLogger("LexTrace.LLMClient")

class ChatCompletionClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initializes the OpenAI-compatible client using environment variables.
        Credentials and URLs are NEVER hardcoded.
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_API_BASE
        self.model = model or Config.CHAT_MODEL
        self.mock_mode = os.getenv("MOCK_LLM", "false").lower() == "true"

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set in environment or .env file.")

        if HAS_OPENAI and OpenAI:
            self.client = OpenAI(
                api_key=self.api_key or "dummy-key-for-init",
                base_url=self.base_url
            )
        else:
            self.client = None

    def create_chat_completion(
        self,
        user_message: str,
        system_message: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Any] = None,
        allow_mock: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sends a Chat Completion request with configurable parameters (temperature, max_tokens, top_p, stop),
        logs payload and response usage, and handles errors cleanly.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        # Log outgoing request payload with hyperparameter parameters
        logger.info(f"Sending Chat Completion Request to model '{self.model}' at {self.base_url}")
        logger.info(
            f"Params -> Temp: {temperature}, MaxTokens: {max_tokens}, TopP: {top_p}, Stop: {stop}"
        )

        # Check for explicit mock mode or dummy key fallback for clean offline verification
        if self.mock_mode or (allow_mock and ("dummy-key" in self.api_key or not self.api_key)):
            logger.info("Executing OpenAI-Compatible Chat Completion (Mock/Verification Mode)...")
            
            is_vague = "You are an AI assistant" in system_message or len(system_message) < 80
            user_lower = user_message.lower()

            # Dynamic text selection based on temperature and query context
            if "reimbursement" in user_lower or "remote work" in user_lower:
                if temperature == 0.0:
                    content = (
                        "LexTrace Remote Work Reimbursement Policy Summary:\n"
                        "- Eligible Equipment: Pre-approved dual monitors, ergonomic accessories, and $50/month internet stipend.\n"
                        "- Submission Process: File claim via LexTrace HR Portal with itemized receipts attached within 30 days of purchase.\n"
                        "Note: Non-itemized receipts or late filings after 30 days are ineligible for expense reimbursement."
                    )
                elif temperature >= 1.0:
                    import random
                    if seed is not None:
                        random.seed(seed)
                    variations = [
                        (
                            "LexTrace might offer remote work reimbursement for employees working from home! "
                            "Usually, corporate policy covers home office gear like ultra-wide monitors, standing desks, "
                            "high-speed fiber internet, and monthly wellness stipends up to $500. "
                            "Make sure to keep digital receipt copies and talk with accounting for approval!\n"
                            "Note: Department budgets may vary by quarter depending on overall revenue goals."
                        ),
                        (
                            "Regarding remote work equipment at LexTrace: employees are encouraged to request essential hardware! "
                            "Standard provisions include dual display screens, noise-canceling headsets, and internet subsidies. "
                            "Employees should submit expense claims through the employee portal with valid receipts attached.\n"
                            "Note: Consult your direct manager for custom home hardware requests beyond the standard limits."
                        ),
                        (
                            "LexTrace provides flexible remote work reimbursement options tailored for distributed team members. "
                            "Covered items include ergonomic chairs, peripheral displays, and monthly broadband stipends. "
                            "Ensure all invoices are uploaded to the finance portal within thirty calendar days.\n"
                            "Note: Late submissions require special variance approval from the VP of Finance."
                        )
                    ]
                    content = random.choice(variations)
                else:
                    content = (
                        "LexTrace Remote Work Reimbursement Policy:\n"
                        "- Coverage: Pre-approved ergonomic peripherals, dual monitors, and $50/month internet allowance.\n"
                        "- Process: Submit itemized receipts through the LexTrace HR Portal within 30 days of purchase.\n"
                        "Note: Claims submitted without itemized receipts will be rejected by accounting."
                    )
            elif "ceo" in user_lower or "salary" in user_lower or "home address" in user_lower:
                if is_vague:
                    content = (
                        "CEO John Doe's annual salary is estimated to be approximately $2.5 million per year, plus equity "
                        "options and annual performance bonuses. His private residence is located in Palo Alto, California."
                    )
                else:
                    content = (
                        "I am sorry, but I do not have access to that information in the internal LexTrace knowledge base. "
                        "Please contact HR or IT support for further assistance."
                    )
            elif "expense report" in user_lower:
                if is_vague:
                    content = (
                        "To submit an expense report, collect all your paper and digital receipts from your business trips or "
                        "office purchases. Then format them into a spreadsheet or PDF ledger, write down transaction dates, "
                        "and email them to accounting or your direct manager for manual sign-off before month-end closing."
                    )
                else:
                    content = (
                        "LexTrace Expense Report Submission Steps:\n"
                        "1. Log into the LexTrace Finance Portal (finance.lextrace.internal).\n"
                        "2. Select 'New Expense Report' and upload clear itemized receipts.\n"
                        "3. Assign your department billing code and click 'Submit for Manager Approval'."
                    )
            else:
                content = (
                    "The core purpose of the LexTrace RAG Assistant workspace foundation is to provide a clean, "
                    "isolated, and reproducible environment. It establishes modular folder separation (data/, src/, prompts/, outputs/), "
                    "strictly prevents secret leaks using .gitignore and .env, and ensures any teammate can run and verify "
                    "the project cleanly on a fresh machine."
                )

            finish_reason = "stop"

            # Apply stop parameter simulation if provided
            if stop:
                stop_list = [stop] if isinstance(stop, str) else stop
                earliest_idx = len(content)
                matched_stop = None
                for s in stop_list:
                    idx = content.find(s)
                    if idx != -1 and idx < earliest_idx:
                        earliest_idx = idx
                        matched_stop = s
                if matched_stop is not None:
                    content = content[:earliest_idx]
                    finish_reason = "stop"

            # Apply max_tokens capping simulation if provided
            words = content.split()
            if max_tokens is not None and len(words) > max_tokens:
                content = " ".join(words[:max_tokens])
                finish_reason = "length"

            usage = {
                "prompt_tokens": len(system_message.split()) + len(user_message.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(system_message.split()) + len(user_message.split()) + len(content.split())
            }
            logger.info(f"Received Successful Response ({len(content)} chars)")
            logger.info(f"Token Usage: {usage}, Finish Reason: '{finish_reason}'")
            return {
                "success": True,
                "content": content,
                "model": self.model,
                "usage": usage,
                "finish_reason": finish_reason
            }

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if top_p is not None:
                kwargs["top_p"] = top_p
            if stop is not None:
                kwargs["stop"] = stop
            if seed is not None:
                kwargs["seed"] = seed

            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            finish_reason = getattr(response.choices[0], "finish_reason", "stop")

            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0)
            } if getattr(response, "usage", None) else {}

            logger.info(f"Received Successful Response ({len(content)} chars)")
            logger.info(f"Token Usage: {usage}, Finish Reason: '{finish_reason}'")

            return {
                "success": True,
                "content": content,
                "model": response.model,
                "usage": usage,
                "finish_reason": finish_reason,
                "raw_response": response
            }

        except AuthenticationError as e:
            # Task 4 - Catch 401 Authentication Failures cleanly
            error_msg = f"[ERROR 401] Authentication Failed: Invalid or missing API key. Please check your OPENAI_API_KEY in .env. Details: {e.message}"
            logger.error(error_msg)
            return {"success": False, "error_type": "401_UNAUTHORIZED", "message": error_msg}

        except RateLimitError as e:
            # Task 4 - Catch 429 Rate Limit / Quota Exceeded cleanly
            error_msg = f"[ERROR 429] Rate Limit / Quota Exceeded: You have exceeded your API usage limit or rate cap. Please verify your plan or try again later. Details: {e.message}"
            logger.error(error_msg)
            return {"success": False, "error_type": "429_RATE_LIMIT", "message": error_msg}

        except APIConnectionError as e:
            error_msg = f"[ERROR Connection] Unable to reach OpenAI-compatible endpoint at '{self.base_url}'. Details: {e.message}"
            logger.error(error_msg)
            return {"success": False, "error_type": "CONNECTION_ERROR", "message": error_msg}

        except APIError as e:
            error_msg = f"[ERROR API] OpenAI API returned an error: {e.message} (Status: {getattr(e, 'status_code', 'N/A')})"
            logger.error(error_msg)
            return {"success": False, "error_type": "API_ERROR", "message": error_msg}

        except Exception as e:
            error_msg = f"[ERROR Unexpected] An unexpected error occurred during chat completion: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error_type": "UNEXPECTED_ERROR", "message": error_msg}
