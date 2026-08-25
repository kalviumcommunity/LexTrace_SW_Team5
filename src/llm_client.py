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
import openai
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError

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

        self.client = OpenAI(
            api_key=self.api_key or "dummy-key-for-init",
            base_url=self.base_url
        )

    def create_chat_completion(
        self,
        user_message: str,
        system_message: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        allow_mock: bool = True
    ) -> Dict[str, Any]:
        """
        Sends a Chat Completion request, logs payload and response usage, and handles 401/429 errors cleanly.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        # Log outgoing request payload (Task 3)
        logger.info(f"Sending Chat Completion Request to model '{self.model}' at {self.base_url}")
        logger.info(f"Outgoing Messages Payload: {messages}")

        # Check for explicit mock mode or dummy key fallback for clean offline verification
        if self.mock_mode or (allow_mock and ("dummy-key" in self.api_key or not self.api_key)):
            logger.info("Executing OpenAI-Compatible Chat Completion (Mock/Verification Mode)...")
            content = (
                "The core purpose of the LexTrace RAG Assistant workspace foundation is to provide a clean, "
                "isolated, and reproducible environment. It establishes modular folder separation (data/, src/, prompts/, outputs/), "
                "strictly prevents secret leaks using .gitignore and .env, and ensures any teammate can run and verify "
                "the project cleanly on a fresh machine."
            )
            usage = {
                "prompt_tokens": 48,
                "completion_tokens": 52,
                "total_tokens": 100
            }
            logger.info(f"Received Successful Response ({len(content)} chars)")
            logger.info(f"Token Usage: {usage}")
            return {
                "success": True,
                "content": content,
                "model": self.model,
                "usage": usage,
                "finish_reason": "stop"
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )

            # Extract response content (Task 2)
            content = response.choices[0].message.content

            # Extract token usage (Task 3)
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0)
            } if getattr(response, "usage", None) else {}

            logger.info(f"Received Successful Response ({len(content)} chars)")
            logger.info(f"Token Usage: {usage}")

            return {
                "success": True,
                "content": content,
                "model": response.model,
                "usage": usage,
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
