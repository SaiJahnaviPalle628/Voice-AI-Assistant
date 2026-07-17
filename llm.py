"""
llm.py
LLM module.
Sends transcribed text to a locally running Ollama model
and returns the generated response.
"""
import ollama
import config
from utils import setup_logger, timeit

logger = setup_logger(__name__)


@timeit
def get_response(user_text: str) -> str:
    """
    Sends user text to the local Ollama model and returns its reply.

    Args:
        user_text: the transcribed speech from the user.

    Returns:
        The model's text response. Returns an empty string on failure
        (caller/fallback logic decides what to do in that case).
    """
    if not user_text:
        logger.warning("get_response called with empty input, skipping LLM call.")
        return ""

    try:
        logger.info(f"Sending to Ollama ({config.OLLAMA_MODEL}): '{user_text}'")
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            options={
                "num_predict": config.LLM_MAX_TOKENS,  # hard cap -> faster, shorter replies
                "temperature": 0.7,
            }
        )
        reply = response["message"]["content"].strip()
        logger.info(f"Ollama response: '{reply}'")
        return reply
    except Exception as e:
        # Covers: Ollama not running, model not pulled, connection refused, etc.
        logger.error(f"LLM call failed: {e}")
        return ""