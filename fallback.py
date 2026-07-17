"""
fallback.py
Fallback and timeout handling.

Wraps risky/slow operations (mainly the LLM call) so that if something
takes too long or fails outright, the assistant still responds gracefully
instead of hanging or crashing. Also handles exit-phrase detection, since
that's part of the same "keep the conversation flow safe and predictable"
responsibility.
"""

import concurrent.futures

import config
from utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Exit phrase detection
# ---------------------------------------------------------------------------

# Fuzzy-matched (substring check) rather than exact-equality, so variations
# like "Bye-bye!", "okay bye", "goodbye then" are all caught — not just an
# exact "bye".
EXIT_PHRASES = {
    "bye",
    "bye-bye",
    "goodbye",
    "good bye",
    "exit",
    "quit",
    "stop",
    "see you",
    "that's all",
    "that will be all",
}


def is_exit_phrase(text: str) -> bool:
    """
    Checks whether the user's transcribed text signals they want to end
    the conversation.

    Args:
        text: the transcribed user speech.

    Returns:
        True if an exit phrase was detected, False otherwise.
    """
    if not text:
        return False

    normalized = text.lower().strip().strip(".!?")

    for phrase in EXIT_PHRASES:
        if phrase in normalized:
            logger.info(f"Exit phrase matched: '{phrase}' in '{text}'")
            return True

    return False


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------

def run_with_timeout(func, args=(), timeout=None):
    """
    Runs a function in a background thread and enforces a timeout.

    If the function doesn't finish in time, returns None instead of
    blocking forever (the thread is abandoned, not killed — Python
    can't force-kill threads, but the caller moves on regardless).

    Also catches any other exception the function might raise (e.g.
    Ollama isn't running, connection refused) so a dead LLM service
    can't crash the whole assistant.

    Args:
        func: the function to call.
        args: tuple of positional arguments to pass to func.
        timeout: seconds to wait before giving up. Defaults to
                 config.LLM_TIMEOUT_SECONDS.

    Returns:
        The function's return value, or None if it timed out or failed.
    """
    timeout = timeout or config.LLM_TIMEOUT_SECONDS
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"{func.__name__} timed out after {timeout}s.")
            return None
        except Exception as e:
            logger.error(f"{func.__name__} raised an exception: {e}")
            return None


# ---------------------------------------------------------------------------
# LLM fallback wrapper
# ---------------------------------------------------------------------------

def get_llm_response_with_fallback(get_response_func, user_text: str) -> str:
    """
    Calls the LLM with a timeout and fallback handling.

    Behavior:
        - If user_text is empty -> return the "didn't catch that" fallback.
        - If the LLM times out or raises an exception -> return the
          "slow filler" fallback.
        - If the LLM call returns empty -> return the "LLM failed" fallback.
        - Otherwise -> return the real LLM response.

    Args:
        get_response_func: the llm.get_response function (passed in to
                            avoid a circular import between llm.py and
                            fallback.py).
        user_text: the transcribed user speech.

    Returns:
        A text response — always something speakable, never empty.
    """
    if not user_text:
        logger.info("No speech detected, using fallback phrase.")
        return config.FALLBACK_NO_SPEECH_DETECTED

    result = run_with_timeout(get_response_func, args=(user_text,))

    if result is None:
        logger.warning("LLM timed out or failed, using slow-filler fallback.")
        return config.FALLBACK_LLM_SLOW_FILLER

    if result == "":
        logger.error("LLM returned empty response, using failure fallback.")
        return config.FALLBACK_LLM_FAILED

    return result