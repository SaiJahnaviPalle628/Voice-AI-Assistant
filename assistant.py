"""
assistant.py

The Assistant class — the "conductor" of the whole pipeline.
Ties together Speech-to-Text, LLM (with fallback/timeout handling),
and Text-to-Speech into a single continuous conversation loop.
"""

import string

import speech
import tts
import llm
from fallback import get_llm_response_with_fallback
from utils import setup_logger

logger = setup_logger(__name__)

EXIT_PHRASES = {"exit", "quit", "stop", "goodbye", "bye"}


class Assistant:
    """
    Orchestrates one full conversation turn:
        listen -> think (LLM, with fallback) -> speak
    and runs this in a loop until the user says an exit phrase
    or interrupts with Ctrl+C.
    """

    def __init__(self):
        logger.info("Assistant initialized.")

    def run_turn(self) -> str:
        """
        Executes a single conversation turn:
        1. Record + transcribe user speech
        2. If it's an exit phrase, respond with a goodbye and skip the LLM
        3. Otherwise, get an LLM response (with timeout/fallback protection)
        4. Speak the response out loud

        Returns:
            The transcribed user text (so the caller can check for exit phrases).
        """
        # 1. Listen
        user_text = speech.listen()
        print(f"You: {user_text}")

        # 2. If the user is ending the conversation, skip the LLM entirely
        # and give a clean, natural goodbye instead of a random follow-up question.
        if self._is_exit_phrase(user_text):
            reply = "Goodbye! Have a great day."
            print(f"Assistant: {reply}")
            tts.speak(reply)
            return user_text

        # 3. Think (protected by fallback/timeout logic)
        reply = get_llm_response_with_fallback(llm.get_response, user_text)
        print(f"Assistant: {reply}")

        # 4. Speak
        tts.speak(reply)

        return user_text

    def _is_exit_phrase(self, text: str) -> bool:
        """
        Checks if the user's speech contains an exit command.
        Handles punctuation (e.g. "Goodbye.") and short phrases
        (e.g. "Okay, bye") rather than requiring an exact match.
        """
        cleaned = text.strip().lower().translate(str.maketrans("", "", string.punctuation))
        words = set(cleaned.split())
        return bool(words & EXIT_PHRASES)

    def run_loop(self) -> None:
        """
        Runs continuous conversation turns until the user says
        an exit phrase, or presses Ctrl+C.
        """
        print("Assistant is running. Say 'exit' or press Ctrl+C to stop.\n")

        try:
            while True:
                user_text = self.run_turn()

                if self._is_exit_phrase(user_text):
                    print("Exit phrase detected. Ending conversation.")
                    logger.info("Assistant stopped via exit phrase.")
                    break

                print()  # blank line between turns for readability

        except KeyboardInterrupt:
            print("\nAssistant stopped by user (Ctrl+C).")
            logger.info("Assistant stopped via KeyboardInterrupt.")