"""
main.py

Entry point for the AI Assistant.
Run this file to start the conversational assistant.
"""

from assistant import Assistant


def main():
    assistant = Assistant()
    assistant.run_loop()


if __name__ == "__main__":
    main()