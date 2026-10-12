from typing import Any, List


def add_user_message(messages: List[dict[str, Any]], text: str) -> None:
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages: List[dict[str, Any]], text: str) -> None:
    messages.append({"role": "assistant", "content": text})


def build_conversation(prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    add_user_message(messages, prompt)
    return messages
