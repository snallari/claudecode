from __future__ import annotations

import time


def build_demo_answer(prompt: str) -> str:
    return (
        "Demo mode: quantum computing uses qubits that can represent multiple states at once, "
        f"which lets certain problems be explored more efficiently. Prompt: {prompt}"
    )


def stream_text(text: str, delay: float = 0.02) -> str:
    chunks: list[str] = []
    for word in text.split():
        chunk = f"{word} "
        print(chunk, end="", flush=True)
        chunks.append(chunk)
        time.sleep(delay)

    print()
    return "".join(chunks).strip()