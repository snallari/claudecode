import os

import anthropic
from dotenv import load_dotenv

from src.demo import build_demo_answer, stream_text
from src.client import chat, print_api_error, read_stream_text, resolve_model
from src.conversation import add_assistant_message, build_conversation

DEFAULT_MODEL = "claude-3-5-sonnet-latest"


def main() -> None:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Anthropic package is installed, but ANTHROPIC_API_KEY is not set.")
        print("Set it in `.env` to enable API calls.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    fallback_model = os.getenv("ANTHROPIC_FALLBACK_MODEL")
    stream = True

    resolved_model, resolution_note = resolve_model(client, model, fallback_model)
    if resolution_note:
        print(resolution_note)

    messages = build_conversation("Define quantum computing in one sentence")

    try:
        response = chat(client=client, messages=messages, model=resolved_model, stream=stream)
    except anthropic.NotFoundError as error:
        if fallback_model and fallback_model != model:
            print(f"Model '{model}' not found. Retrying with fallback model '{fallback_model}'.")
            response = chat(client=client, messages=messages, model=fallback_model, stream=stream)
        else:
            print(f"Model '{resolved_model}' was not found for this API key.")
            print("Set `ANTHROPIC_MODEL` to an available model, or set `ANTHROPIC_FALLBACK_MODEL`.")
            print(f"Original error: {error}")
            return
    except anthropic.APIError as error:
        print_api_error(error)
        return

    try:
        answer = read_stream_text(response) if stream else response
    except anthropic.AuthenticationError as error:
        print(f"API key rejected: {error}")
        print("Showing a local streaming demo instead.")
        answer = stream_text(build_demo_answer("Define quantum computing in one sentence"))
    except anthropic.APIError as error:
        print_api_error(error)
        return

    add_assistant_message(messages, answer)

    print("Claude response:")
    print(answer)


if __name__ == "__main__":
    main()