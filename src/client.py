from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import anthropic


def chat(
    client: anthropic.Anthropic,
    messages: Iterable[dict[str, Any]],
    model: str,
    max_tokens: int = 1000,
    stream: bool = False,
):
    request_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": list(messages),
    }

    if stream:
        stream_method = getattr(client.messages, "stream", None)
        if callable(stream_method):
            return stream_method(**request_kwargs)
        request_kwargs["stream"] = True
        return client.messages.create(**request_kwargs)

    return client.messages.create(**request_kwargs)


def resolve_model(
    client: anthropic.Anthropic,
    preferred_model: str,
    fallback_model: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    candidates = [m for m in (preferred_model, fallback_model) if m]
    deduped_candidates = list(dict.fromkeys(candidates))

    try:
        model_page = client.models.list(limit=100)
        available_models = [model.id for model in model_page.data]
    except anthropic.APIError:
        return preferred_model, "Could not list models; using configured model."

    for model_name in deduped_candidates:
        if model_name in available_models:
            if model_name != preferred_model:
                return model_name, f"Configured model unavailable. Using '{model_name}'."
            return model_name, None

    sonnet_models = [m for m in available_models if "sonnet" in m]
    if sonnet_models:
        selected = sonnet_models[0]
        return selected, f"Configured models unavailable. Using discovered model '{selected}'."

    if available_models:
        selected = available_models[0]
        return selected, f"Configured models unavailable. Using discovered model '{selected}'."

    return preferred_model, "No models returned by API; using configured model."


def get_text_from_response(response) -> str:
    if hasattr(response, "text_stream"):
        chunks: list[str] = []
        stream_manager = response
        close_stream = False

        try:
            if hasattr(stream_manager, "__enter__") and hasattr(stream_manager, "__exit__"):
                stream_manager = stream_manager.__enter__()
                close_stream = True
            for text in getattr(stream_manager, "text_stream", ()):
                if text:
                    chunks.append(text)
        finally:
            if close_stream:
                response.__exit__(None, None, None)

        if chunks:
            return "".join(chunks)

    if hasattr(response, "content"):
        if not response.content:
            return ""
        first_block = response.content[0]
        return getattr(first_block, "text", "")

    chunks: list[str] = []
    for event in response:
        if getattr(event, "type", "") == "content_block_delta":
            delta = getattr(event, "delta", None)
            text = getattr(delta, "text", "") if delta else ""
            if text:
                chunks.append(text)
    return "".join(chunks)


def read_stream_text(stream_response) -> str:
    chunks: list[str] = []

    if hasattr(stream_response, "__enter__") and hasattr(stream_response, "__exit__"):
        with stream_response as stream:
            for event in stream:
                if getattr(event, "type", None) != "content_block_delta":
                    continue
                text = getattr(getattr(event, "delta", None), "text", None)
                if text:
                    print(text, end="", flush=True)
                    chunks.append(text)
    else:
        for event in stream_response:
            if getattr(event, "type", None) != "content_block_delta":
                continue
            text = getattr(getattr(event, "delta", None), "text", None)
            if text:
                print(text, end="", flush=True)
                chunks.append(text)

    print()
    return "".join(chunks).strip()


def print_api_error(error: anthropic.AnthropicError) -> None:
    if isinstance(error, anthropic.AuthenticationError):
        print("Authentication failed: invalid or expired `ANTHROPIC_API_KEY`.")
    elif isinstance(error, anthropic.PermissionDeniedError):
        print("Permission denied: this API key does not have access to the requested resource.")
    elif isinstance(error, anthropic.RateLimitError):
        print("Rate limit reached: reduce request frequency or retry shortly.")
    elif isinstance(error, (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.Timeout)):
        print("Network/timeout error while contacting Anthropic API.")
    elif isinstance(error, anthropic.NotFoundError):
        print("Requested model or resource was not found for this account.")
    elif isinstance(error, anthropic.BadRequestError):
        print("Bad request: verify request parameters and message payload.")
    elif isinstance(error, anthropic.OverloadedError):
        print("Anthropic service is overloaded right now; retry shortly.")
    else:
        print("Anthropic API request failed.")

    status_code = getattr(error, "status_code", None)
    if status_code:
        print(f"Status code: {status_code}")
    print(f"Error details: {error}")
