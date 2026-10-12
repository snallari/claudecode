def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(client, messages, model, max_tokens=1000, stream=False):
    request_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if stream:
        stream_method = getattr(client.messages, "stream", None)
        if callable(stream_method):
            return stream_method(**request_kwargs)
        request_kwargs["stream"] = True
        return client.messages.create(**request_kwargs)
    return client.messages.create(**request_kwargs)


def get_text_from_response(response):
    if hasattr(response, "text_stream"):
        chunks = []
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

    chunks = []
    for event in response:
        if getattr(event, "type", "") == "content_block_delta":
            delta = getattr(event, "delta", None)
            text = getattr(delta, "text", "") if delta else ""
            if text:
                chunks.append(text)
    return "".join(chunks)