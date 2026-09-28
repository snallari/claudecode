def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(client, messages, model, max_tokens=1000):
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=0.1
    )


def get_text_from_response(response):
    if not response.content:
        return ""
    first_block = response.content[0]
    return getattr(first_block, "text", "")