import os
from dotenv import load_dotenv
import anthropic
from src.multi import add_assistant_message, add_user_message, chat, get_text_from_response


def resolve_model(client, preferred_model, fallback_model=None):
    candidates = [preferred_model, fallback_model]
    requested_models = []
    for model_name in candidates:
        if model_name and model_name not in requested_models:
            requested_models.append(model_name)

    try:
        model_page = client.models.list(limit=100)
        available_models = [model.id for model in model_page.data]
    except anthropic.APIError:
        return preferred_model, "Could not list available models; using configured model."

    for model_name in requested_models:
        if model_name in available_models:
            if model_name != preferred_model:
                return model_name, f"Configured model unavailable. Using '{model_name}'."
            return model_name, None

    sonnet_models = [model_name for model_name in available_models if "sonnet" in model_name]
    if sonnet_models:
        return sonnet_models[0], f"Configured models unavailable. Using discovered model '{sonnet_models[0]}'."

    if available_models:
        return available_models[0], f"Configured models unavailable. Using discovered model '{available_models[0]}'."

    return preferred_model, "No models returned by API; using configured model."


def main() -> None:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        fallback_model = os.getenv("ANTHROPIC_FALLBACK_MODEL")

        resolved_model, resolution_note = resolve_model(client, model, fallback_model)
        if resolution_note:
            print(resolution_note)

        messages = []
        add_user_message(messages, "Define quantum computing in one sentence")

        try:
            response = chat(client=client, messages=messages, model=resolved_model)
        except anthropic.NotFoundError as error:
            if fallback_model and fallback_model != model:
                print(f"Model '{model}' not found. Retrying with fallback model '{fallback_model}'.")
                response = chat(client=client, messages=messages, model=fallback_model)
            else:
                print(f"Model '{resolved_model}' was not found for this API key.")
                print("Set `ANTHROPIC_MODEL` to an available model, or set `ANTHROPIC_FALLBACK_MODEL`.")
                print(f"Original error: {error}")
                return
        except anthropic.APIError as error:
            print("Anthropic API request failed.")
            print(f"Error details: {error}")
            return

        answer = get_text_from_response(response)
        add_assistant_message(messages, answer)

        print("Claude response:")
        print(answer)
    else:
        print("Anthropic package is installed, but ANTHROPIC_API_KEY is not set.")
        print("Set it in `.env` to enable API calls.")


if __name__ == "__main__":
    main()
