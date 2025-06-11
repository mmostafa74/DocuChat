from openai import OpenAI
from config import config


def get_openrouter_client():
    """Initialize OpenRouter client"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.app.get("openrouter_api_key"),
    )


def chat_with_openrouter(client, model, messages, temperature=0.7, max_tokens=1024):
    """Send chat request to OpenRouter"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
