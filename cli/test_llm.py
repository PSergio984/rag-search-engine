# Load environment variables from .env file
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

# Create an OpenAI-compatible client pointed at OpenRouter's API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Build the message payload with a single user prompt
messages = [
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]

# Send the request to OpenRouter using the free tier model
response = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
)

# Print the model's reply and token usage statistics
print(response.choices[0].message.content)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Response tokens: {response.usage.completion_tokens}")