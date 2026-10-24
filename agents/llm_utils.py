import os
import time
from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# Read SiliconFlow API key from environment
api_key = os.getenv("SILICONFLOW_API_KEY")
if not api_key:
    print("Warning: SILICONFLOW_API_KEY is not set in .env!")

# Initialize SiliconFlow OpenAI-compatible client
if OpenAI is None:
    print("Warning: openai package is not installed. LLM calls will be skipped.")
    client = None
else:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
    ) if api_key else None

# Use neutral entry name ask_llm across agents
def ask_llm(prompt: str) -> str:
    if not client:
        return "ERROR"

    model_name = os.getenv("SILICONFLOW_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")
    timeout_seconds = float(os.getenv("SILICONFLOW_TIMEOUT_SECONDS", "90"))
    max_retries = int(os.getenv("SILICONFLOW_MAX_RETRIES", "2"))

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a highly capable cryptocurrency trading AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                timeout=timeout_seconds,
            )
            content = completion.choices[0].message.content
            return content if isinstance(content, str) else "ERROR"
        except Exception as e:
            print(
                f"SiliconFlow API Error (attempt {attempt + 1}/{max_retries + 1}, "
                f"timeout={timeout_seconds}s): {e}"
            )
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))

    return "ERROR"


# Backward-compatible alias for legacy callers
def ask_groq(prompt: str) -> str:
    return ask_llm(prompt)