import os
from openai import OpenAI
from config import HF_BASE_URL

def init_eval_client() -> OpenAI:
    return OpenAI(
        base_url=HF_BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def get_eval_model_id() -> str:
    # .env: EVAL_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
    return os.getenv("EVAL_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
