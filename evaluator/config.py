import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY") and os.getenv("HF_TOKEN"):
    os.environ["OPENAI_API_KEY"] = os.getenv("HF_TOKEN")

HF_TOKEN    = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")