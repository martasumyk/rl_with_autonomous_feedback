import os
from typing import Any, Dict

from PIL import Image

from .client import get_eval_model_id, init_eval_client
from .parser import normalize_verdict, safe_json_extract
from .prompts import EVAL_PROMPT


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return "data:image/png;base64," + b64


def judge_from_screenshot_path(task: str, screenshot_path: str) -> Dict[str, Any]:
    """
    Returns:
      { "completed": 0/1, "justification": str, "raw": str }
    """
    client = init_eval_client()
    model_id = get_eval_model_id()

    img = Image.open(screenshot_path).convert("RGB")
    img_b64 = encode_png_b64(img)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": f"{EVAL_PROMPT}\n\nTASK:\n{task}\n"},
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=200,
        stream=False,
    )

    raw = completion.choices[0].message.content or ""
    parsed = safe_json_extract(raw)
    return normalize_verdict(parsed, raw)


def judge_last_session_screenshot(task: str, session_dir: str) -> Dict[str, Any]:
    """
    Picks the latest step_*.png from session_dir and evaluates it.
    """
    pngs = sorted(
        p for p in os.listdir(session_dir)
        if p.startswith("step_") and p.endswith(".png")
    )
    if not pngs:
        raise FileNotFoundError(f"No step_*.png found in {session_dir}")

    last_path = os.path.join(session_dir, pngs[-1])
    return judge_from_screenshot_path(task, last_path)
