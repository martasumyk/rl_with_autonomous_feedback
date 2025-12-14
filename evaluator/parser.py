import json
import re
from typing import Any, Dict, Optional


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def normalize_verdict(parsed: Optional[Dict[str, Any]], raw: str) -> Dict[str, Any]:
    if not parsed:
        m = re.search(r"\bcompleted\b\D*(0|1)\b", raw or "")
        completed = int(m.group(1)) if m else 0
        justification = (raw or "").strip()[:300]
        return {"completed": completed, "justification": justification, "raw": raw}

    try:
        completed = int(parsed.get("completed", 0))
    except Exception:
        completed = 0

    completed = 1 if completed == 1 else 0
    justification = str(parsed.get("justification", "")).strip()
    if not justification:
        justification = (raw or "").strip()[:300]

    return {"completed": completed, "justification": justification, "raw": raw}
