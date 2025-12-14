# Evaluator

This folder contains a lightweight visual evaluator that judges, from the latest screenshot, whether the task is completed. It returns:

- `completed`: 0 (not done / unsure) or 1 (done)

- `justification`: short evidence-based explanation

## Setup

Add to `.env` file:

```
HF_TOKEN=hf_...

# Evaluator model (Qwen2-VL)
EVAL_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
```

## Usage

Use evaluator to judge a single trajectory:

```
from judge import judge_last_session_screenshot
from config import TASK

verdict = judge_last_session_screenshot(task=TASK, session_dir="sessions/2025-12-14_18-32-10")
print(verdict["completed"], verdict["justification"])
```
