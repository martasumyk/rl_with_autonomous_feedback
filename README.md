# Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation

This repository contains the code for the paper **“Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation.”**

The project explores how to fine-tune Computer-Use Agents (CUAs) with reinforcement learning when no explicit environment reward is available. Instead of relying on handcrafted success checks or manual labels during training, the framework uses an autonomous vision-language evaluator to judge whether a GUI task has been completed from the final screenshot and instruction.

Because autonomous evaluators are imperfect, the repository also includes a reward-correction procedure that accounts for evaluator false positives and false negatives before using the signal for PPO fine-tuning.

## Overview

Computer-Use Agents operate graphical user interfaces from natural-language instructions by observing screenshots and executing actions such as clicking, typing, scrolling, and dragging. However, reinforcement learning in desktop GUI environments is difficult because task success is often visually grounded and not exposed as a clean machine-readable reward.

This repository implements a pipeline for:

1. running a Computer-Use Agent on GUI tasks;
2. evaluating task completion with a vision-language model;
3. estimating evaluator noise from calibration data;
4. correcting noisy binary evaluator rewards;
5. fine-tuning the agent with PPO using the corrected reward signal.

## Repository Structure

```text
.
├── agents/
│   └── UI_TARS/          # UI-TARS agent implementation and execution loop
├── evaluator/            # Vision-based task-completion evaluator
├── rl_finetuning/        # PPO fine-tuning with noise-corrected rewards
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
├── Makefile              # Utility commands
└── README.md
```

## Method

The framework treats the autonomous evaluator as a noisy binary reward channel. For a given task, the evaluator observes the final screenshot and the original instruction, then predicts whether the task was completed:

```math
\tilde r \in \{0, 1\}
```

where `1` means that the evaluator judged the task as successful.

Since this signal may contain false positives and false negatives, the method estimates evaluator error rates on a held-out calibration split:

```math
e_+ = P(\tilde r = 0 \mid r^* = 1)
```

```math
e_- = P(\tilde r = 1 \mid r^* = 0)
```

The noisy evaluator reward is then corrected using:

```math
\hat r = \frac{\tilde r - e_-}{1 - e_+ - e_-}
```

This corrected reward is used as a terminal reward for PPO fine-tuning.

## Main Components

### `agents/`

Contains the Computer-Use Agent code used to execute GUI tasks. The current implementation is based on UI-TARS-style desktop interaction.

### `evaluator/`

Contains the autonomous task-completion evaluator. The evaluator takes the final GUI screenshot and the original task instruction as input and predicts whether the task was successfully completed.

### `rl_finetuning/`

Contains the reinforcement-learning fine-tuning code. This module uses evaluator-based rewards and supports reward correction before PPO updates.

## Installation

Clone the repository:

```bash
git clone https://github.com/martasumyk/rl_with_autonomous_feedback.git
cd rl_with_autonomous_feedback
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, if you use `pyproject.toml`:

```bash
pip install -e .
```

## Usage

The repository is organized around three main stages:

### 1. Run the agent

Use the code in `agents/` to execute a Computer-Use Agent on GUI tasks and collect trajectories.

### 2. Evaluate task completion

Use the evaluator to judge whether the final GUI state satisfies the original instruction.

### 3. Fine-tune with PPO

Use the corrected evaluator reward as the terminal reward for reinforcement learning.

Example workflow:

```bash
# Run agent trajectories
python agents/UI_TARS/run_all.py

# Run autonomous evaluation
python evaluator/evaluate.py

# Run PPO fine-tuning
python rl_finetuning/train.py
```

> Note: Script names may need to be adjusted depending on your local configuration and experiment setup.

## Citation

If you use this repository, please cite the paper:

```
To be added.
```


## License

Please check the repository license before using or redistributing the code.
