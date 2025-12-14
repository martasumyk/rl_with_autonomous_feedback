# Reinforcement Learning fine-tuning with PPO based on reward estimator from autonomous evaluator feedback

This folder contains code to fine-tune a Computer-Use Agent (UI-TARS) using
Proximal Policy Optimization (PPO) with an autonomous
vision-based evaluator (in our case, Qwen2-VL-7B) as a source of reward.

The evaluator provides a noisy binary signal
\[
\tilde r \in \{0,1\},
\]
indicating whether the current GUI state appears to satisfy the user instruction.
Due to evaluator imperfections, this signal may contain false positives and false negatives.

Let the evaluator error rates be defined as
\[
e_+ := \Pr(\tilde r = 0 \mid r = 1) \quad \text{(false negative)},
\]
\[
e_- := \Pr(\tilde r = 1 \mid r = 0) \quad \text{(false positive)}.
\]

Assuming the separability condition
\[
1 - e_+ - e_- > 0,
\]
we construct a noise-corrected reward estimator
\[
\hat r
\;=\;
\frac{\tilde r - e_-}{1 - e_+ - e_-}.
\]

This estimator is an asymptotically unbiased estimator of the true task-completion
reward and is used directly as the scalar reward signal for PPO optimization.



## Setup

Set the following variables (e.g., in \texttt{.env}):

\begin{verbatim}
PPO_POLICY_MODEL=ByteDance-Seed/UI-TARS-1.5-7B
PPO_OUT=ppo_ui_tars_lora_mm

EVAL_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
E_PLUS=...
E_MINUS=...
\end{verbatim}


## Running Training

From the directory containing the UI-TARS codebase:

\begin{verbatim}
python rl_finetuning/ppo_train.py
\end{verbatim}

Training periodically saves LoRA weights to \texttt{PPO\_OUT/}.

