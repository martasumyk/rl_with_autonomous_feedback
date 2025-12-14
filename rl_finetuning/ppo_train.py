import os
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model

from agents.UI_TARS.config import TASK
from rollout_env import DesktopRolloutEnv
from reward import NoiseRates, corrected_reward


@dataclass
class TrainCfg:
    policy_model_name_or_path: str = os.getenv("PPO_POLICY_MODEL", "ByteDance-Seed/UI-TARS-1.5-7B")
    output_dir: str = os.getenv("PPO_OUT", "ppo_uinars_lora")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    episodes_per_update: int = 2
    max_steps_per_episode: int = 8

    # Evaluator FN and FP parameters
    e_plus: float = float(os.getenv("E_PLUS", "0.10"))   # FN rate
    e_minus: float = float(os.getenv("E_MINUS", "0.05")) # FP rate

    # PPO parameters
    lr: float = 1e-5
    batch_size: int = 2
    mini_batch_size: int = 1
    ppo_epochs: int = 2
    cliprange: float = 0.2
    vf_coef: float = 0.0
    gamma: float = 1.0


def build_text_from_messages(messages: List[dict]) -> str:
    user_parts = messages[0]["content"]
    instr = ""
    for p in user_parts:
        if p.get("type") == "text":
            instr = p["text"]
    return instr


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def policy_generate_fn_factory(model, tokenizer):
    def _fn(messages: List[dict]) -> str:
        prompt = build_text_from_messages(messages)
        return generate_response(model, tokenizer, prompt)
    return _fn

def main():
    cfg = TrainCfg()
    rates = NoiseRates(e_plus=cfg.e_plus, e_minus=cfg.e_minus)
    assert rates.D > 0.0, f"Separability violated: D={rates.D} must be >0"

    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_model_name_or_path,
        torch_dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
        device_map="auto" if cfg.device == "cuda" else None,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(base_model, lora)
    model.train()

    ppo_config = PPOConfig(
        learning_rate=cfg.lr,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        cliprange=cfg.cliprange,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    env = DesktopRolloutEnv(task=TASK, max_steps=cfg.max_steps_per_episode)

    policy_generate_fn = policy_generate_fn_factory(model, tokenizer)

    step = 0
    while True:
        episodes = []
        for _ in range(cfg.episodes_per_update):
            ep = env.run_episode(policy_generate_fn)
            episodes.append(ep)

        corrected_rewards = []
        for ep in episodes:
            tilde = float(ep.evaluator_completed)
            rhat = corrected_reward(tilde, rates)
            corrected_rewards.append(rhat)

        queries = []
        responses = []
        rewards = []

        for ep, rhat in zip(episodes, corrected_rewards):
            last = ep.transitions[-1]
            prompt = build_text_from_messages(last.prompt_messages)
            response = last.response_text

            queries.append(prompt)
            responses.append(response)
            rewards.append(rhat)

        query_tensors = [tokenizer(q, return_tensors="pt").input_ids.squeeze(0) for q in queries]
        response_tensors = [tokenizer(r, return_tensors="pt").input_ids.squeeze(0) for r in responses]
        reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        print(f"PPO update {step}")
        for i, ep in enumerate(episodes):
            print(f"Episode {i}: tilde={ep.evaluator_completed}  rhat={corrected_rewards[i]:.3f}")
            print("Justification:", ep.evaluator_justification[:200])

        step += 1

        if step % 5 == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            ppo_trainer.save_pretrained(cfg.output_dir)

        print(stats)

if __name__ == "__main__":
    main()
