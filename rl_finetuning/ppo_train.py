import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from logprobs import sequence_logprob
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from reward import NoiseRates, corrected_reward
from rollout_env import DesktopRolloutEnv
from transformers import AutoModelForCausalLM, AutoProcessor

from agents.UI_TARS.config import TASK


@dataclass
class TrainCfg:
    policy_model_name_or_path: str = os.getenv("PPO_POLICY_MODEL", "ByteDance-Seed/UI-TARS-1.5-7B")
    output_dir: str = os.getenv("PPO_OUT", "ppo_uinars_lora_mm")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    episodes_per_update: int = 2
    max_steps_per_episode: int = 8

    # Evaluator metrics noise
    e_plus: float = float(os.getenv("E_PLUS", "0.10"))   # FN
    e_minus: float = float(os.getenv("E_MINUS", "0.05")) # FP

    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.00 
    entropy_coef: float = 0.00
    max_new_tokens: int = 256


def instruction_text_from_messages(messages: List[dict]) -> str:
    user_parts = messages[0]["content"]
    for p in user_parts:
        if p.get("type") == "text":
            return p["text"]
    return ""


def build_mm_messages(screenshot_pil: Image.Image, instruction_text: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot_pil},
                {"type": "text", "text": instruction_text},
            ],
        }
    ]


def build_mm_inputs(processor: AutoProcessor, mm_messages: List[Dict[str, Any]], device: str):
    """
    Returns dict with input_ids + vision tensors (pixel values etc.)
    """
    text = processor.apply_chat_template(mm_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(mm_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}



@torch.no_grad()
def generate_with_ids(model, processor, inputs, max_new_tokens: int) -> Tuple[str, torch.Tensor]:
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)

    gen = out[:, inputs["input_ids"].shape[-1]:]
    text = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
    return text, gen


def ppo_update(
    model,
    optimizer,
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float,
) -> Dict[str, float]:
    """
    L = - min(ratio*A, clip(ratio,1-eps,1+eps)*A)
    """
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -(torch.min(unclipped, clipped))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "ratio": float(ratio.detach().cpu()),
        "old_logp": float(old_logp.detach().cpu()),
        "new_logp": float(new_logp.detach().cpu()),
        "adv": float(advantage.detach().cpu()),
    }


def main():
    cfg = TrainCfg()
    rates = NoiseRates(e_plus=cfg.e_plus, e_minus=cfg.e_minus)
    assert rates.D > 0.0, f"Separability condition for D violated: D={rates.D} must be >0"

    device = cfg.device

    processor = AutoProcessor.from_pretrained(cfg.policy_model_name_or_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    env = DesktopRolloutEnv(task=TASK, max_steps=cfg.max_steps_per_episode)

    step = 0
    while True:
        episodes = [env.run_episode(lambda msgs: "DUMMY") for _ in range(cfg.episodes_per_update)]

        batch_stats = []

        for ep in episodes:
            last = ep.transitions[-1]

            screenshot_pil = Image.open(last.screenshot_path).convert("RGB")

            instr = instruction_text_from_messages(last.prompt_messages)

            mm_messages = build_mm_messages(screenshot_pil, instr)
            inputs = build_mm_inputs(processor, mm_messages, device=device)

            with torch.no_grad():
                response_text, gen_ids = generate_with_ids(
                    model, processor, inputs, max_new_tokens=cfg.max_new_tokens
                )
                old_logp = sequence_logprob(model, inputs, gen_ids)

            tilde = float(ep.evaluator_completed)
            rhat = corrected_reward(tilde, rates)

            advantage = torch.tensor(rhat, device=device, dtype=torch.float32)

            new_logp = sequence_logprob(model, inputs, gen_ids)

            stats = ppo_update(
                model=model,
                optimizer=optimizer,
                old_logp=old_logp,
                new_logp=new_logp,
                advantage=advantage,
                clip_eps=cfg.clip_eps,
            )
            stats["tilde"] = tilde
            stats["rhat"] = float(rhat)
            batch_stats.append(stats)

            print("\nEpisode:")
            print("tilde =", tilde, " rhat =", rhat)
            print("evaluator justification:", ep.evaluator_justification[:200])
            print("sampled response head:", response_text[:200])

        step += 1
        if step % 5 == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            model.save_pretrained(cfg.output_dir)
            processor.save_pretrained(cfg.output_dir)

        print("\nUpdate", step)


if __name__ == "__main__":
    main()
