from typing import Dict

import torch
import torch.nn.functional as F


def sequence_logprob(
    model,
    inputs: Dict[str, torch.Tensor],
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log p(generated_ids | inputs) under the model.
    """
    input_ids = inputs["input_ids"]
    full_ids = torch.cat([input_ids, generated_ids], dim=-1)

    model_inputs = dict(inputs)
    model_inputs["input_ids"] = full_ids

    outputs = model(**model_inputs)
    logits = outputs.logits

    prompt_len = input_ids.shape[-1]
    gen_len = generated_ids.shape[-1]

    relevant_logits = logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
    logprobs = F.log_softmax(relevant_logits, dim=-1)

    token_logprobs = logprobs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum(dim=-1).squeeze(0)
