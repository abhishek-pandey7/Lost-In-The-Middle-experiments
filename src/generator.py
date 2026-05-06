"""Text generation wrapper with chat-template support."""
import logging
import torch
from typing import List, Dict, Any
from .model_loader import load_model

logger = logging.getLogger(__name__)


def generate_text(
    messages: List[Dict[str, str]],
    model_name: str,
    max_new_tokens: int = 80,
    load_in_4bit: bool = True,
) -> str:
    """Generate text from a chat-formatted message list."""
    model, tokenizer = load_model(model_name, load_in_4bit=load_in_4bit)

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )

    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()
