"""Model loading with 4-bit quantization for T4/GPU inference."""
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

_model_cache = {}
_tok_cache = {}


def load_model(model_name: str, load_in_4bit: bool = True, device_map: str = "auto"):
    """Load model with optional 4-bit quantization. Cached for reuse."""
    cache_key = f"{model_name}:{load_in_4bit}:{device_map}"
    if cache_key in _model_cache:
        return _model_cache[cache_key], _tok_cache[cache_key]

    logger.info(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    dev = next(model.parameters()).device
    logger.info(f"Model loaded on {dev}")

    _model_cache[cache_key] = model
    _tok_cache[cache_key] = tok
    return model, tok
