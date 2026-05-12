#!/usr/bin/env python3
"""
================================================================================
Exp 2 FINAL Standalone — Needle in Haystack with Memory Management
1000 sentences, 5 same-format decoys, optimized for T4 16GB.
Self-contained: no git clone, no external files.
Run in Kaggle: Paste the cell below
================================================================================
"""
import os
# CRITICAL: Must be set BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/tmp/hf_cache"

import random
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

FILLERS = [
    "The history of pottery spans thousands of years.",
    "Marine biologists study coral reef ecosystems.",
    "Railway engineering requires precise curvature.",
    "The periodic table arranges elements by number.",
    "Clouds are classified as cumulus and stratus.",
    "Beekeeping traditions differ between continents.",
    "The Great Wall was built over many dynasties.",
    "Thermodynamics governs heat transfer.",
    "Impressionist painters captured fleeting light.",
    "Volcanic activity is tracked with seismographs.",
    "The Dewey Decimal System organizes libraries.",
    "Irrigation evolved from canals to drip systems.",
    "Neural networks are inspired by biological brains.",
    "Light speed is 299,792,458 meters per second.",
    "Classical composition follows harmonic rules.",
    "Urban planning addresses zoning and transport.",
    "Photosynthesis converts CO2 into glucose.",
    "The Fibonacci sequence appears in nature.",
    "GPS uses triangulation from satellites.",
    "Cryptography secures digital communication.",
]

NUM_DECOYS = 5
NUM_SENTENCES = 1000
NUM_EXAMPLES = 30


def make_haystack(n_sents, target_code, depth, num_decoys=NUM_DECOYS):
    sents = []
    target_idx = int(depth * n_sents)
    available = [i for i in range(n_sents) if abs(i - target_idx) > 20]
    
    if num_decoys > 0 and len(available) >= num_decoys:
        decoy_positions = set(random.sample(available, num_decoys))
    else:
        decoy_positions = set()
    
    for i in range(n_sents):
        if i in decoy_positions:
            decoy_code = f"CL-{random.randint(10000, 99999)}"
            sents.append(f"The classified identifier is {decoy_code}.")
        else:
            sents.append(random.choice(FILLERS))
    
    idx = int(depth * len(sents))
    sents.insert(idx, f"The classified identifier is {target_code}.")
    return " ".join(sents)


def ask(text, target_code):
    prompt = (
        f"Read the text carefully. Several codes appear, but only one is "
        f"the true classified identifier. Find it.\n\n"
        f"{text}\n\n"
        f"What is the classified identifier? Answer with only the code (including CL- prefix)."
    )
    msg = [{"role": "user", "content": prompt}]
    txt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = tok(txt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
    
    ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # CRITICAL: Free memory to prevent fragmentation
    del out, inp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return ans


def position_bias_index(depths, accs):
    start = accs[0]
    end = accs[-1]
    middle = accs[len(accs) // 2]
    return (start + end) / 2 - middle


depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

print("=" * 70)
print(f"EXP 2: Needle in Haystack")
print(f"Model: {MODEL} | Sentences: {NUM_SENTENCES} | Decoys: {NUM_DECOYS}")
print(f"Examples: {NUM_EXAMPLES}")
print("=" * 70)

results = {}
start_time = time.time()

for depth in depths:
    correct = 0
    for i in tqdm(range(NUM_EXAMPLES), desc=f"Depth {depth:.0%}"):
        code = f"CL-{random.randint(10000, 99999)}"
        text = make_haystack(NUM_SENTENCES, code, depth)
        ans = ask(text, code)
        if ans == code:
            correct += 1
        if i < 2:
            status = "✓" if ans == code else "✗"
            print(f"  Ex {i}: expected '{code}', got '{ans}' {status}")
    
    acc = correct / NUM_EXAMPLES
    results[depth] = acc
    print(f"Depth {depth:5.1%}: accuracy = {acc:.3f}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

elapsed = (time.time() - start_time) / 60
accs = [results[d] for d in depths]
pbi = position_bias_index(depths, accs)

print(f"\n{'='*70}")
print(f"EXP 2 COMPLETE | Time: {elapsed:.1f} min | PBI = {pbi:+.3f}")
for d, a in zip(depths, accs):
    print(f"  Depth {d:5.1%}: {a:.3f}")
print(f"{'='*70}")

# Save
out_dir = "/kaggle/working/exp2_results"
os.makedirs(out_dir, exist_ok=True)
summary = {
    "experiment": "needle_in_haystack",
    "model": MODEL,
    "num_sentences": NUM_SENTENCES,
    "num_decoys": NUM_DECOYS,
    "num_examples": NUM_EXAMPLES,
    "depths": {str(k): v for k, v in results.items()},
    "pbi": pbi,
    "time_minutes": elapsed,
}
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {out_dir}/summary.json")
