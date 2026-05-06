# Kaggle Notebook Usage

This folder contains standalone scripts for running individual experiments in Kaggle notebooks.

## Quick Start (Kaggle T4)

In a Kaggle notebook cell, clone the repo and run any experiment:

```python
# Cell 1: Clone and setup
!git clone https://huggingface.co/abhshkp/litm-benchmark-suite-v4 litm
%cd litm
!pip install -q -r requirements.txt

# Cell 2: Run a single experiment
!python kaggle/run_exp1a_kv100.py --model Qwen/Qwen2.5-1.5B-Instruct --n-examples 50

# Or run the full suite
!python run_all.py --model Qwen/Qwen2.5-1.5B-Instruct --output /kaggle/working/results --zip
```

## Individual Experiment Scripts

| Script | Experiment | Approx. Time (T4) |
|--------|-----------|-------------------|
| `run_exp1a_kv100.py` | KV Retrieval (100 keys) | ~15 min |
| `run_exp1b_kv200.py` | KV Retrieval (200 keys) | ~25 min |
| `run_exp2_needle.py` | Needle in Haystack | ~20 min |
| `run_exp3_multi.py` | Multi-Needle | ~15 min |
| `run_exp4_reason.py` | Fact-Dependent Reasoning | ~20 min |
| `run_exp5_semantic.py` | Semantic Similarity Distractors | ~15 min |
| `run_exp6_narrative.py` | Temporal Narrative | ~15 min |
| `run_exp7_conversation.py` | Conversation Memory | ~15 min |

## Downloading Results

All scripts write to `/kaggle/working/litm_results/` by default. After running:

```python
# Cell 3: Zip and download
import shutil
shutil.make_archive("/kaggle/working/litm_results", "zip", "/kaggle/working/litm_results")
# Download from the Output tab
```

## Running Multiple Experiments

Each script is completely independent — you can run them in separate Kaggle sessions or sequentially in one notebook. They share no state (model is reloaded each time), so no conflicts.
