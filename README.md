# Lost in the Middle — Benchmark Suite v4

A modular, reproducible benchmark suite for evaluating **position bias** in long-context language models, extending the original Liu et al. (2023) experiments.

## Paper Reference

> **"Lost in the Middle: How Language Models Use Long Contexts"**  
> Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang  
> arXiv:2307.03172

## What This Suite Measures

Position bias (also called "Lost in the Middle" effect): LLMs perform best when relevant information is at the **beginning** or **end** of a long context, and worst when it is in the **middle**.

## 7 Experiments

| # | Experiment | What It Tests | Position Bias Type |
|---|-----------|-------------|-------------------|
| 1a | **KV Retrieval (100 keys)** | Exact-match retrieval from structured data | Classic U-shape |
| 1b | **KV Retrieval (200 keys)** | Same, with longer context | U-shape amplification |
| 2 | **Needle in Haystack** | Secret code hidden in prose | Text retrieval |
| 3 | **Multi-Needle** | 3 codes at start/middle/end | Selective multi-retrieval |
| 4 | **Fact-Dependent Reasoning** | Math problem requiring buried fact | Reasoning × retrieval |
| 5 | **Semantic Similarity Distractors** | Gold fact among similar-looking facts | Associative interference |
| 6 | **Temporal Narrative** | Event recall from chronology | Temporal ordering |
| 7 | **Conversation Memory** | Critical instruction in chat history | Dialog state tracking |

## Position Bias Index (PBI)

We compute a unified metric across all experiments:

```
PBI = (accuracy_start + accuracy_end) / 2 - accuracy_middle
```

- **PBI > 0.3**: Strong U-shape (bad)
- **PBI ≈ 0**: Flat curve (good)
- **PBI < 0**: Inverted bias (rare)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python run_all.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./results

# Run single experiment
python run_all.py --experiments needle --output ./results

# Run on Kaggle (T4 GPU)
python run_all.py --model Qwen/Qwen2.5-1.5B-Instruct --output /kaggle/working/results --zip
```

## Output Structure

```
results/
├── exp1a_kv100/
│   ├── kv100_data.jsonl
│   ├── kv100_pos_*.jsonl
│   ├── kv100_summary.json
│   └── kv100_curve.png
├── exp2_needle/
│   ├── needle_depth_*.jsonl
│   ├── needle_summary.json
│   └── needle_curve.png
├── ... (exp3-7)
└── master_summary.json
```

## Modules

- `src/model_loader.py` — 4-bit quantized model loading with caching
- `src/generator.py` — Chat-template text generation
- `src/metrics.py` — Scoring and Position Bias Index
- `src/plotting.py` — Standardized curve/bar plots
- `src/utils.py` — JSONL/JSON I/O helpers
- `experiments/` — Self-contained experiment modules

## Extending

To add a new experiment:

1. Create `experiments/my_experiment.py` with `run_my_experiment(model_name, ..., out_dir)`
2. Import in `run_all.py`
3. Add to `main()` with a new `--experiments` flag

## Citation

```bibtex
@software{litm_benchmark_suite_v4,
  title={Lost in the Middle Benchmark Suite v4},
  author={abhshkp},
  year={2026},
  url={https://huggingface.co/abhshkp/litm-benchmark-suite-v4}
}
```
