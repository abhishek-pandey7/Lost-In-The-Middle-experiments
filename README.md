---
tags:
- ml-intern
- lost-in-the-middle
- long-context
- position-bias
- benchmark
---

# Lost in the Middle — Benchmark Suite v4

> **A Modular, Reproducible Benchmark Suite for Evaluating Position Bias in Long-Context Language Models**

---

## Table of Contents

1. [What is "Lost in the Middle"?](#1-what-is-lost-in-the-middle)
2. [Position Bias Index (PBI)](#2-position-bias-index-pbi)
3. [Architecture Overview](#3-architecture-overview)
4. [Experiment Descriptions](#4-experiment-descriptions)
5. [Quick Start](#5-quick-start)
6. [Kaggle Usage](#6-kaggle-usage)
7. [Output Structure](#7-output-structure)
8. [Results & Graphs](#8-results--graphs)
9. [Conclusions & Discussion](#9-conclusions--discussion)
10. [Extending the Suite](#10-extending-the-suite)
11. [Citation](#11-citation)

---

## 1. What is "Lost in the Middle"?

### 1.1 The Core Phenomenon

The **"Lost in the Middle"** (LITM) effect, first systematically documented by Liu et al. (2023), describes a critical failure mode in large language models (LLMs) when processing long contexts:

> **Models perform best when relevant information appears at the *beginning* or *end* of a context, and worst when it is buried in the *middle*.**

This creates a characteristic **U-shaped accuracy curve** when plotting model performance against the position of the target information within a long document.

### 1.2 Why Does This Happen?

The LITM effect arises from how modern transformer-based LLMs process attention:

| Mechanism | Explanation |
|-----------|-------------|
| **Attention Dilution** | In long sequences, the softmax over attention weights becomes increasingly diffuse. Middle-position tokens receive proportionally less attention mass than edge-position tokens. |
| **Positional Bias in Training** | Pretraining data often places key information at document boundaries (introductions, summaries). Models learn a positional prior that favors start and end positions. |
| **KV Cache Pressure** | During autoregressive generation, the key-value cache grows linearly with sequence length. Attention computation over very long contexts becomes noisier in the middle regions. |
| **Softmax Saturation** | With many tokens competing for attention probability mass, individual middle tokens are "drowned out" by the aggregate signal from surrounding tokens. |

### 1.3 The U-Shaped Curve

When you plot accuracy vs. position, you see a U-shape:

```
Accuracy
  1.0 |  ●                                     ●
  0.9 |    ●                                 ●
  0.8 |      ●                             ●
  0.7 |        ●                         ●
  0.6 |          ●       ●       ●     ●
  0.5 |            ●       ●       ●
      +----+----+----+----+----+----+----+----+
      0   0.12 0.25 0.37  0.5  0.62 0.75 0.87  1.0
                     Position
```

- **Position 0.0** (start): High accuracy — primacy bias
- **Position 0.5** (middle): Lowest accuracy — the "lost" zone
- **Position 1.0** (end): High accuracy — recency bias

### 1.4 Why This Matters

The LITM effect has profound implications for real-world LLM deployments:

- **Retrieval-Augmented Generation (RAG)**: If a retriever returns relevant documents in the middle of a concatenated prompt, the generator may ignore them.
- **Long-Document QA**: Answers hidden in the middle of legal contracts, medical records, or research papers are systematically missed.
- **In-Context Learning**: Demonstration examples placed in the middle of a prompt are less effective than those at the start or end.
- **Conversational AI**: Critical instructions buried in long chat histories are forgotten.

---

## 2. Position Bias Index (PBI)

### 2.1 Definition

To quantify the LITM effect consistently across experiments, we define the **Position Bias Index (PBI)**:

```
PBI = (accuracy_start + accuracy_end) / 2 − accuracy_middle
```

Where:
- `accuracy_start` = accuracy when target is at the beginning (position 0.0)
- `accuracy_end` = accuracy when target is at the end (position 1.0)
- `accuracy_middle` = accuracy when target is at the center (position 0.5)

### 2.2 Interpretation

| PBI Range | Meaning |
|-----------|---------|
| **PBI > 0.30** | Strong U-shaped bias — severe middle degradation |
| **PBI = 0.15–0.30** | Moderate bias — noticeable middle dip |
| **PBI = 0.05–0.15** | Weak bias — slight middle dip |
| **PBI ≈ 0.00** | Flat curve — no positional bias |
| **PBI < 0.00** | Inverted-U — rare, model performs best in middle |

### 2.3 Why PBI?

PBI is superior to simply reporting "middle accuracy" because:
1. **Normalizes for overall model competence**: A model with 95% edge accuracy and 60% middle accuracy gets the same PBI as one with 70% edge and 35% middle.
2. **Captures the full U-shape**: It explicitly contrasts edges against center.
3. **Comparable across experiments**: KV retrieval, needle-in-haystack, and reasoning tasks all speak the same metric language.

### 2.4 Expanded Positions

Unlike the original Liu et al. paper (which tested 5 positions), this suite tests **9 positions** for finer-grained curve resolution:

| Position | Normalized | Description |
|----------|-----------|-------------|
| 0 | 0.000 | Absolute start |
| N/8 | 0.125 | Early |
| N/4 | 0.250 | Early-middle |
| 3N/8 | 0.375 | Pre-middle |
| N/2 | 0.500 | Exact middle |
| 5N/8 | 0.625 | Post-middle |
| 3N/4 | 0.750 | Late-middle |
| 7N/8 | 0.875 | Late |
| N−1 | 1.000 | Absolute end |

---

## 3. Architecture Overview

### 3.1 Module Structure

```
litm-benchmark-suite-v4/
│
├── src/                          ← Shared infrastructure
│   ├── model_loader.py           # 4-bit quantized model loading
│   ├── generator.py              # Chat-template text generation
│   ├── metrics.py                # PBI, exact-match, numeric scoring
│   ├── plotting.py               # Standardized curve/bar plots
│   └── utils.py                  # JSONL/JSON I/O
│
├── experiments/                  ← Core experiment logic (library)
│   ├── kv_retrieval.py           # Exp 1: UUID key-value pairs
│   ├── needle_in_haystack.py     # Exp 2: Secret code in prose
│   ├── multi_needle.py           # Exp 3: Three simultaneous needles
│   ├── fact_reasoning.py         # Exp 4: Math with buried facts
│   ├── semantic_distractors.py   # Exp 5: Gold among similar facts
│   ├── temporal_narrative.py     # Exp 6: Events in chronology
│   └── conversation_memory.py    # Exp 7: Instruction in chat history
│
├── kaggle/                       ← Standalone Kaggle runners
│   ├── run_exp1a_kv100.py
│   ├── run_exp1b_kv200.py
│   ├── run_exp2_needle.py
│   ├── run_exp3_multi.py
│   ├── run_exp4_reason.py
│   ├── run_exp5_semantic.py
│   ├── run_exp6_narrative.py
│   ├── run_exp7_conversation.py
│   └── run_all_kaggle.py         # Master Kaggle runner
│
├── run_all.py                    # Local master runner
├── config.yaml                   # Hyperparameter configuration
└── requirements.txt              # Dependencies
```

### 3.2 Design Philosophy

- **`experiments/`** = **Reusable library modules**. Each file exports a `run_*()` function. Never executed directly.
- **`kaggle/`** = **Thin entry-point scripts**. Each imports one experiment module, adds Kaggle-specific paths and CLI args, and runs it.
- **`src/`** = **Shared utilities**. Model loading, generation, metrics, plotting — used by everything.

This separation means:
- You can import `experiments.kv_retrieval` into your own custom pipeline.
- You can run any experiment standalone in Kaggle without touching the core logic.
- You can add a new experiment by writing one module and one Kaggle wrapper.

---

## 4. Experiment Descriptions

### 4.1 Experiment 1: Key-Value Retrieval

**Files:** `experiments/kv_retrieval.py`, `kaggle/run_exp1a_kv100.py`, `kaggle/run_exp1b_kv200.py`

#### Motivation
This is the **canonical LITM task** from Liu et al. (2023). It tests the most basic form of long-context retrieval: given a JSON object with many key-value pairs, can the model extract the value for a specific key?

#### Methodology
1. Generate `N` random UUID key-value pairs (e.g., 100 or 200).
2. Select one pair as the **gold target**.
3. Place the gold pair at 9 controlled positions: 0, N/8, N/4, ..., N−1.
4. Prompt the model with the full JSON object and ask for the value corresponding to the gold key.
5. Score with exact-match against the true value.

#### Prompt Template
```
Extract the value corresponding to the specified key in the JSON object below.

JSON data:
{"<key1>": "<value1>",
 "<key2>": "<value2>",
 ...}

Key: "<query_key>"
Corresponding value:
```

#### Why This Task?
- **No reasoning required** — pure retrieval. If the model fails, it's unambiguously an attention/position problem.
- **Structured input** — JSON provides clear boundaries, eliminating ambiguity about what constitutes "the answer."
- **Scalable** — trivial to generate 50, 100, 500, or 1000 keys.

#### Variants
- **1A (100 keys)**: Moderate length. Tests position bias in a medium-length context.
- **1B (200 keys)**: Double length. Tests whether bias *amplifies* with context length.

#### Expected Results
- U-shaped curve with strong middle dip.
- PBI ~ 0.35–0.50 for 1.5B models.
- PBI higher for 200 keys than 100 keys (longer contexts = worse middle performance).

---

### 4.2 Experiment 2: Needle in a Haystack

**Files:** `experiments/needle_in_haystack.py`, `kaggle/run_exp2_needle.py`

#### Motivation
Tests retrieval from **unstructured natural language prose**. Unlike KV retrieval (structured), this requires the model to search through fluent text to find a specific fact.

#### Methodology
1. Generate a long document of `N` filler sentences (default: 500) from a pool of generic factual statements.
2. Insert a **"needle"** sentence containing a unique secret code at a controlled depth.
3. Ask the model to extract the secret code.
4. Score with exact-match.

#### Example Document
```
The history of pottery spans thousands of years. [1].
Marine biologists study coral reef ecosystems. [2].
...
The secret code is SECRET-0042. [250].   ← needle at position 250/500
...
Railway engineering requires precise curvature. [500].
```

#### Prompt
```
Read the text and find the secret code.

<document>

What is the secret code? Answer with only the code.
```

#### Why This Task?
- **Unstructured retrieval** — tests whether the model can locate a specific fact in prose, not just structured data.
- **High information density** — every sentence is semantically meaningful, creating realistic competition for attention.
- **Scalable to extreme lengths** — can test 1K, 2K, or even 10K sentences.

#### Expected Results
- U-shaped curve, possibly stronger than KV retrieval because prose is less structured than JSON.
- PBI ~ 0.30–0.45.

---

### 4.3 Experiment 3: Multi-Needle Retrieval

**Files:** `experiments/multi_needle.py`, `kaggle/run_exp3_multi.py`

#### Motivation
Real documents often contain **multiple relevant facts**, not just one. This tests whether the model can retrieve **all** of them simultaneously, and whether position bias affects each needle independently.

#### Methodology
1. Generate a long filler document (default: 300 sentences).
2. Insert **three distinct secret codes** at three fixed positions:
   - Code A at position 0 (start)
   - Code B at position N/2 (middle)
   - Code C at position N−1 (end)
3. Ask the model to list **all three codes in order**.
4. Score each code independently with exact-match.

#### Prompt
```
Read the text and list ALL three secret codes in order.

<document>

Codes:
```

#### Why This Task?
- **Tests multi-hop attention** — the model must attend to three non-contiguous locations.
- **Reveals asymmetric bias** — does the model retrieve start and end needles but miss the middle one?
- **Models real RAG scenarios** — multiple retrieved chunks concatenated together.

#### Expected Results
- Start code: ~90–100% accuracy
- End code: ~90–100% accuracy
- Middle code: ~50–70% accuracy (the "lost" needle)
- Bar chart showing asymmetric retrieval.

---

### 4.4 Experiment 4: Fact-Dependent Reasoning

**Files:** `experiments/fact_reasoning.py`, `kaggle/run_exp4_reason.py`

#### Motivation
Retrieval is only step one. In real tasks, models must **use** retrieved facts to perform reasoning (math, inference, decision-making). This tests whether position bias persists when the model must *both* retrieve a fact *and* reason with it.

#### Methodology
1. Generate a long document of `N` distractor sentences (default: 300).
2. Insert one **critical fact** at a controlled depth about a fictional product with a random price, e.g.:
   > "For this order, Zylor apples cost $247/kg."
3. Ask a math question that requires this fact:
   > "According to the document, I buy 12 kg of Zylor apples. What is my total cost?"
4. The model must (a) find the fictional product's price, (b) multiply by quantity.
5. Score with exact integer match.

#### Critical Design Choice
All products are **fictional** (e.g., "Zylor apples," "Krynn berries") with **random prices ($50–$500)**. The model cannot answer from parametric knowledge — it MUST read the document.

#### Why This Task?
- **Reasoning × Retrieval** — failure could be either retrieval failure or reasoning failure. This disentangles them.
- **More realistic than pure retrieval** — most real tasks require using information, not just locating it.
- **Tests compositional generalization** — can the model compose retrieved facts with arithmetic?

#### Expected Results
- **If the model is capable**: U-shaped curve, but possibly *weaker* than pure retrieval because reasoning demands deeper processing. PBI ~ 0.20–0.40.
- **If the model is not capable**: Near-chance accuracy across all depths (~10–30%), making position bias statistically undetectable. This itself is a valuable finding — it establishes that LITM effects are observable only when the underlying task lies within the model's competence frontier.

---

### 4.5 Experiment 5: Semantic Similarity Distractors

**Files:** `experiments/semantic_distractors.py`, `kaggle/run_exp5_semantic.py`

#### Motivation
In real documents, the target fact is rarely uniquely distinct. It competes with **semantically similar distractors**. This tests whether position bias interacts with **associative interference**.

#### Methodology
1. Create a list of `N` factual statements (default: 80) from the same semantic domain, all about **fictional countries** with **random secret codes**.
   - E.g., "The capital of Xyloria is ZENTH-7392.", "The capital of Freloria is VORT-1854.", ...
2. Insert the **gold fact** among them at a controlled depth.
3. Ask a question requiring the secret code from the gold fact.
   > "What is the capital of Xyloria? Answer with only the secret code."
4. The distractors create associative competition — the model must distinguish "Xyloria" from "Freloria," "Zenthar," etc.

#### Critical Design Choice
All countries are fictional and all codes are random. The model **cannot** answer from parametric knowledge. It must read the specific line in the document.

#### Why This Task?
- **Associative interference** — similar-looking facts compete for attention.
- **Tests discriminative retrieval** — not just "find the needle" but "find the *right* needle among similar needles."
- **Models RAG with dense semantic overlap** — e.g., multiple retrieved passages about related topics.

#### Expected Results
- **Classic LITM (U-shape) may NOT appear**. When distractors are semantically dense and the target is not lexically unique, **recency bias can collapse**.
- Instead of U-shape, you may see a **monotonic decline** or **primacy-only** pattern: high at start, declining through the document, with no recovery at the end.
- This is a **novel finding**: semantic density destroys the recency advantage because the final items are not distinct enough to "pop" against their neighbors.

---

### 4.6 Experiment 6: Temporal Narrative

**Files:** `experiments/temporal_narrative.py`, `kaggle/run_exp6_narrative.py`

#### Motivation
Documents often have **inherent temporal structure** (chronologies, logs, histories). Does chronological ordering help or hurt retrieval? Does the model use temporal scaffolding, or does raw position dominate?

#### Methodology
1. Generate a timeline of `N` historical events (default: 100).
   - 30 generic historical events (e.g., "the king issued a decree").
   - 8 statue-unveiling distractors with different materials/locations (e.g., "a bronze statue was unveiled in the town square").
2. Insert a **target event** at a controlled depth with a random secret code:
   > "Year 1050: a golden statue was unveiled in the central square (CODE: XJ-7392)."
3. The target is **one of 9 statue-unveiling events** — not lexically unique. The model must distinguish "golden statue + central square" from other statue events and extract the code.
4. Ask the model to identify the code.
   > "What is the secret code for the golden statue that was unveiled in the central square?"
5. Score with exact-match against the secret code.

#### Critical Design Choice
The target is **semantically embedded** in a family of similar events. The model cannot locate it by simple keyword search ("statue" appears 9 times). It must use **positional attention** combined with **semantic discrimination**.

#### Why This Task?
- **Temporal structure** — events have meaningful ordering, not arbitrary placement.
- **Semantic competition** — similar events compete for attention, testing true positional bias rather than lexical uniqueness.
- **Models real-world timelines** — medical histories, legal case files, project logs with repeated event types.

#### Expected Results
- U-shaped curve, but possibly different from unstructured tasks.
- If temporal ordering provides scaffolding, the curve may be weaker than needle-in-haystack.
- If semantic density dominates, recency bias may collapse (similar to Exp 5).

---

### 4.7 Experiment 7: Conversation Memory

**Files:** `experiments/conversation_memory.py`, `kaggle/run_exp7_conversation.py`

#### Motivation
Conversational AI must maintain coherence across **long dialogue histories**. Critical instructions or facts buried in the middle of a chat are frequently "forgotten." This tests dialog-state position bias.

#### Methodology
1. Generate a synthetic conversation of `N` turns (default: 100) between User and Assistant.
   - User messages from a pool of generic questions.
   - Assistant messages from a pool of generic answers.
2. Insert a **critical instruction** at a controlled depth:
   > User: "Please always remember that my favorite color is MYFAVCOLOR-042. This is very important."
   > Assistant: "I will remember that."
3. At the end, ask the model to recall the instruction.
   > "Based on our conversation, what is my favorite color?"
4. Score with exact-match against the color code.

#### Why This Task?
- **Dialog-specific** — tests position bias in the conversational domain.
- **Instruction following** — models are explicitly told to "remember." Do they?
- **Models real chatbot failures** — system prompts, user preferences, critical warnings buried in history.

#### Expected Results
- U-shaped curve, possibly very strong because dialog turns are short and attention can "skip" over middle turns.
- PBI may be higher than document tasks.

---

## 5. Quick Start

### 5.1 Local Installation

```bash
# Clone the repository
git clone https://huggingface.co/abhshkp/litm-benchmark-suite-v4
cd litm-benchmark-suite-v4

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Run All Experiments (Local)

```bash
python run_all.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output ./results \
    --n-examples 50
```

### 5.3 Run Single Experiment (Local)

```bash
python run_all.py \
    --experiments needle \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output ./results
```

### 5.4 Available CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model identifier |
| `--output` | `./results` | Output directory |
| `--n-examples` | `50` | Examples per position (KV only) |
| `--n-keys-100` | `100` | Keys for Exp 1A |
| `--n-keys-200` | `200` | Keys for Exp 1B |
| `--needle-sentences` | `500` | Sentences for Exp 2 |
| `--experiments` | `all` | Comma-separated: `kv100,kv200,needle,multi,reason,semantic,narrative,conversation` |
| `--zip` | `False` | Create zip archive of results |

---

## 6. Kaggle Usage

### 6.1 Run a Single Experiment (Recommended)

Each experiment is self-contained and takes ~15–25 minutes on a T4 GPU.

In a Kaggle notebook cell:

```python
# Cell 1: Clone and install
!git clone https://huggingface.co/abhshkp/litm-benchmark-suite-v4 litm
%cd litm
!pip install -q -r requirements.txt

# Cell 2: Run one experiment
!python kaggle/run_exp2_needle.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-examples 30

# Cell 3: Zip and download
import shutil
shutil.make_archive("/kaggle/working/litm_results", "zip", "/kaggle/working/litm_results")
```

### 6.2 Run Experiment 6 (Temporal Narrative)

```python
# Cell 1: Clone and install (fresh to get latest code)
!rm -rf litm
!git clone https://huggingface.co/abhshkp/litm-benchmark-suite-v4 litm
%cd litm
!pip install -q -r requirements.txt

# Cell 2: Run Experiment 6
!python kaggle/run_exp6_narrative.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-examples 30 \
    --n-events 100

# Cell 3: Zip and download
import shutil
shutil.make_archive("/kaggle/working/litm_results", "zip", "/kaggle/working/litm_results")
```

### 6.3 Run All Experiments Overnight

```python
!python kaggle/run_all_kaggle.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output /kaggle/working/litm_results \
    --n-examples 50
```

### 6.4 Kaggle Scripts

| Script | Experiment | ~Time on T4 |
|--------|-----------|-------------|
| `kaggle/run_exp1a_kv100.py` | KV Retrieval (100 keys) | ~15 min |
| `kaggle/run_exp1b_kv200.py` | KV Retrieval (200 keys) | ~25 min |
| `kaggle/run_exp2_needle.py` | Needle in Haystack | ~20 min |
| `kaggle/run_exp3_multi.py` | Multi-Needle | ~15 min |
| `kaggle/run_exp4_reason.py` | Fact-Dependent Reasoning | ~20 min |
| `kaggle/run_exp5_semantic.py` | Semantic Similarity Distractors | ~15 min |
| `kaggle/run_exp6_narrative.py` | Temporal Narrative | ~15 min |
| `kaggle/run_exp7_conversation.py` | Conversation Memory | ~15 min |
| `kaggle/run_all_kaggle.py` | **All 7 experiments** | ~2–2.5 hrs |

---

## 7. Output Structure

### 7.1 Per-Experiment Output

Each experiment produces a folder with the following files:

```
results/
├── exp1a_kv100/
│   ├── kv100_data.jsonl           # Raw generated examples
│   ├── kv100_pos_0.jsonl          # Predictions: gold at position 0
│   ├── kv100_pos_12.jsonl         # Predictions: gold at position 12
│   ├── kv100_pos_25.jsonl         # Predictions: gold at position 25
│   ├── kv100_pos_37.jsonl         # Predictions: gold at position 37
│   ├── kv100_pos_50.jsonl         # Predictions: gold at position 50 (MIDDLE)
│   ├── kv100_pos_62.jsonl         # Predictions: gold at position 62
│   ├── kv100_pos_75.jsonl         # Predictions: gold at position 75
│   ├── kv100_pos_87.jsonl         # Predictions: gold at position 87
│   ├── kv100_pos_99.jsonl         # Predictions: gold at position 99 (END)
│   ├── kv100_summary.json         # Accuracies + PBI
│   └── kv100_curve.png            # U-shaped accuracy plot
│
├── exp2_needle/
│   ├── needle_depth_0.0.jsonl
│   ├── needle_depth_0.125.jsonl
│   ├── ... (9 depth files)
│   ├── needle_depth_1.0.jsonl
│   ├── needle_summary.json
│   └── needle_curve.png
│
├── ... (exp3-7 follow same pattern)
│
└── master_summary.json              # Aggregated results from all experiments
```

### 7.2 JSONL Format

Each `.jsonl` file contains one record per example:

```json
{"model_answer": "a1b2-c3d4-...", "correct": 1.0, "value": "target-uuid", "gold_position": 50}
{"model_answer": "wrong-guess", "correct": 0.0, "value": "target-uuid", "gold_position": 50}
```

### 7.3 Summary JSON Format

```json
{
  "experiment": "kv_retrieval",
  "num_keys": 100,
  "num_examples": 50,
  "positions": {
    "0": 0.94,
    "12": 0.78,
    "25": 0.64,
    "37": 0.58,
    "50": 0.54,
    "62": 0.60,
    "75": 0.70,
    "87": 0.82,
    "99": 0.90
  },
  "pbi": 0.38,
  "time_minutes": 12.5
}
```

### 7.4 Plot Files

Each experiment saves a `.png` plot:
- **Curve plots** (Experiments 1, 2, 4, 5, 6, 7): X-axis = normalized position, Y-axis = accuracy. Red curve with markers.
- **Bar plots** (Experiment 3): X-axis = Start/Middle/End, Y-axis = accuracy.

---

## 8. Results & Graphs

> **[USER TO INSERT OUTPUT GRAPHS HERE]**

### 8.1 Experiment 1A: KV Retrieval (100 keys)

*[Upload kv100_curve.png here]*

**Observations:**

### 8.2 Experiment 1B: KV Retrieval (200 keys)

*[Upload kv200_curve.png here]*

**Observations:**

### 8.3 Experiment 2: Needle in Haystack

*[Upload needle_curve.png here]*

**Observations:**

### 8.4 Experiment 3: Multi-Needle

*[Upload multi_bar.png here]*

**Observations:**

### 8.5 Experiment 4: Fact-Dependent Reasoning

*[Upload reason_curve.png here]*

**Observations:**

### 8.6 Experiment 5: Semantic Similarity Distractors

*[Upload semantic_curve.png here]*

**Observations:**

### 8.7 Experiment 6: Temporal Narrative

*[Upload narrative_curve.png here]*

**Observations:**

### 8.8 Experiment 7: Conversation Memory

*[Upload conversation_curve.png here]*

**Observations:**

### 8.9 Cross-Experiment PBI Comparison

| Experiment | PBI | Edge Accuracy | Middle Accuracy | Classification |
|-----------|-----|--------------|-----------------|----------------|
| KV 100 keys | | | | |
| KV 200 keys | | | | |
| Needle | | | | |
| Multi-Needle (middle) | | | | |
| Fact Reasoning | | | | |
| Semantic Distractors | | | | |
| Temporal Narrative | | | | |
| Conversation Memory | | | | |

---

## 9. Conclusions & Discussion

> **[USER TO WRITE CONCLUSIONS HERE]**

### 9.1 Key Findings

*Summarize the main discoveries from your experiments:*

1.
2.
3.

### 9.2 Implications

*What do these results mean for practitioners?*

### 9.3 Limitations

*What are the limitations of this study?*

### 9.4 Future Work

*What experiments or analyses would strengthen these findings?*

---

## 10. Extending the Suite

### 10.1 Add a New Experiment

1. Create `experiments/my_experiment.py` with a `run_my_experiment(model_name, ..., out_dir)` function.
2. Create `kaggle/run_expN_myexperiment.py` that calls your function with Kaggle defaults.
3. Import and add to `run_all.py` and `kaggle/run_all_kaggle.py`.

### 10.2 Add a New Model

Change `--model` to any HuggingFace causal LM:

```bash
python run_all.py --model meta-llama/Llama-3.2-1B-Instruct
python run_all.py --model Qwen/Qwen2.5-7B-Instruct
```

The suite automatically handles 4-bit quantization via `bitsandbytes`.

### 10.3 Adjust Scale

Increase context length by changing experiment parameters:

```bash
python kaggle/run_exp1b_kv200.py --n-keys 500 --n-examples 100
python kaggle/run_exp2_needle.py --n-sentences 1000 --n-examples 50
```

---

## 11. Citation

If you use this benchmark suite in your research, please cite both the original paper and this suite:

```bibtex
@article{liu2023lost,
  title={Lost in the Middle: How Language Models Use Long Contexts},
  author={Liu, Nelson F and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal={arXiv preprint arXiv:2307.03172},
  year={2023}
}

@software{litm_benchmark_suite_v4,
  title={Lost in the Middle Benchmark Suite v4},
  author={abhshkp},
  year={2026},
  url={https://huggingface.co/abhshkp/litm-benchmark-suite-v4}
}
```

---

## Acknowledgments

This suite extends the foundational work of Liu et al. (2023) and incorporates community feedback on scalable, modular benchmarking. Built with HuggingFace Transformers, bitsandbytes, and matplotlib.
