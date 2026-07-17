# Section 3 - LLM Quantization Benchmark

Compare a small open-weight LLM at **full precision (FP16)** vs **4-bit quantized (BitsAndBytes NF4)** and measure the trade-offs.

## Model

**Qwen2.5-1.5B-Instruct** - 1.5 billion parameters, small enough to run FP16 on a 6 GB GPU while still producing coherent, diverse outputs.

## What This Measures

| Metric | Method |
|---|---|
| **GPU VRAM** | `torch.cuda.max_memory_allocated()` |
| **CPU RAM** | `psutil.Process().memory_info().rss` |
| **Parameter memory** | Sum of tensor sizes from `model.parameters()` |
| **Throughput** | `generated_tokens / wall_time` (averaged over 5 prompts, 1 warm-up) |
| **Output quality** | Side-by-side text comparison on 5 fixed prompts |

## Setup

```bash
cd Quantization
pip install -r requirements.txt
```

> **Note:** Requires an NVIDIA GPU with CUDA support for the full FP16 + NF4 benchmark. Tested on RTX 3060 6GB (Laptop).

## Run

```bash
python benchmark.py
```

Optional arguments:
```bash
python benchmark.py --model "Qwen/Qwen2.5-1.5B-Instruct" --output-dir ./results
```

## Output

After running, you'll find:

- **`results/benchmark_results.json`** - Full metrics (memory, throughput, per-prompt outputs)
- **`results/comparison_table.md`** - Markdown table comparing FP16 vs NF4

## The 5 Benchmark Prompts

| # | Category | Dataset Source | Description / Short Prompt |
|---|---|---|---|
| 1 | MMLU | cais/mmlu – college_chemistry | Lanthanide elements multiple choice question. |
| 2 | ARC | allenai/ai2_arc – ARC-Challenge | Oak tree acorn and chipmunk population logic problem. |
| 3 | TruthfulQA | truthful_qa – misconceptions | What happens when you eat watermelon seeds. |
| 4 | HumanEval | openai/humaneval – HumanEval/0 | Python coding task: `has_close_elements`. |
| 5 | HellaSwag | Rowan/hellaswag – activitynet | Completing a scenario about washing a car. |

## Project Structure

```
Quantization/
├── benchmark.py        # Main entry point
├── config.py           # Prompts, model config, constants
├── model_loader.py     # FP16 and BNB-NF4 model loading
├── metrics.py          # Memory & throughput measurement
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── NOTES.md            # Production trade-off analysis
└── results/            # Auto-generated output
    ├── benchmark_results.json
    └── comparison_table.md
```

## See Also

- [NOTES.md](NOTES.md) - When to pick GPTQ/AWQ vs BitsAndBytes vs GGUF for production
