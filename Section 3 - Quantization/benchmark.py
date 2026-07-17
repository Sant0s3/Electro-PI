import argparse
import json
import os
import sys
from datetime import datetime

# Check if required libraries are installed
try:
    import torch
    import psutil
    import transformers
    import bitsandbytes
    from tabulate import tabulate
except ImportError as e:
    print(f"\n[!] Dependency Error: {e}")
    print("Please make sure you have installed all required packages.")
    print("You can install them by running:")
    print("  pip install -r requirements.txt\n")
    sys.exit(1)

from config import GENERATION_KWARGS, MODEL_ID, PROMPTS, WARMUP_PROMPT
from metrics import (
    cleanup_model,
    measure_generation,
    measure_memory_after_load,
    reset_gpu_memory_stats,
)
from model_loader import load_bnb_nf4, load_fp16


# Runs the benchmark suite on a loaded model and returns the metrics
def run_benchmark(model, tokenizer, prompts, gen_kwargs, label):
    print(f"\nBenchmarking configuration: {label}")
    print("-" * 40)

    # Capture memory footprint
    mem = measure_memory_after_load(model)
    print("Memory Usage:")
    print(f"  GPU Allocated : {mem['gpu_allocated_mb']:.1f} MB")
    print(f"  GPU Peak      : {mem['gpu_peak_mb']:.1f} MB")
    print(f"  RAM (Process) : {mem['ram_mb']:.1f} MB")
    print(f"  Parameters    : {mem['param_size_mb']:.1f} MB")

    # Run warm-up
    print("\nRunning warm-up generation...")
    _ = measure_generation(model, tokenizer, WARMUP_PROMPT, gen_kwargs)
    print("Warm-up complete.")

    # Evaluate all test prompts
    results = []
    print("\nRunning benchmark prompts:")
    for idx, prompt in enumerate(prompts, 1):
        print(f"  [{idx}/{len(prompts)}] Category: {prompt['category']}...")
        res = measure_generation(model, tokenizer, prompt["text"], gen_kwargs)
        res.update({
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "prompt_text": prompt["text"]
        })
        results.append(res)
        print(f"        -> {res['num_output_tokens']} tokens | {res['tokens_per_sec']:.1f} tok/s | {res['wall_time_s']:.2f}s")

    avg_speed = sum(r["tokens_per_sec"] for r in results) / len(prompts)
    return {
        "label": label,
        "memory": mem,
        "prompt_results": results,
        "avg_tokens_per_sec": round(avg_speed, 2),
    }


# Saves both the raw JSON metrics and the human-readable Markdown report
def save_reports(fp16_res, nf4_res, output_dir, model_id):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save raw results as JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    raw_data = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "fp16": fp16_res,
        "nf4_4bit": nf4_res,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Raw results saved to: {json_path}")

    # Helper calculations for the report
    fp_mem, nf_mem = fp16_res["memory"], nf4_res["memory"]
    
    def calc_saving(b, q):
        return ((b - q) / b * 100) if b > 0 else 0.0

    gpu_saving = calc_saving(fp_mem["gpu_allocated_mb"], nf_mem["gpu_allocated_mb"])
    param_saving = calc_saving(fp_mem["param_size_mb"], nf_mem["param_size_mb"])
    speed_change = ((nf4_res["avg_tokens_per_sec"] - fp16_res["avg_tokens_per_sec"]) 
                    / fp16_res["avg_tokens_per_sec"] * 100) if fp16_res["avg_tokens_per_sec"] > 0 else 0.0

    # 2. Generate Markdown comparison report
    md_path = os.path.join(output_dir, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Quantization Benchmark Results\n\n")
        f.write(f"**Model:** {model_id}  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Hardware:** NVIDIA RTX 3060 6GB (Laptop)\n\n")

        # Trade-off Table
        f.write("## Trade-off Summary: Precision vs. Size vs. Speed vs. Quality\n\n")
        tradeoff_headers = ["Dimension", "FP16 (Full Precision)", "NF4 4-bit (Quantized)"]
        tradeoff_data = [
            ["Precision", "16-bit floating point", "4-bit NormalFloat (NF4)"],
            ["Model Size (VRAM)", f"{fp_mem['gpu_allocated_mb']:.0f} MB", f"{nf_mem['gpu_allocated_mb']:.0f} MB"],
            ["Param Memory", f"{fp_mem['param_size_mb']:.0f} MB", f"{nf_mem['param_size_mb']:.0f} MB"],
            ["Speed (tok/s)", f"{fp16_res['avg_tokens_per_sec']:.1f}", f"{nf4_res['avg_tokens_per_sec']:.1f}"],
            ["Quality", "Baseline (full fidelity)", "Near-baseline (see output comparison below)"],
        ]
        f.write(tabulate(tradeoff_data, headers=tradeoff_headers, tablefmt="github") + "\n\n")

        # Detailed Stats
        f.write("## Detailed Performance & Memory Comparison\n\n")
        f.write("| Metric | FP16 (baseline) | NF4 4-bit | Delta |\n")
        f.write("|--------|-----------------|-----------|-------|\n")
        f.write(f"| GPU VRAM (MB) | {fp_mem['gpu_allocated_mb']:.1f} | {nf_mem['gpu_allocated_mb']:.1f} | {gpu_saving:+.1f}% saved |\n")
        f.write(f"| Param Memory (MB) | {fp_mem['param_size_mb']:.1f} | {nf_mem['param_size_mb']:.1f} | {param_saving:+.1f}% saved |\n")
        f.write(f"| RAM / RSS (MB) | {fp_mem['ram_mb']:.1f} | {nf_mem['ram_mb']:.1f} | - |\n")
        f.write(f"| Avg Tokens/sec | {fp16_res['avg_tokens_per_sec']:.1f} | {nf4_res['avg_tokens_per_sec']:.1f} | {speed_change:+.1f}% |\n\n")

        # Per-Prompt Speed Table
        f.write("## Per-Prompt Throughput (tokens/sec)\n\n")
        prompt_headers = ["Prompt", "FP16 tok/s", "FP16 tokens", "FP16 time (s)", "NF4 tok/s", "NF4 tokens", "NF4 time (s)"]
        prompt_rows = []
        for fp_p, nf_p in zip(fp16_res["prompt_results"], nf4_res["prompt_results"]):
            prompt_rows.append([
                fp_p["category"],
                f"{fp_p['tokens_per_sec']:.1f}",
                fp_p["num_output_tokens"],
                f"{fp_p['wall_time_s']:.2f}",
                f"{nf_p['tokens_per_sec']:.1f}",
                nf_p["num_output_tokens"],
                f"{nf_p['wall_time_s']:.2f}"
            ])
        f.write(tabulate(prompt_rows, headers=prompt_headers, tablefmt="github") + "\n\n")

        # Qualitative comparison
        f.write("## Qualitative Output Comparison\n\n")
        f.write("*Same 5 prompts run through both FP16 and NF4 versions for side-by-side quality assessment.*\n\n")
        for fp_p, nf_p in zip(fp16_res["prompt_results"], nf4_res["prompt_results"]):
            f.write(f"### {fp_p['category']}\n\n")
            f.write(f"**Prompt:** {fp_p['prompt_text']}\n\n")
            f.write(f"**FP16 Output:**\n```\n{fp_p['output_text']}\n```\n\n")
            f.write(f"**NF4 4-bit Output:**\n```\n{nf_p['output_text']}\n```\n\n")
            f.write("---\n\n")

    print(f"[+] Comparison markdown report saved to: {md_path}")


# Parses arguments, runs the benchmarks for both models, prints results, and saves files
def main():
    parser = argparse.ArgumentParser(description="Quantization Benchmark: FP16 vs NF4 4-bit")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save benchmark reports")
    args = parser.parse_args()

    print("=" * 50)
    print("  STARTING LLM QUANTIZATION BENCHMARK")
    print(f"  Model: {args.model}")
    print(f"  Test Prompts: {len(PROMPTS)}")
    print("=" * 50)

    # --- Phase 1: Full Precision FP16 ---
    reset_gpu_memory_stats()
    model, tokenizer = load_fp16(args.model)
    fp16_results = run_benchmark(model, tokenizer, PROMPTS, GENERATION_KWARGS, "FP16")
    
    cleanup_model(model)
    del tokenizer
    print("\nFP16 model unloaded and VRAM cleared.")

    # --- Phase 2: BitsAndBytes NF4 4-bit ---
    reset_gpu_memory_stats()
    model, tokenizer = load_bnb_nf4(args.model)
    nf4_results = run_benchmark(model, tokenizer, PROMPTS, GENERATION_KWARGS, "NF4 4-bit")
    
    cleanup_model(model)
    del tokenizer
    print("\nNF4 model unloaded and VRAM cleared.")

    # --- Print Terminal Summary ---
    print("\n" + "=" * 50)
    print("  BENCHMARK SUMMARY")
    print("=" * 50)

    # Detailed Comparison Terminal Table
    gpu_diff = ((nf4_results["memory"]["gpu_allocated_mb"] - fp16_results["memory"]["gpu_allocated_mb"])
                / fp16_results["memory"]["gpu_allocated_mb"] * 100) if fp16_results["memory"]["gpu_allocated_mb"] > 0 else 0
    param_diff = ((nf4_results["memory"]["param_size_mb"] - fp16_results["memory"]["param_size_mb"])
                  / fp16_results["memory"]["param_size_mb"] * 100) if fp16_results["memory"]["param_size_mb"] > 0 else 0
    speed_diff = ((nf4_results["avg_tokens_per_sec"] - fp16_results["avg_tokens_per_sec"])
                  / fp16_results["avg_tokens_per_sec"] * 100) if fp16_results["avg_tokens_per_sec"] > 0 else 0

    summary_headers = ["Metric", "FP16 (baseline)", "NF4 4-bit", "Delta"]
    summary_rows = [
        ["GPU VRAM (MB)", f"{fp16_results['memory']['gpu_allocated_mb']:.1f}", f"{nf4_results['memory']['gpu_allocated_mb']:.1f}", f"{gpu_diff:+.1f}%"],
        ["Param Memory (MB)", f"{fp16_results['memory']['param_size_mb']:.1f}", f"{nf4_results['memory']['param_size_mb']:.1f}", f"{param_diff:+.1f}%"],
        ["RAM / RSS (MB)", f"{fp16_results['memory']['ram_mb']:.1f}", f"{nf4_results['memory']['ram_mb']:.1f}", "-"],
        ["Avg Tokens/sec", f"{fp16_results['avg_tokens_per_sec']:.1f}", f"{nf4_results['avg_tokens_per_sec']:.1f}", f"{speed_diff:+.1f}%"]
    ]
    print("\n" + tabulate(summary_rows, headers=summary_headers, tablefmt="grid"))

    # Quick preview of outputs
    print("\nQualitative Preview (Truncated):")
    print("-" * 50)
    for fp_p, nf_p in zip(fp16_results["prompt_results"], nf4_results["prompt_results"]):
        print(f"\n[{fp_p['category']}]")
        print(f"  Prompt: {fp_p['prompt_text'][:70]}...")
        print(f"  FP16  : {fp_p['output_text'][:100].strip()}...")
        print(f"  NF4   : {nf_p['output_text'][:100].strip()}...")

    # Save to disk
    save_reports(fp16_results, nf4_results, args.output_dir, args.model)
    print("\n" + "=" * 50)
    print("  BENCHMARK COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
