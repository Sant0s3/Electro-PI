import gc
import time
import psutil
import torch


# Converts bytes to megabytes
def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


# Returns the current GPU memory allocated in megabytes
def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return bytes_to_mb(torch.cuda.memory_allocated())
    return 0.0


# Returns the peak GPU memory allocated in megabytes since reset
def get_peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return bytes_to_mb(torch.cuda.max_memory_allocated())
    return 0.0


# Resets peak memory statistics and empties CUDA cache
def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# Returns current process resident set size (RSS) in megabytes
def get_ram_usage_mb() -> float:
    return bytes_to_mb(psutil.Process().memory_info().rss)


# Estimates the size of the model parameters in memory
def estimate_model_size_mb(model) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_to_mb(total_bytes)


# Gets a snapshot of the memory consumption after loading a model
def measure_memory_after_load(model) -> dict:
    return {
        "gpu_allocated_mb": round(get_gpu_memory_mb(), 2),
        "gpu_peak_mb": round(get_peak_gpu_memory_mb(), 2),
        "ram_mb": round(get_ram_usage_mb(), 2),
        "param_size_mb": round(estimate_model_size_mb(model), 2),
    }


# Runs generation for a prompt and measures wall-clock time and throughput
def measure_generation(model, tokenizer, prompt_text: str, gen_kwargs: dict) -> dict:
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    num_input_tokens = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    new_tokens = output_ids[0, num_input_tokens:]
    num_output_tokens = len(new_tokens)
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    wall_time = end_time - start_time
    tokens_per_sec = num_output_tokens / wall_time if wall_time > 0 else 0.0

    return {
        "output_text": output_text,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "wall_time_s": round(wall_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


# Deletes model reference and runs garbage collection to free GPU/RAM memory
def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
