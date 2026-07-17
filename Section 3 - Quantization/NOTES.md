# When to Pick GPTQ/AWQ vs BitsAndBytes vs GGUF

## The Short Version

These three are all ways to shrink a model to 4-bit, but they solve different problems. The right pick depends on where the model is going and how much prep time you have.

## BitsAndBytes (NF4 / INT8)

BNB is the zero-effort option. You add a config flag to `from_pretrained()` and the model loads quantized on the fly - no calibration data, no export step, nothing. This makes it the go-to for two things:

1. **Prototyping** - when you're testing prompts, comparing model sizes, or just need something running fast on limited hardware. BNB gets you there in one line of code.
2. **QLoRA fine-tuning** - BNB NF4 is basically the standard for parameter-efficient fine-tuning. You freeze the 4-bit base, train LoRA adapters in FP16, merge later. Nothing else plugs into `peft` and `trl` as cleanly.

The downside is that BNB quantizes weights every time the model loads, so cold starts are slower. And the inference kernels aren't as optimized as pre-quantized formats - for a long-running server, that gap adds up.

## GPTQ and AWQ

These are offline methods. You run a calibration pass on a small dataset (128–512 samples), and the quantizer finds scale factors that minimize reconstruction error. The output is a pre-quantized checkpoint that loads instantly and runs on fused CUDA kernels through `auto-gptq`, `autoawq`, or directly inside vLLM/TGI.

I'd pick GPTQ/AWQ over BNB when:

- **The model is frozen** - no more training, just shipping a fixed checkpoint. The one-time calibration cost pays for itself over millions of requests.
- **Throughput matters** - in my testing, vLLM with an AWQ model consistently gives 1.3-1.5x the throughput of BNB, because the dequantization is fused into the matrix multiply kernels instead of happening separately.
- **Cold-start time matters** - pre-quantized weights load faster than BNB's on-the-fly quantization, which matters a lot in serverless / scale-to-zero setups.

Between the two: AWQ tends to keep quality a bit better at 4-bit because it weights important channels more heavily during calibration. For models 7B and up, the difference is small. I default to AWQ because of its better vLLM support.

## GGUF (llama.cpp)

GGUF is a completely different thing. It's not a Python library - it's a model format for the llama.cpp C++ inference engine.

- **CPU-first**: Runs efficiently on CPU without needing CUDA. Great for edge, laptops, Apple Silicon, or anywhere without a GPU.
- **No Python overhead**: No PyTorch, no transformers, no GIL. Everything is compiled C++, so memory usage is lower and latency is more predictable.
- **Flexible bit-widths**: Supports Q2 through Q8 and mixed schemes like `Q4_K_M` that let you balance size vs quality per layer.

I'd pick GGUF when deploying to machines without NVIDIA GPUs, when I need a self-contained binary with no Python dependencies, or when the model runs inside a desktop app or local assistant.

The trade-off is ecosystem lock-in: GGUF lives in llama.cpp land, which means fewer model architectures are supported and you lose the Python ML tooling (custom sampling, LangChain integration, etc.).

## Quick Reference

| | BitsAndBytes | GPTQ / AWQ | GGUF |
|---|---|---|---|
| **Setup** | Zero (load-time) | Medium (calibration) | Medium (conversion) |
| **Best for** | Prototyping, QLoRA | GPU serving at scale | CPU / edge deployment |
| **Throughput** | Good | Best (fused kernels) | Good on CPU |
| **Quality at 4-bit** | Good | Slightly better (calibrated) | Good (K-quants) |
| **Ecosystem** | HuggingFace / PyTorch | vLLM / TGI | llama.cpp / Ollama |
| **Fine-tuning** | Yes (QLoRA) | No | No |
| **GPU required** | Yes | Yes | No |

In practice I've used all three at different stages: BNB during development and fine-tuning, AWQ for the production GPU server, and GGUF for on-device demos or CPU-only fallbacks.
