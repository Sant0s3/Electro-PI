# Electro PI - AI Engineer Technical Assessment

## Structure
```
Electro-PI/
├── Section 1 - Voice Agent/    # Section 1: LiveKit voice agent + tool calling
│   ├── agent.py                # Main agent with STT → LLM → TTS pipeline
│   ├── tools.py                # @function_tool definitions
│   ├── transcript.md           # Full conversation log with tool call evidence
│   ├── NOTES.md                # Write-up on barge-in and adding tools safely
│   └── Section 1.2 - Bonus/    # Section 1.2 (bonus)
│       ├── agent.py            # Same agent with swapped STT provider
│       └── custom_stt.py       # FasterWhisper STT plugin (replaces Deepgram)
│
├── Section 2 - RAG System/     # Section 2: LangChain RAG pipeline
│   ├── agent.py                # LangGraph-based retrieval chain over a PDF
│   ├── pdf.pdf                 # Source document (ensemble learning paper)
│   └── README.md               # Write-up on chunking, retrieval improvements
│
├── Section 3 - Quantization/   # Section 3: FP16 vs NF4 quantization benchmark
│   ├── benchmark.py            # Main benchmark runner
│   ├── config.py               # Prompts and model configuration
│   ├── model_loader.py         # FP16 and NF4 model loading
│   ├── metrics.py              # Memory and throughput measurement
│   ├── NOTES.md                # Write-up on GPTQ/AWQ vs BNB vs GGUF
│   └── results/
│       ├── benchmark_results.json
│       └── comparison_table.md
│
├── Section 4 - Model Service/  # Section 4: FastAPI inference server
│   ├── app.py                  # Streaming + JSON generation endpoints
│   ├── load_test.py            # 10-concurrent-request load tester
│   ├── Dockerfile              # Containerization
│   ├── NOTES.md                # Write-up on scaling to 50+ users
│   └── requirements.txt
│
└── README.md                   # This file
```

## Quick Start

**Requirements**: Python 3.11, NVIDIA GPU with CUDA (tested on RTX 3060 6GB Laptop)

### Section 1 - LiveKit Voice Agent
```bash
cd "Section 1 - Voice Agent"
pip install -r requirements.txt
# Copy .env.example to .env and fill in your API keys (Deepgram, Groq, LiveKit)
python agent.py console
```
See `transcript.md` for the recorded conversation with tool calls.

### Section 2 - LangChain RAG
```bash
cd "Section 2 - RAG System"
pip install -r requirements.txt
python agent.py
```
Runs example questions over a PDF about ensemble learning methods.

### Section 3 - Quantization Benchmark
```bash
cd "Section 3 - Quantization"
pip install -r requirements.txt
python benchmark.py
```
Compares Qwen2.5-1.5B at FP16 vs NF4 4-bit. Results saved to `results/`.

### Section 4 - Model Deployment
```bash
cd "Section 4 - Model Service"
pip install -r requirements.txt
python app.py
# In another terminal:
python load_test.py
```
Or with Docker:
```bash
docker build -t qwen-service .
docker run --gpus all -p 8000:8000 qwen-service
```

## Hardware Used
- **GPU**: NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)
- **OS**: Windows 10
- **Python**: 3.11

## Limitations & Trade-offs
- **Section 1**: Used Deepgram + Groq cloud APIs (requires API keys). The bonus section swaps Deepgram STT with a local FasterWhisper model running on CPU accuracy wasn't that good.
- **Section 2**: The RAG pipeline retrieves relevant chunks from the PDF and generates answers using Groq's Llama 3.3 70B model. Requires a `GROQ_API_KEY` in `.env`.
- **Section 3**: BitsAndBytes NF4 quantization crashes silently on native Windows during weight loading. We documented this and measured both modes successfully by running them in separate processes.
- **Section 4**: Used FastAPI instead of vLLM/TGI because vLLM doesn't support native Windows and pre-allocates too much VRAM for a small model on a consumer GPU. Justification is in `Section 4 - Model Service/NOTES.md`.
