# Qwen Inference Service Notes

Quick wrapper for running `Qwen2.5-1.5B-Instruct` locally using FastAPI.

## Stack Choice: Why FastAPI over vLLM/TGI?
We went with FastAPI + Hugging Face Transformers instead of enterprise engines like vLLM or TGI for a few reasons:
- **Windows Support**: vLLM/TGI are tailored for enterprise Linux systems and high-end GPUs. They're a pain to run on native Windows. FastAPI runs anywhere without issues.
- **Resource Footprint**: vLLM pre-allocates up to 90% of GPU memory for its KV cache by default. For a small 1.5B model, that's way too much VRAM overhead on a consumer card (like an RTX 3060 6GB). Running FP16 directly in FastAPI only uses ~3GB VRAM, leaving room for other things.
- **Simplicity**: Pure Python allows us to write custom streaming logic (`TextIteratorStreamer`) and health endpoints in just a few lines.

## Load Test Results (10 concurrent requests)
Ran `load_test.py` with 10 parallel requests sending: *"Write a short paragraph explaining the theory of relativity."*

- **Success Rate**: 100% (10/10)
- **Avg TTFT**: 1.690s
- **Avg Latency**: 44.803s
- **Throughput**: ~23.0 tokens/sec total (~2.3 tokens/sec per user)

### Raw Request Logs
```
Request ID  TTFT (s)    Total Latency (s)   Tokens    
-------------------------------------------------------
1           1.712       44.893              101       
2           1.691       45.025              107       
3           1.682       44.891              99        
4           1.707       45.120              109       
5           1.717       45.241              93        
6           1.675       44.508              108       
7           1.694       44.591              99        
8           1.675       44.781              112       
9           1.660       44.392              102       
10          1.687       44.586              109       
```

## 3. Production Scaling for 50+ Concurrent Users

To scale our local developer FastAPI service to serve 50+ concurrent users in a production environment, we need to introduce four core optimization layers: **Queueing**, **Dynamic Batching**, **Caching**, and **Autoscaling**.

### 1. Request Queueing
Instead of sending incoming requests directly to the GPU (which blocks execution and can crash the server under load), we drop them into a queue (like an `asyncio.Queue` locally, or `Celery with Redis` in production). The queue acts as a buffer, protecting the model from sudden traffic spikes and holding requests safely until the engine can process them.

### 2. Dynamic Batching
A single GPU forward pass on 1 prompt takes nearly the same time as running 8 or 16 prompts. By pulling requests from the queue in batches and padding them to a uniform length, we can perform **dynamic batching**. This allows the GPU to compute multiple generations in parallel, multiplying the server's throughput without adding hardware.

### 3. Caching
Many production queries are repetitive. By setting up an in-memory or Redis-backed cache for prompt-response pairs, we can check incoming queries first. If a match exists, the response is returned instantly (<10ms) with zero GPU overhead, freeing up precious compute resources.

### 4. Horizontal Autoscaling
A single GPU has a physical limit on throughput. To support high concurrent loads, we run multiple containers of the stateless service behind a load balancer (like Nginx or AWS ALB) and use an event-driven autoscale framework (like KEDA on Kubernetes) to automatically spin up additional GPU instances when the request queue starts building up.
