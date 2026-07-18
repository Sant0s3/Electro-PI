import logging
import os
import sys
import threading
from typing import Generator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("qwen-service")

# Initialize configuration
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Qwen Inference Service",
    description="FastAPI service for Qwen2.5-1.5B-Instruct model",
)


logger.info("Loading tokenizer and model: %s on %s...", MODEL_ID, device)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        # Load in FP16 to balance VRAM usage and loading stability on Windows
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to(device)

    model.eval()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.exception("Failed to load model or tokenizer: %s", e)
    sys.exit(1)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@app.get("/health")
def health():
    return {"status": "healthy", "device": device}


@app.post("/generate")
def generate_stream(request: GenerateRequest):
    """Streams the generated tokens step-by-step."""
    try:
        messages = [{"role": "user", "content": request.prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            top_p=request.top_p,
        )

        # Run model.generate in a separate thread so it doesn't block the FastAPI event loop
        def run_generation():
            try:
                model.generate(**generation_kwargs)
            except Exception as e:
                logger.error("Generation failed inside worker thread: %s", e, exc_info=True)
            finally:
                # Guarantee that streamer is closed so the generator does not hang forever
                streamer.end()

        thread = threading.Thread(target=run_generation)
        thread.start()

        def token_generator() -> Generator[str, None, None]:
            for token in streamer:
                yield token

        return StreamingResponse(token_generator(), media_type="text/plain")

    except Exception as e:
        logger.exception("Error initiating streaming response: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_json")
def generate_json(request: GenerateRequest):
    """Generates response and returns it as a single JSON object."""
    try:
        messages = [{"role": "user", "content": request.prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                top_p=request.top_p,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "prompt": request.prompt,
            "response": output_text,
            "tokens_generated": len(new_tokens),
        }
    except Exception as e:
        logger.exception("Error generating JSON response: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=HOST, port=PORT)
