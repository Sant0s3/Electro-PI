import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Downloads and sets up the tokenizer for the model
def _get_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Loads the full precision (FP16) model on the GPU
def load_fp16(model_id: str):
    print(f"\nLoading model in FP16: {model_id}")
    tokenizer = _get_tokenizer(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  -> Loaded on: {model.device}")
    print(f"  -> dtype: {next(model.parameters()).dtype}")
    return model, tokenizer


# Loads the model in 4-bit mode using BitsAndBytes NF4
def load_bnb_nf4(model_id: str):
    print(f"\nLoading model in 4-bit NF4: {model_id}")
    tokenizer = _get_tokenizer(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("  -> Loaded in 4-bit NF4 mode")
    print("  -> Compute dtype: float16")
    return model, tokenizer
