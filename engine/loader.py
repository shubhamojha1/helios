# engine/loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
      model_path,
      dtype=dtype,
      ).to(device)
    model.eval()
    return model, tokenizer