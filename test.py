# test it
from engine.loader import load_model

model, tokenizer = load_model("models/qwen2.5-3b-instruct")
print(model.device)
print(f"Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")