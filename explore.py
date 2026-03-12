from llama_cpp import Llama

print("Loading model...")
llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)
print("Model loaded.")

prompt = "[INST] Explain what a binary search tree is in two sentences. [/INST]"
tokens = llm.tokenize(prompt.encode())
print(f"Prompt tokenized to {len(tokens)} tokens")

llm.eval(tokens)

print("\nGenerating: ", end="", flush=True)

generated = []
for _ in range(200):
    token_id = llm.sample()
    
    if token_id == llm.token_eos():
        break
    
    token_text = llm.detokenize([token_id]).decode("utf-8", errors="ignore")
    print(token_text, end="", flush=True)
    
    generated.append(token_id)
    llm.eval([token_id])

print(f"\n\nGenerated {len(generated)} tokens.")