import llama_cpp
print(llama_cpp.__version__)

from llama_cpp import Llama

llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=True  # This will show GPU layer loading
)