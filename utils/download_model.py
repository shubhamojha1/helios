# download_model.py
from huggingface_hub import snapshot_download, login

# login()

# snapshot_download(
#     repo_id="meta-llama/Llama-3.2-3B-Instruct",
#     local_dir="models/llama-3.2-3b-instruct",
#     ignore_patterns=["*.bin", "original/*"]
# )
snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    local_dir="models/qwen2.5-3b-instruct",
    ignore_patterns=["*.bin", "*.gguf"]
)