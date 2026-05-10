# Helios

Local LLM inference serving engine with OpenAI-compatible API routes.

## Requirements

- Python 3.11+
- A GGUF model file
- `llama-cpp-python` support for your OS and hardware

## Setup

Clone the repo and enter the project directory:

```powershell
git clone <your-repo-url>
cd helios
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies and the local CLI:

```powershell
pip install -r requirements.txt
pip install -e .
```

The editable install makes the `helios` command available.

## Download A Model

The default server command expects this model path:

```text
models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Download Mistral-7B-Instruct GGUF with:

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF', filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf', local_dir='./models')"
```

You can also use another GGUF model and pass its path with `--model`.

## Run The Server

Start Helios:

```powershell
helios serve
```

By default, the server runs at:

```text
http://127.0.0.1:8080
```

Useful options:

```powershell
helios serve --host 0.0.0.0 --port 8080
helios serve --model .\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf
helios serve --n-ctx 2048 --n-gpu-layers -1 --max-batch-size 4
```

The old direct Python entry point still works:

```powershell
python main.py
```

## API Examples

List models:

```powershell
curl http://127.0.0.1:8080/v1/models
```

Chat completions:

```powershell
curl http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"helios\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50,\"stream\":false,\"priority\":\"normal\"}"
```

Legacy text completions:

```powershell
curl http://127.0.0.1:8080/v1/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"helios\",\"prompt\":\"Hello\",\"max_tokens\":50,\"stream\":false,\"priority\":\"normal\"}"
```

Health check:

```powershell
curl http://127.0.0.1:8080/health
```

## Request Notes

Both completion routes accept:

- `model`: use `helios` as an alias for the currently loaded model, or use the loaded GGUF filename.
- `max_tokens`: maximum number of tokens to generate.
- `priority`: one of `high`, `normal`, or `low`.
- `stream`: supported on `/v1/chat/completions`; `/v1/completions` currently supports non-streaming requests only.

