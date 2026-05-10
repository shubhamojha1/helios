import argparse
import asyncio

from main import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="helios")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Start the Helios API server")
    serve_parser.add_argument(
        "--model",
        default="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        help="Path to the GGUF model file",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    serve_parser.add_argument("--n-ctx", type=int, default=2048, help="Model context size")
    serve_parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers to offload")
    serve_parser.add_argument("--max-batch-size", type=int, default=4, help="Scheduler batch size")

    args = parser.parse_args()

    if args.command == "serve":
        asyncio.run(
            serve(
                model_path=args.model,
                host=args.host,
                port=args.port,
                n_gpu_layers=args.n_gpu_layers,
                n_ctx=args.n_ctx,
                max_batch_size=args.max_batch_size,
            )
        )


if __name__ == "__main__":
    main()
