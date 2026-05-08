"""
Modal deployment of an OpenAI-compatible llama.cpp server on a T4 GPU.

It serves a base .gguf model with a LoRA adapter and a vision projector
(mmproj). The frontend (App.jsx) calls
    POST {VITE_LLAMA_URL}                   ->  /v1/chat/completions
exactly the same way it currently calls the local llama-server during dev.

==============================================================================
One-time setup (run from the repo root, with Modal already authenticated):

    pip install modal
    modal setup

    # Drop your three artifacts into ./backend/, then upload them to a Modal
    # Volume (so they don't bloat the image and survive redeploys):
    modal volume create isl-models
    modal volume put isl-models backend/base.gguf   /base.gguf
    modal volume put isl-models backend/lora.gguf   /lora.gguf
    modal volume put isl-models backend/mmproj.gguf /mmproj.gguf

    # Deploy the server:
    modal deploy backend/app.py

Modal prints the public URL. Set it on Vercel as
    VITE_LLAMA_URL=https://<workspace>--isl-llama-server-serve.modal.run/v1/chat/completions
==============================================================================
"""

import subprocess

import modal

# ---------------------------------------------------------------------------
# Image: CUDA + llama-cpp-python built with cuBLAS so the T4 is actually used.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .env(
        {
            # Build llama-cpp-python with CUDA support.
            "CMAKE_ARGS": "-DGGML_CUDA=on -DLLAMA_CURL=on",
            "FORCE_CMAKE": "1",
        }
    )
    .pip_install(
        "llama-cpp-python[server]==0.3.16",
        "fastapi",
        "uvicorn[standard]",
        "huggingface-hub",
        "pydantic",
    )
)

# ---------------------------------------------------------------------------
# Volume holds the .gguf artifacts so the container image stays small and the
# files persist across deploys.
#   /models/base.gguf
#   /models/lora.gguf
#   /models/mmproj.gguf
# ---------------------------------------------------------------------------
MODELS_DIR = "/models"
models_vol = modal.Volume.from_name("isl-models", create_if_missing=True)

BASE_GGUF = f"{MODELS_DIR}/base.gguf"
LORA_GGUF = f"{MODELS_DIR}/lora.gguf"
MMPROJ_GGUF = f"{MODELS_DIR}/mmproj.gguf"

PORT = 8000

app = modal.App("isl-llama-server")


@app.function(
    image=image,
    gpu="T4",
    volumes={MODELS_DIR: models_vol},
    timeout=60 * 60,
    scaledown_window=300,    # keep warm 5 min after last request
    max_containers=1,        # single-replica demo
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=180)
def serve():
    """
    Boot the OpenAI-compatible llama.cpp server. Frontend hits:
        POST  https://<modal-url>/v1/chat/completions
    """
    import os

    for p in (BASE_GGUF, LORA_GGUF, MMPROJ_GGUF):
        if not os.path.exists(p):
            raise RuntimeError(
                f"Missing artifact {p}. Upload it with "
                f"`modal volume put isl-models <local> {p[len(MODELS_DIR):]}`."
            )

    cmd = [
        "python", "-m", "llama_cpp.server",
        "--model", BASE_GGUF,
        "--lora_path", LORA_GGUF,
        "--clip_model_path", MMPROJ_GGUF,   # vision projector (mmproj)
        "--chat_format", "gemma",           # change to your model's template
        "--n_gpu_layers", "-1",             # offload everything to the T4
        "--n_ctx", "4096",
        "--host", "0.0.0.0",
        "--port", str(PORT),
    ]
    # Detach so @web_server can poll the port.
    subprocess.Popen(cmd)


# Optional: `modal run backend/app.py` runs a local sanity check against the
# deployed endpoint without committing the curl invocation to your codebase.
@app.local_entrypoint()
def main():
    print("Deploy with:  modal deploy backend/app.py")
    print("Then set VITE_LLAMA_URL on Vercel to the printed URL + /v1/chat/completions")
