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

import modal

# ---------------------------------------------------------------------------
# Image: CUDA + llama-cpp-python built with cuBLAS so the T4 is actually used.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        "echo '/usr/local/cuda/lib64/stubs' > /etc/ld.so.conf.d/cuda-stubs.conf",
        "ldconfig"
    )
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",

            #"LDFLAGS": "-L/usr/local/cuda/lib64/stubs",
            #"LIBRARY_PATH": "/usr/local/cuda/lib64/stubs",
            #"LD_LIBRARY_PATH": "/usr/local/cuda/lib64/stubs",

            # Build llama-cpp-python with CUDA support.
            "CMAKE_ARGS": "-DGGML_CUDA=on -DLLAMA_CURL=on",
            "FORCE_CMAKE": "1",
        }
    )
    .pip_install(
        # No version pin — gemma4 arch support was added after 0.3.16.
        # Latest build will always have the newest architecture support.
        "llama-cpp-python[server]",
        "fastapi",
        "uvicorn[standard]",
        "httpx",            # for the CORS reverse-proxy
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

LLAMA_INTERNAL_PORT = 8001   # not exposed, used only inside the container

app = modal.App("isl-llama-server")


@app.function(
    image=image,
    gpu="T4",
    volumes={MODELS_DIR: models_vol},
    timeout=60 * 60,
    scaledown_window=300,   # keep warm 5 min after last request
    max_containers=1,       # single-replica demo
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def serve():
    """
    Architecture:

        Browser  →  Modal HTTPS edge  →  this FastAPI proxy (Modal owns the
                                          web server, no manual uvicorn)
                                            │  CORS middleware
                                            ▼
                                         127.0.0.1:8001
                                            │
                                         python -m llama_cpp.server
                                         (subprocess started in lifespan)

    We use @modal.asgi_app rather than @modal.web_server so Modal's runtime
    serves the ASGI app directly. The previous @modal.web_server +
    blocking uvicorn.run() approach silently failed to bind port 8000,
    which is why the browser got responses with no CORS headers.
    """
    import os
    import subprocess
    from contextlib import asynccontextmanager

    import httpx
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from starlette.background import BackgroundTask

    for p in (BASE_GGUF, LORA_GGUF, MMPROJ_GGUF):
        if not os.path.exists(p):
            raise RuntimeError(
                f"Missing artifact {p}. Upload it with:\n"
                f"  modal volume put isl-models <local-path> {p[len(MODELS_DIR):]}"
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start llama.cpp as a subprocess on the internal port.
        # --chat_format is intentionally omitted: the GGUF has
        # `tokenizer.chat_template` baked in, so llama.cpp uses it directly.
        llama_proc = subprocess.Popen([
            "python", "-m", "llama_cpp.server",
            "--model",           BASE_GGUF,
            "--lora_path",       LORA_GGUF,
            "--clip_model_path", MMPROJ_GGUF,
            "--n_gpu_layers",    "-1",       # full GPU offload on T4
            "--n_ctx",           "4096",
            "--host",            "127.0.0.1",
            "--port",            str(LLAMA_INTERNAL_PORT),
        ])

        # One persistent async client — long timeout because inference is slow.
        app.state.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{LLAMA_INTERNAL_PORT}",
            timeout=httpx.Timeout(300.0),
        )

        try:
            yield
        finally:
            await app.state.client.aclose()
            llama_proc.terminate()

    proxy = FastAPI(lifespan=lifespan)

    # CORS — the entire reason for this proxy layer.
    proxy.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # tighten to your Vercel URL in production
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @proxy.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    )
    async def reverse_proxy(request: Request, path: str):
        client: httpx.AsyncClient = request.app.state.client
        url = httpx.URL(path=f"/{path}", query=request.url.query.encode("utf-8"))
        upstream_req = client.build_request(
            method=request.method,
            url=url,
            # forward everything except Host (must match 127.0.0.1)
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            content=await request.body(),
        )
        upstream_resp = await client.send(upstream_req, stream=True)
        # Strip content-length — we stream so the length can differ.
        fwd_headers = {
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() != "content-length"
        }
        return StreamingResponse(
            upstream_resp.aiter_bytes(),
            status_code=upstream_resp.status_code,
            headers=fwd_headers,
            background=BackgroundTask(upstream_resp.aclose),
        )

    return proxy


# Optional: `modal run backend/app.py` runs a local sanity check against the
# deployed endpoint without committing the curl invocation to your codebase.
@app.local_entrypoint()
def main():
    print("Deploy with:  modal deploy backend/app.py")
    print("Then set VITE_LLAMA_URL on Vercel to the printed URL + /v1/chat/completions")
