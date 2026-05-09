"""
Modal deployment of the upstream llama.cpp `llama-server` binary on a GPU.

It serves a base .gguf model with a LoRA adapter and a multimodal projector
(mmproj) that handles BOTH vision and audio for Gemma 4 / 4n. The frontend
(App.jsx) calls
    POST {VITE_LLAMA_URL}                   ->  /v1/chat/completions
exactly the same way it currently calls the local llama-server during dev.

Why upstream binary instead of llama-cpp-python:
  The Python wrapper's vision pipeline silently dropped image_url content
  for our Gemma build — the model never saw the image. The upstream
  `llama-server` binary handles --mmproj correctly for both vision and
  audio inputs, matching the local Windows build that works.

==============================================================================
One-time setup (run from the repo root, with Modal already authenticated):

    pip install modal
    modal setup

    # Drop your three artifacts into ./backend/, then upload them to a Modal
    # Volume (so they don't bloat the image and survive redeploys):
    modal volume create isl-models
    modal volume put isl-models backend/base_f16.gguf /base_f16.gguf
    modal volume put isl-models backend/lora.gguf     /lora.gguf
    modal volume put isl-models backend/mmproj.gguf   /mmproj.gguf

    # Deploy the server:
    modal deploy backend/app.py

Modal prints the public URL. Set it on Vercel as
    VITE_LLAMA_URL=https://<workspace>--isl-llama-server-serve.modal.run/v1/chat/completions
==============================================================================
"""

import modal

# ---------------------------------------------------------------------------
# Image: clone llama.cpp from source and build the `llama-server` binary
# with CUDA + multimodal (vision + audio via mmproj) support.
# ---------------------------------------------------------------------------
LLAMA_CPP_DIR = "/opt/llama.cpp"
LLAMA_SERVER_BIN = f"{LLAMA_CPP_DIR}/build/bin/llama-server"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "curl",
        "libcurl4-openssl-dev",
        "ccache",
    )
    .run_commands(
        # CUDA stubs so the linker has libcuda.so at build time. The real
        # driver is provided by Modal's host at runtime.
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        "echo '/usr/local/cuda/lib64/stubs' > /etc/ld.so.conf.d/cuda-stubs.conf",
        "ldconfig",
    )
    .run_commands(
        # Clone upstream llama.cpp. The `master` branch carries the latest
        # multimodal (mtmd) support including Gemma 3n audio. If you need a
        # reproducible build, replace with `&& git checkout <sha>`.
        f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR}",
        # Configure with CUDA + curl + multimodal server.
        # GGML_CUDA_F16 turns on fp16 compute (faster on T4/L4).
        # The server target pulls in the multimodal (mtmd) code automatically.
        f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build "
        "  -DGGML_CUDA=ON "
        "  -DGGML_CUDA_F16=ON "
        "  -DLLAMA_CURL=ON "
        "  -DCMAKE_BUILD_TYPE=Release "
        "  -DLLAMA_BUILD_SERVER=ON "
        "  -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs' "
        "  -DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs'",
        # Parallel build of just the server (and its deps).
        f"cmake --build {LLAMA_CPP_DIR}/build --config Release -j --target llama-server",
        # Sanity check — fail the image build early if the binary is missing.
        f"test -x {LLAMA_SERVER_BIN}",
    )
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "httpx",            # for the CORS reverse-proxy
        "pydantic",
    )
)

# ---------------------------------------------------------------------------
# Volume holds the .gguf artifacts so the container image stays small and the
# files persist across deploys.
#   /models/base_f16.gguf
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
    # f16 9 GB base + mmproj + KV cache spills on T4 (16 GB). L4 has 24 GB
    # and is the same Modal price tier — switch back to "T4" only if you
    # later swap to a Q8_0/Q5 quant of the base.
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
                                         /opt/llama.cpp/build/bin/llama-server
                                         (subprocess started in lifespan)
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
        # Start the upstream llama-server binary. Flags mirror the local
        # Windows invocation that works.
        #
        #   -m       base model
        #   --lora   LoRA adapter (merged at runtime)
        #   --mmproj multimodal projector (vision + audio for Gemma 4n)
        #   -ngl     GPU layers (-1 = all)
        #   -c       context window
        #   -fa      flash-attention (faster + lower VRAM)
        #   --jinja  use the chat template baked into the GGUF
        cmd = [
            LLAMA_SERVER_BIN,
            "-m",        BASE_GGUF,
            "--lora",    LORA_GGUF,
            "--mmproj",  MMPROJ_GGUF,
            "-ngl",      "999",
            "-c",        "4096",
            "-fa",       "on",
            "--jinja",
            "--host",    "127.0.0.1",
            "--port",    str(LLAMA_INTERNAL_PORT),
        ]
        print("[llama-server] launching:", " ".join(cmd), flush=True)
        llama_proc = subprocess.Popen(cmd)

        # One persistent async client — long timeout because inference is slow.
        app.state.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{LLAMA_INTERNAL_PORT}",
            timeout=httpx.Timeout(300.0),
        )

        # Wait for the server to start accepting connections before letting
        # any client requests hit the proxy. Otherwise the first few
        # requests get ConnectError 500s while the model is still loading.
        import asyncio
        for _ in range(180):  # up to ~3 minutes for cold load
            if llama_proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {llama_proc.returncode}"
                )
            try:
                r = await app.state.client.get("/health", timeout=2.0)
                if r.status_code == 200:
                    print("[llama-server] ready", flush=True)
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("llama-server did not become ready in time")

        try:
            yield
        finally:
            await app.state.client.aclose()
            llama_proc.terminate()
            try:
                llama_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llama_proc.kill()

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
