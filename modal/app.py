import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

APP_NAME = os.getenv("APP_NAME", "deepseek-v4-flash")
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V4-Flash")
MODEL_REVISION = os.getenv("MODEL_REVISION", "")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "deepseek-v4-flash")
GPU_TYPE = os.getenv("GPU_TYPE", "B200")
GPU_COUNT = int(os.getenv("GPU_COUNT", "4"))
TP_SIZE = int(os.getenv("TP_SIZE", str(GPU_COUNT)))
PORT = int(os.getenv("PORT", "8000"))
SGLANG_PORT = int(os.getenv("SGLANG_PORT", "30000"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "131072"))
MEM_FRACTION_STATIC = float(os.getenv("MEM_FRACTION_STATIC", "0.90"))
CHUNKED_PREFILL_SIZE = int(os.getenv("CHUNKED_PREFILL_SIZE", "4096"))
MAX_RUNNING_REQUESTS = int(os.getenv("MAX_RUNNING_REQUESTS", "64"))
STARTUP_TIMEOUT_SECONDS = int(os.getenv("STARTUP_TIMEOUT_SECONDS", "7200"))
ROUTING_REGION = os.getenv("ROUTING_REGION", "us-east")
COMPUTE_REGION = os.getenv("COMPUTE_REGION", ROUTING_REGION)
MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW_SECONDS = int(os.getenv("SCALEDOWN_WINDOW_SECONDS", "1200"))
TARGET_CONCURRENCY = int(os.getenv("TARGET_CONCURRENCY", "1"))
SGLANG_IMAGE = os.getenv("SGLANG_IMAGE", "lmsysorg/sglang:deepseek-v4-blackwell")
API_KEY = os.getenv("DEEPSEEK_MODAL_API_KEY", "")
HF_CACHE = modal.Volume.from_name(
    os.getenv("HF_VOLUME_NAME", "deepseek-v4-flash-hf-cache"), create_if_missing=True
)

app = modal.App(APP_NAME)

image = modal.Image.from_registry(SGLANG_IMAGE).entrypoint([])


def build_sglang_command():
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        MODEL_ID,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(SGLANG_PORT),
        "--trust-remote-code",
        "--tp",
        str(TP_SIZE),
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--attention-backend",
        "dsv4",
        "--mem-fraction-static",
        str(MEM_FRACTION_STATIC),
        "--chunked-prefill-size",
        str(CHUNKED_PREFILL_SIZE),
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        "--reasoning-parser",
        "deepseek-v4",
        "--tool-call-parser",
        "deepseekv4",
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]
    return cmd


class DeepSeekServer:
    @modal.enter()
    def start(self):
        subprocess.check_call(
            [
                "pip",
                "install",
                "-q",
                "fastapi>=0.115,<1",
                "uvicorn[standard]>=0.34,<1",
                "httpx>=0.28,<1",
            ]
        )
        for key in list(sys.modules):
            if "typing_extensions" in key:
                del sys.modules[key]

        from fastapi import FastAPI, Header, HTTPException, Request
        from fastapi.responses import Response

        import httpx
        import uvicorn

        self.process = None
        self.log_file = None
        self.gateway_thread = None
        self.sglang_client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{SGLANG_PORT}",
            timeout=httpx.Timeout(connect=5, read=300, write=300, pool=5),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
        )
        gateway = FastAPI(title="DeepSeek V4 Flash Gateway")

        def authorized(a):
            return (not API_KEY) or a == f"Bearer {API_KEY}"

        def auth(a):
            if not authorized(a):
                raise HTTPException(status_code=401, detail="Unauthorized")

        @gateway.get("/health")
        async def health(authorization: str | None = Header(default=None)):
            auth(authorization)
            try:
                r = await self.sglang_client.get("/health")
                if r.status_code >= 500:
                    raise HTTPException(status_code=503, detail="SGLang unhealthy")
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=503, detail="SGLang unavailable")
            return {"status": "ok", "model": SERVED_MODEL_NAME}

        @gateway.get("/v1/models")
        async def models(authorization: str | None = Header(default=None)):
            auth(authorization)
            try:
                r = await self.sglang_client.get("/v1/models")
            except Exception:
                raise HTTPException(status_code=503, detail="SGLang unavailable")
            return Response(
                r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )

        @gateway.api_route(
            "/v1/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def proxy(
            path: str,
            request: Request,
            authorization: str | None = Header(default=None),
        ):
            auth(authorization)
            body = await request.body()
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)
            try:
                r = await self.sglang_client.request(
                    request.method,
                    f"/v1/{path}",
                    content=body,
                    headers=headers,
                    params=request.query_params,
                )
            except Exception:
                raise HTTPException(status_code=503, detail="SGLang unavailable")
            excluded = {"content-length", "transfer-encoding", "connection"}
            rh = {
                k: v
                for k, v in r.headers.items()
                if k.lower() not in excluded
            }
            return Response(
                r.content,
                status_code=r.status_code,
                headers=rh,
                media_type=r.headers.get("content-type"),
            )

        self.log_file = open("/tmp/sglang.log", "ab", buffering=0)
        cmd = ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate && exec " + " ".join(f'"{c}"' for c in build_sglang_command())]
        print("Starting SGLang:", " ".join(cmd), flush=True)
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        url = f"http://127.0.0.1:{SGLANG_PORT}/health"
        last = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = Path("/tmp/sglang.log").read_text(errors="replace")[-16000:]
                raise RuntimeError(
                    f"SGLang exited with code {self.process.returncode}\n{tail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if 200 <= r.status < 500:
                        print("SGLang is healthy.", flush=True)
                        break
            except Exception as e:
                last = e
            time.sleep(2)
        else:
            tail = Path("/tmp/sglang.log").read_text(errors="replace")[-16000:]
            raise RuntimeError(
                f"SGLang readiness timeout after {STARTUP_TIMEOUT_SECONDS}s; last_error={last}\n{tail}"
            )
        self.gateway_thread = threading.Thread(
            target=lambda: uvicorn.run(
                gateway, host="0.0.0.0", port=PORT, log_level="info"
            ),
            daemon=True,
        )
        self.gateway_thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not self.gateway_thread.is_alive():
                raise RuntimeError("Gateway thread exited unexpectedly")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status < 500:
                        break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            print("Warning: Gateway readiness check timed out, proceeding anyway", flush=True)
        print(f"Gateway listening on :{PORT}", flush=True)

    @modal.exit()
    def shutdown(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            except Exception as e:
                print(f"SGLang shutdown warning: {e}", flush=True)
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
        if self.sglang_client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.sglang_client.aclose())
                else:
                    loop.run_until_complete(self.sglang_client.aclose())
            except Exception:
                pass


@app.server(
    port=PORT,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=MIN_CONTAINERS,
    target_concurrency=TARGET_CONCURRENCY,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={"/root/.cache/huggingface": HF_CACHE},
    secrets=[
        modal.Secret.from_name(
            os.getenv("MODAL_API_SECRET_NAME", "deepseek-v4-flash-api")
        )
    ],
    routing_region=ROUTING_REGION,
    compute_region=COMPUTE_REGION,
)
class Server(DeepSeekServer):
    pass
