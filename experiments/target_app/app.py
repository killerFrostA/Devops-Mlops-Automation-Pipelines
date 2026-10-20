"""Instrumented target service observed by Agent 1, with opt-in fault injection.

Fault endpoints exist only when CHAOS_ENABLED=true (local/test). Every injection is appended to
an injection log so detection delay (MTTD) can be measured against the exact injection time.
"""

import asyncio
import json
import logging
import os
import random
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

SERVICE_NAME = os.getenv("SERVICE_NAME", "payment-service")
CHAOS_ENABLED = os.getenv("CHAOS_ENABLED", "false").lower() == "true"
INJECTION_LOG = Path(os.getenv("INJECTION_LOG", "experiments/runs/injections.jsonl"))
MAX_MEMORY_MB = int(os.getenv("CHAOS_MAX_MEMORY_MB", "2048"))
# Optional downstream service called on every /pay (e.g. http://payments-db:8080): its faults
# propagate to this service, as in a real call chain. Empty: no dependency.
DEPENDENCY_URL = os.getenv("DEPENDENCY_URL", "").rstrip("/")
DEPENDENCY_NAME = os.getenv("DEPENDENCY_NAME", "payments-db")
DEPENDENCY_TIMEOUT_SECONDS = float(os.getenv("DEPENDENCY_TIMEOUT_SECONDS", "2"))

# Bucket 0.5 matches the 500 ms latency SLO: requests <= 0.5 s are "good events".
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUESTS = Counter(
    "http_requests_total", "HTTP requests handled", ["service", "method", "path", "status"]
)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(SERVICE_NAME)

# Log line written for each injected failure, by error kind (read by Agent 1 through Loki).
ERROR_MESSAGES = {
    "server_error": "Unhandled exception in /pay: ValueError request_id={request_id}",
    "connection_refused": "Connection refused to payments-db:5432 request_id={request_id}",
    "timeout": "Timeout after 30s calling payment-gateway request_id={request_id}",
    "auth": "Invalid token for merchant request_id={request_id}",
}

LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["service", "path"],
    buckets=LATENCY_BUCKETS,
)


class ChaosState:
    """Active faults. Plain attributes: one event loop, CPU burn runs in its own thread."""

    def __init__(self) -> None:
        self.error_rate = 0.0
        self.error_kind = "server_error"
        self.latency_ms = 0
        self.memory_mb_per_min = 0.0
        self.cpu_until = 0.0
        self.leaked: list[bytearray] = []

    def reset(self) -> None:
        self.error_rate = 0.0
        self.error_kind = "server_error"
        self.latency_ms = 0
        self.memory_mb_per_min = 0.0
        self.cpu_until = 0.0
        self.leaked.clear()


chaos = ChaosState()


def record_injection(fault: str, parameters: dict[str, Any]) -> None:
    INJECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "injected_at": datetime.now(UTC).isoformat(),
        "service": SERVICE_NAME,
        "fault": fault,
        "parameters": parameters,
    }
    with INJECTION_LOG.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")


async def grow_memory() -> None:
    """Allocate memory every second while a memory fault is active, up to MAX_MEMORY_MB.

    The target is cumulative (rate x elapsed time), so fractional rates such as 0.5 MB/s are
    honoured exactly instead of being rounded up to 1 MB every second.
    """
    started: float | None = None
    while True:
        await asyncio.sleep(1)
        if chaos.memory_mb_per_min <= 0:
            started = None
            continue
        if started is None:
            started = time.monotonic() - len(chaos.leaked) * 60 / chaos.memory_mb_per_min
        target_mb = int((time.monotonic() - started) * chaos.memory_mb_per_min / 60)
        while len(chaos.leaked) < min(target_mb, MAX_MEMORY_MB):
            chaos.leaked.append(bytearray(1024 * 1024))


def burn_cpu() -> None:
    while time.monotonic() < chaos.cpu_until:
        sum(i * i for i in range(10_000))


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    task = asyncio.create_task(grow_memory())
    yield
    task.cancel()


app = FastAPI(title=SERVICE_NAME, lifespan=lifespan)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def measure(request: Request, call_next):  # type: ignore[no-untyped-def]
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        response = JSONResponse({"detail": "internal error"}, status_code=500)
    route = request.scope.get("route")
    # Label by route template, not raw URL, to keep metric cardinality bounded.
    label = getattr(route, "path", "unmatched")
    LATENCY.labels(SERVICE_NAME, label).observe(time.perf_counter() - start)
    REQUESTS.labels(SERVICE_NAME, request.method, label, str(response.status_code)).inc()
    return response


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pay")
async def pay() -> dict[str, str]:
    """Simulated business endpoint; active faults change its behaviour."""
    if chaos.latency_ms:
        await asyncio.sleep(chaos.latency_ms / 1000)
    if random.random() < chaos.error_rate:
        request_id = f"r{random.randrange(10**6):06d}"
        log.error(ERROR_MESSAGES[chaos.error_kind].format(request_id=request_id))
        raise HTTPException(status_code=500, detail="injected failure")
    if DEPENDENCY_URL:
        await use_dependency()
    else:
        await asyncio.sleep(random.uniform(0.01, 0.06))
    return {"status": "ok"}


def call_dependency() -> None:
    """Blocking call to the downstream service (run in a worker thread)."""
    with urllib.request.urlopen(f"{DEPENDENCY_URL}/pay", timeout=DEPENDENCY_TIMEOUT_SECONDS):
        pass


async def use_dependency() -> None:
    """Call the dependency; its failure becomes this service's failure, with a log line."""
    request_id = f"r{random.randrange(10**6):06d}"
    try:
        await asyncio.to_thread(call_dependency)
    except urllib.error.HTTPError as error:
        log.error(f"Dependency {DEPENDENCY_NAME} failed: HTTP {error.code} request_id={request_id}")
        raise HTTPException(status_code=500, detail="dependency failure") from error
    except TimeoutError as error:
        log.error(
            f"Timeout after {DEPENDENCY_TIMEOUT_SECONDS:g}s calling {DEPENDENCY_NAME} "
            f"request_id={request_id}"
        )
        raise HTTPException(status_code=500, detail="dependency timeout") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            log.error(f"Timeout calling {DEPENDENCY_NAME} request_id={request_id}")
        else:
            log.error(f"Connection refused to {DEPENDENCY_NAME} request_id={request_id}")
        raise HTTPException(status_code=500, detail="dependency unreachable") from error


def require_chaos() -> None:
    if not CHAOS_ENABLED:
        raise HTTPException(status_code=404, detail="fault injection disabled")


@app.post("/chaos/memory")
async def inject_memory(mb_per_min: float = 10.0) -> dict[str, Any]:
    require_chaos()
    chaos.memory_mb_per_min = mb_per_min
    record_injection("memory_growth", {"mb_per_min": mb_per_min})
    return {"fault": "memory_growth", "mb_per_min": mb_per_min}


@app.post("/chaos/cpu")
async def inject_cpu(seconds: int = 120) -> dict[str, Any]:
    require_chaos()
    chaos.cpu_until = time.monotonic() + seconds
    threading.Thread(target=burn_cpu, daemon=True).start()
    record_injection("cpu_burn", {"seconds": seconds})
    return {"fault": "cpu_burn", "seconds": seconds}


@app.post("/chaos/errors")
async def inject_errors(rate: float = 0.2, kind: str = "server_error") -> dict[str, Any]:
    """`kind` chooses the error message written to the logs for each failed request."""
    require_chaos()
    if not 0 <= rate <= 1:
        raise HTTPException(status_code=422, detail="rate must be between 0 and 1")
    if kind not in ERROR_MESSAGES:
        raise HTTPException(status_code=422, detail=f"kind must be one of {list(ERROR_MESSAGES)}")
    chaos.error_rate = rate
    chaos.error_kind = kind
    record_injection("http_errors", {"rate": rate, "kind": kind})
    return {"fault": "http_errors", "rate": rate, "kind": kind}


@app.post("/chaos/latency")
async def inject_latency(ms: int = 800) -> dict[str, Any]:
    require_chaos()
    chaos.latency_ms = ms
    record_injection("latency", {"ms": ms})
    return {"fault": "latency", "ms": ms}


@app.post("/chaos/reset")
async def reset() -> dict[str, str]:
    require_chaos()
    chaos.reset()
    record_injection("reset", {})
    return {"status": "reset"}
