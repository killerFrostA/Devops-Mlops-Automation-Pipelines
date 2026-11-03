"""Lab workload with real failure behaviour, selected by ROLE.

- ``api``: payment-service. Each request takes a pooled connection to ledger-db over TCP.
  Faults come from its configuration, as in a bad release:
  ``LEAK_KB_PER_REQUEST`` keeps memory per request (grows until the OOM kill),
  ``PAYMENT_TIMEOUT_MS`` must be a positive integer (else exit 1, CrashLoopBackOff),
  ``DB_POOL_SIZE`` / ``DB_QUERY_MS`` size the pool (too small: pool exhaustion).
- ``db``: ledger-db, a TCP service answering on 5432 (scaled to 0 for an outage).
- ``loadgen``: steady traffic to payment-service.

Standard library only; metrics use Prometheus text format on /metrics.
"""

from __future__ import annotations

import contextlib
import os
import socket
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def log(level: str, message: str) -> None:
    print(f"{datetime.now(UTC).isoformat(timespec='milliseconds')} {level} {message}", flush=True)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0 or (value == 0 and name == "PAYMENT_TIMEOUT_MS"):
        log("FATAL", f"invalid configuration: {name}={raw!r} is not a valid positive integer")
        sys.exit(1)
    return value


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: Counter[str] = Counter()
        self.bucket_counts = [0] * len(BUCKETS)
        self.duration_sum = 0.0
        self.duration_count = 0

    def observe(self, status: int, seconds: float) -> None:
        with self.lock:
            self.requests[str(status)] += 1
            self.duration_sum += seconds
            self.duration_count += 1
            for index, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    self.bucket_counts[index] += 1

    def render(self) -> str:
        lines = ["# TYPE http_requests_total counter"]
        with self.lock:
            lines += [
                f'http_requests_total{{status="{code}"}} {count}'
                for code, count in sorted(self.requests.items())
            ]
            lines.append("# TYPE http_request_duration_seconds histogram")
            lines += [
                f'http_request_duration_seconds_bucket{{le="{bound}"}} {count}'
                for bound, count in zip(BUCKETS, self.bucket_counts, strict=True)
            ]
            lines.append(f'http_request_duration_seconds_bucket{{le="+Inf"}} {self.duration_count}')
            lines.append(f"http_request_duration_seconds_sum {self.duration_sum}")
            lines.append(f"http_request_duration_seconds_count {self.duration_count}")
        lines += [
            "# TYPE process_resident_memory_bytes gauge",
            f"process_resident_memory_bytes {_rss_bytes()}",
        ]
        return "\n".join(lines) + "\n"


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def run_api() -> None:
    version = os.environ.get("APP_VERSION", "unknown")
    timeout_ms = _int_env("PAYMENT_TIMEOUT_MS", 500)
    leak_kb = _int_env("LEAK_KB_PER_REQUEST", 0)
    pool_size = max(_int_env("DB_POOL_SIZE", 10), 1)
    query_ms = _int_env("DB_QUERY_MS", 20)
    pool_wait_ms = _int_env("DB_POOL_WAIT_MS", 200)
    db_host = os.environ.get("DB_HOST", "ledger-db")
    db_port = _int_env("DB_PORT", 5432)
    pool = threading.BoundedSemaphore(pool_size)
    metrics = Metrics()
    retained: list[bytes] = []  # The "response cache" of the leaking release.

    def handle_payment() -> int:
        if leak_kb:
            retained.append(b"\x01" * (leak_kb * 1024))
        if not pool.acquire(timeout=pool_wait_ms / 1000):
            log(
                "ERROR",
                f"could not obtain a connection: connection pool exhausted (size={pool_size})",
            )
            return 503
        try:
            with socket.create_connection((db_host, db_port), timeout=timeout_ms / 1000) as conn:
                conn.recv(16)
                time.sleep(query_ms / 1000)
            return 200
        except ConnectionRefusedError:
            log(
                "ERROR",
                f"payment failed: dial tcp {db_host}:{db_port}: connect: connection refused",
            )
        except TimeoutError:
            log("ERROR", f"payment failed: connect timed out to {db_host}:{db_port}")
        except socket.gaierror:
            log("ERROR", f"payment failed: dial tcp {db_host}:{db_port}: name or service not known")
        except OSError as exc:
            log("ERROR", f"payment failed: dial tcp {db_host}:{db_port}: {exc}")
        finally:
            pool.release()
        return 503

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/metrics":
                self._reply(200, metrics.render(), "text/plain; version=0.0.4")
            elif self.path == "/healthz":
                self._reply(200, "ok\n")
            else:
                started = time.perf_counter()
                status = handle_payment()
                metrics.observe(status, time.perf_counter() - started)
                self._reply(status, "paid\n" if status == 200 else "unavailable\n")

        def _reply(self, status: int, body: str, content_type: str = "text/plain") -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_: object) -> None:
            pass

    log("INFO", f"payment-service {version} started (pool={pool_size}, timeout={timeout_ms}ms)")
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


def run_db() -> None:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.sendall(b"OK\n")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    log("INFO", "ledger-db listening on 5432")
    socketserver.ThreadingTCPServer(("0.0.0.0", 5432), Handler).serve_forever()


def run_loadgen() -> None:
    target = os.environ.get("TARGET_URL", "http://payment-service:8080/pay")
    interval = 1 / float(os.environ.get("REQUESTS_PER_SECOND", "5"))
    concurrency = int(os.environ.get("CONCURRENCY", "4"))
    log("INFO", f"loadgen sending {1 / interval:.0f} rps x {concurrency} to {target}")

    def worker() -> None:
        while True:
            with contextlib.suppress(urllib.error.URLError, OSError):
                urllib.request.urlopen(target, timeout=5).read()
            time.sleep(interval)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    {"api": run_api, "db": run_db, "loadgen": run_loadgen}[os.environ.get("ROLE", "api")]()
