"""Steady background traffic for the target service (standard library only).

Usage: python experiments/load/traffic.py --url http://127.0.0.1:8080/pay --rps 5
"""

import argparse
import threading
import time
import urllib.error
import urllib.request


def send(url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError):
        pass  # Failures are expected during injected faults; the service metrics record them.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/pay")
    parser.add_argument("--rps", type=float, default=5.0, help="requests per second")
    parser.add_argument("--duration", type=float, default=0, help="seconds; 0 runs until Ctrl+C")
    args = parser.parse_args()

    interval = 1 / args.rps
    deadline = time.monotonic() + args.duration if args.duration else float("inf")
    print(f"Sending {args.rps} req/s to {args.url} (Ctrl+C to stop)")
    try:
        while time.monotonic() < deadline:
            # One thread per request so injected latency does not lower the request rate.
            threading.Thread(target=send, args=(args.url,), daemon=True).start()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
