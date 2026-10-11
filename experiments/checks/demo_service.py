"""Step-by-step tour of the test service (payment-service): what it does and how it fails.

The service never has real errors: it fails only when told to, through /chaos/*. This script
shows each step: a normal payment, the counters, an injected fault, the failed payment, the
error log line, and the recovery. The service must be running (docker compose ... target-app).

Usage: python experiments/checks/demo_service.py
"""

import subprocess
import time
import urllib.error
import urllib.request

SERVICE = "http://127.0.0.1:8080"


def call(method: str, path: str) -> tuple[int, str]:
    request = urllib.request.Request(SERVICE + path, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def counters() -> dict[str, int]:
    _, text = call("GET", "/metrics")
    found = {"200": 0, "500": 0}
    for line in text.splitlines():
        if line.startswith("http_requests_total") and 'path="/pay"' in line:
            status = line.split('status="')[1].split('"')[0]
            found[status] = int(float(line.rsplit(" ", 1)[1]))
    return found


def last_error_log() -> str:
    output = subprocess.run(
        ["docker", "compose", "logs", "target-app", "--no-log-prefix", "--tail", "30"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    errors = [line for line in output.splitlines() if line.startswith("ERROR")]
    return errors[-1] if errors else "(aucune ligne ERROR)"


def step(number: int, title: str) -> None:
    print(f"\n=== ETAPE {number} : {title}")


def main() -> None:
    call("POST", "/chaos/reset")

    step(1, "un paiement normal")
    status, body = call("GET", "/pay")
    print(f"   GET /pay  ->  code {status}  {body}")
    print(f"   compteurs : {counters()}   (200 = reussis, 500 = en erreur)")

    step(2, "TOI : tu demandes au service d'echouer (100 %, connexion refusee)")
    status, body = call("POST", "/chaos/errors?rate=1&kind=connection_refused")
    print(f"   POST /chaos/errors  ->  {body}")

    step(3, "un paiement pendant la panne")
    status, body = call("GET", "/pay")
    print(f"   GET /pay  ->  code {status}  {body}")
    print(f"   compteurs : {counters()}   <- le compteur 500 a augmente")
    time.sleep(1)
    print(f"   log ecrit par le service : {last_error_log()}")

    step(4, "TOI : tu gueris le service")
    status, body = call("POST", "/chaos/reset")
    print(f"   POST /chaos/reset  ->  {body}")
    status, body = call("GET", "/pay")
    print(f"   GET /pay  ->  code {status}  {body}   <- de nouveau normal")

    print(
        "\nC'est tout ce que fait le service. Ton agent, lui, lit ces compteurs (via Prometheus)"
        "\net ces lignes ERROR (via Loki), et DECOUVRE seul que le service va mal."
    )


if __name__ == "__main__":
    main()
