"""Live check of the log detector (step 5) against the running test bench (Loki on :3100).

Usage: python experiments/checks/live_logs_check.py
Gemini is used only if PIDS_GEMINI_API_KEY is set in .env; otherwise unknown templates stay
UNKNOWN. The bench policy uses a 10 min baseline and a 2 min recent window.
"""

import asyncio

from src.agents.monitoring.collectors.loki import LokiClient
from src.agents.monitoring.detectors.logs import LogDetector, LogPatternDetector, LogPolicy
from src.agents.monitoring.labelers.gemini import GeminiLabeler
from src.config import Settings

SERVICE = "payment-service"
BENCH_POLICY = LogPolicy(baseline_seconds=600, recent_seconds=120)


async def main() -> None:
    settings = Settings()
    labeler = (
        GeminiLabeler(settings.gemini_api_key, settings.gemini_model)
        if settings.gemini_api_key
        else None
    )
    print(f"== Etape 5 - logs ({SERVICE}), Gemini: {'active' if labeler else 'non configure'}")
    async with LokiClient(settings.loki_url, timeout_seconds=5) as loki:
        detector = LogDetector(loki, LogPatternDetector(labeler, BENCH_POLICY))
        findings = await detector.evaluate(SERVICE)
    if labeler:
        await labeler.aclose()
    if not findings:
        print("   aucun nouveau modele ni hausse anormale")
    for f in findings:
        review = " [a verifier]" if f.requires_review else ""
        print(
            f"   {f.kind:11} {f.label:20} ({f.label_source}) recent={f.count_recent:4} "
            f"habituel={f.count_baseline:4}{review}"
        )
        print(f"      modele : {f.template}")
        print(f"      exemple: {f.example}")


if __name__ == "__main__":
    asyncio.run(main())
