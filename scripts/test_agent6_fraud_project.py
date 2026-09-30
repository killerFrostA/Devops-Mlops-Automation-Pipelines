"""Offline Agent 6 health check against an external application's actual saved model.

The CSV has no event IDs or timestamps, so this is a static cohort test, not live monitoring.
The scorer runs in the application's model-compatible Python environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES = (
    "accountAgeDays",
    "numItems",
    "localTime",
    "paymentMethod",
    "paymentMethodAgeDays",
    "Category",
    "isWeekend",
)
NUMERIC_FEATURES = ("accountAgeDays", "numItems", "paymentMethodAgeDays")


def _score_model(input_path: Path, output_path: Path) -> None:
    """Run only in the model's isolated Python environment, from the application root."""
    import mlflow
    import pandas as pd
    import sklearn
    from API.services import MODEL_ARTIFACT_PATH, model

    frame = pd.read_csv(input_path)
    predictions = [int(value) for value in model.predict(frame[list(FEATURES)])]
    if set(predictions) - {0, 1}:
        raise ValueError("The serving model must return binary predictions")
    artifact_dir = Path(MODEL_ARTIFACT_PATH)
    artifact_hash = hashlib.sha256()
    for file in sorted(path for path in artifact_dir.rglob("*") if path.is_file()):
        artifact_hash.update(file.relative_to(artifact_dir).as_posix().encode("utf-8"))
        artifact_hash.update(file.read_bytes())
    output_path.write_text(
        json.dumps(
            {
                "predictions": predictions,
                "model_artifact_sha256": artifact_hash.hexdigest(),
                "model_artifact": MODEL_ARTIFACT_PATH,
                "python_version": sys.version.split()[0],
                "sklearn_version": sklearn.__version__,
                "mlflow_version": mlflow.__version__,
            }
        ),
        encoding="utf-8",
    )


def _numeric_distribution(name: str, reference: object, current: object) -> object:
    import numpy as np
    import pandas as pd

    from src.contracts.mlops import DistributionComparison

    reference_values = pd.to_numeric(reference, errors="raise").to_numpy(dtype=float)
    current_values = pd.to_numeric(current, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(reference_values).all() or not np.isfinite(current_values).all():
        raise ValueError(f"{name} contains non-finite values")
    edges = np.unique(np.quantile(reference_values, [0.2, 0.4, 0.6, 0.8]))
    if len(edges) == 0:
        raise ValueError(f"{name} has no usable reference bins")
    reference_counts = np.bincount(
        np.searchsorted(edges, reference_values, side="right"), minlength=len(edges) + 1
    )
    current_counts = np.bincount(
        np.searchsorted(edges, current_values, side="right"), minlength=len(edges) + 1
    )
    bins = tuple(f"bin_{index}" for index in range(len(edges) + 1))
    return DistributionComparison(
        name=name,
        scope="feature",
        bins=bins,
        reference_counts=tuple(int(value) for value in reference_counts),
        current_counts=tuple(int(value) for value in current_counts),
    )


def _categorical_distribution(reference: object, current: object) -> object:
    import pandas as pd

    from src.contracts.mlops import DistributionComparison

    ref_values = pd.Series(reference).fillna("__MISSING__").astype(str)
    cur_values = pd.Series(current).fillna("__MISSING__").astype(str)
    categories = sorted(set(ref_values) - {"__MISSING__"})
    bins = (*categories, "__OTHER__", "__MISSING__")

    def counts(values: object) -> tuple[int, ...]:
        counter = Counter(
            value if value in categories or value == "__MISSING__" else "__OTHER__"
            for value in values
        )
        return tuple(counter.get(name, 0) for name in bins)

    return DistributionComparison(
        name="Category",
        scope="feature",
        bins=bins,
        reference_counts=counts(ref_values),
        current_counts=counts(cur_values),
    )


def _prediction_distribution(reference: list[int], current: list[int]) -> object:
    from src.contracts.mlops import DistributionComparison

    return DistributionComparison(
        name="predicted_class",
        scope="prediction",
        bins=("0", "1"),
        reference_counts=(reference.count(0), reference.count(1)),
        current_counts=(current.count(0), current.count(1)),
    )


def run(
    target_root: Path, model_python: Path, output_dir: Path, cohort_size: int
) -> dict[str, object]:
    import pandas as pd
    from sklearn.metrics import f1_score

    from src.agents.mlops_lifecycle.service import MLOpsLifecycleAgent
    from src.contracts.mlops import LabeledPredictionWindow, ModelHealthInput

    target_root = target_root.resolve()
    output_dir = output_dir.resolve()
    model_python = model_python.resolve()
    dataset_path = target_root / "Data/payment_fraud.csv"
    if not dataset_path.is_file():
        raise ValueError(f"Fraud dataset not found: {dataset_path}")
    if not (target_root / "API/services.py").is_file():
        raise ValueError("The target has no expected API/services.py model loader")
    if not model_python.is_file():
        raise ValueError(f"Model Python executable not found: {model_python}")
    if cohort_size < 100:
        raise ValueError("At least 100 records per cohort are required")
    frame = pd.read_csv(dataset_path)
    required = {*FEATURES, "label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Fraud dataset lacks columns: {sorted(missing)}")
    if len(frame) < 2 * cohort_size:
        raise ValueError("Dataset needs two disjoint cohorts")
    if frame["label"].isna().any() or not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Fraud labels must be confirmed binary values")
    reference = frame.head(cohort_size).copy()
    current = frame.tail(cohort_size).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mlruns").mkdir(exist_ok=True)
    (output_dir / "mpl").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="score-", dir=output_dir) as temporary:
        input_path = Path(temporary) / "cohorts.csv"
        predictions_path = Path(temporary) / "predictions.json"
        pd.concat([reference, current], ignore_index=True).to_csv(input_path, index=False)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(target_root)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MPLCONFIGDIR"] = str(output_dir / "mpl")
        environment["MLFLOW_TRACKING_URI"] = (output_dir / "mlruns").as_uri()
        environment["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        completed = subprocess.run(
            [
                str(model_python),
                str(Path(__file__).resolve()),
                "--score-input",
                str(input_path),
                "--score-output",
                str(predictions_path),
            ],
            cwd=target_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Serving-model scoring failed:\n{completed.stderr[-3000:]}")
        scored = json.loads(predictions_path.read_text(encoding="utf-8"))

    predictions = scored["predictions"]
    if len(predictions) != 2 * cohort_size:
        raise ValueError("Serving model returned the wrong number of predictions")
    baseline_predictions = predictions[:cohort_size]
    current_predictions = predictions[cohort_size:]
    baseline_labels = [int(value) for value in reference["label"]]
    current_labels = [int(value) for value in current["label"]]
    baseline_f1 = float(f1_score(baseline_labels, baseline_predictions, zero_division=0))
    distributions = [
        _numeric_distribution(name, reference[name], current[name]) for name in NUMERIC_FEATURES
    ]
    distributions.append(_categorical_distribution(reference["Category"], current["Category"]))
    distributions.append(_prediction_distribution(baseline_predictions, current_predictions))
    import numpy as np

    numeric_bin_edges = {
        name: [
            float(value)
            for value in np.unique(
                np.quantile(reference[name].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8])
            )
        ]
        for name in NUMERIC_FEATURES
    }
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    window_start = datetime(2026, 9, 29, tzinfo=UTC)
    request = ModelHealthInput(
        model_name="fraud-detection",
        model_version=f"api-artifact-{scored['model_artifact_sha256'][:12]}",
        reference_id=f"csv-reference-first-{cohort_size}",
        window_start=window_start,
        window_end=window_start + timedelta(days=1),
        distributions=tuple(distributions),
        evidence_refs=(
            f"sha256:{dataset_digest}",
            f"sha256:{scored['model_artifact_sha256']}",
        ),
        performance=LabeledPredictionWindow(
            predictions=tuple(current_predictions),
            labels=tuple(current_labels),
            baseline_f1=baseline_f1,
        ),
        dataset_ref="fraud-project-offline-csv",
    )
    report = asyncio.run(MLOpsLifecycleAgent().assess_model_health(request))
    result: dict[str, object] = {
        "test_kind": "offline_external_project_health",
        "target_root": str(target_root),
        "dataset_sha256": dataset_digest,
        "serving_model_artifact_sha256": scored["model_artifact_sha256"],
        "serving_model_artifact": scored["model_artifact"],
        "scoring_environment": {
            "python": scored["python_version"],
            "scikit_learn": scored["sklearn_version"],
            "mlflow": scored["mlflow_version"],
        },
        "numeric_bin_edges": numeric_bin_edges,
        "reference": {
            "selection": f"first {cohort_size} CSV rows",
            "rows": cohort_size,
            "positive_labels": sum(baseline_labels),
            "f1": baseline_f1,
        },
        "current": {
            "selection": f"last {cohort_size} CSV rows",
            "rows": cohort_size,
            "positive_labels": sum(current_labels),
            "f1": report.performance.current if report.performance else None,
        },
        "agent6_health_report": report.model_dump(mode="json"),
        "limitations": [
            "CSV row order is not a production event-time window.",
            "The dataset has no transaction IDs, event timestamps or label-confirmation times.",
            "The saved model may have trained on these rows; F1 only checks wiring.",
            "The API exposes only class predictions, so calibrated score drift is not assessed.",
            "This command assesses health only; it does not train, register or deploy a model.",
        ],
    }
    (output_dir / "fraud-project-health-report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/agent6-fraud-project/assessment"
    )
    parser.add_argument("--cohort-size", type=int, default=2000)
    parser.add_argument("--score-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--score-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.score_input or args.score_output:
        if not args.score_input or not args.score_output:
            parser.error("Both score-input and score-output are required")
        _score_model(args.score_input, args.score_output)
        return
    if args.target_root is None or args.model_python is None:
        parser.error("--target-root and --model-python are required")
    result = run(args.target_root, args.model_python, args.output_dir, args.cohort_size)
    print(
        json.dumps(
            {
                "decision": result["agent6_health_report"]["assessment"]["decision"],
                "drifted_features": [
                    item["name"]
                    for item in result["agent6_health_report"]["distributions"]
                    if item["detected"]
                ],
                "reference_f1": result["reference"]["f1"],
                "current_f1": result["current"]["f1"],
                "report": str((args.output_dir / "fraud-project-health-report.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
