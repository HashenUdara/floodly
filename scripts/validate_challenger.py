"""Validate a retrained challenger model against the current champion."""

import argparse
import json
from pathlib import Path


def metric(metadata: dict, name: str) -> float:
    return float(metadata["metrics"][name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FloodLens challenger gates.")
    parser.add_argument("--champion-metadata", required=True)
    parser.add_argument("--challenger-metadata", required=True)
    parser.add_argument("--mae-tolerance", type=float, default=0.002)
    parser.add_argument("--rmse-tolerance", type=float, default=0.003)
    parser.add_argument("--report", default="challenger_report.json")
    args = parser.parse_args()

    champion = json.loads(Path(args.champion_metadata).read_text(encoding="utf-8"))
    challenger = json.loads(Path(args.challenger_metadata).read_text(encoding="utf-8"))

    champion_mae = metric(champion, "oof_mae")
    challenger_mae = metric(challenger, "oof_mae")
    champion_rmse = metric(champion, "oof_rmse")
    challenger_rmse = metric(challenger, "oof_rmse")
    test_min = metric(challenger, "test_min")
    test_max = metric(challenger, "test_max")

    gates = {
        "mae_gate": challenger_mae <= champion_mae + args.mae_tolerance,
        "rmse_gate": challenger_rmse <= champion_rmse + args.rmse_tolerance,
        "prediction_range_gate": 0 <= test_min <= test_max <= 1,
    }
    accepted = all(gates.values())
    report = {
        "accepted": accepted,
        "gates": gates,
        "champion": {
            "model_version": champion.get("model_version"),
            "oof_mae": champion_mae,
            "oof_rmse": champion_rmse,
        },
        "challenger": {
            "model_version": challenger.get("model_version"),
            "oof_mae": challenger_mae,
            "oof_rmse": challenger_rmse,
            "test_min": test_min,
            "test_max": test_max,
        },
        "tolerances": {
            "mae": args.mae_tolerance,
            "rmse": args.rmse_tolerance,
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not accepted:
        raise SystemExit("Challenger rejected by quality gates.")


if __name__ == "__main__":
    main()
