"""Summarize the non-overlapping exploratory Farm A event runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


MODEL_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "0": "71_24_73_0_26",
    "10": "40_3_42_10_17",
    "11": "92_25_68_69",
    "13": "14_45_38_84",
    "21": "13_22_51",  # 72 overlaps 51 for 1,918 prediction timestamps.
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=["shared_reference", "regime_residual"],
        default="shared_reference",
    )
    args = parser.parse_args()
    rows = []
    for asset, events in RUNS.items():
        path = (
            MODEL_ROOT / "outputs" / args.experiment
            / f"farm_A_asset_{asset}_events_{events}" / "report.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        for event_id, event in report["events"].items():
            rows.append({
                "asset_id": asset,
                "event_id": event_id,
                "event_label": event["label"],
                "event_start": event["event_start"],
                "before_rows": event["before"]["rows"],
                "before_flagged": event["before"]["flagged"],
                "before_flag_rate": event["before"]["flag_rate"],
                "all_rows": event["all"]["rows"],
                "all_flagged": event["all"]["flagged"],
                "all_flag_rate": event["all"]["flag_rate"],
                "selected_k": report["selected_k"],
            })
    frame = pd.DataFrame(rows).sort_values(["asset_id", "event_start"])
    if frame["event_id"].duplicated().any():
        raise ValueError("Um evento apareceu em mais de um relatório.")
    early = frame.loc[frame["before_rows"].gt(0)]
    metrics = {
        "events_total": int(len(frame)),
        "normal_events": int(frame["event_label"].eq("normal").sum()),
        "anomaly_events": int(frame["event_label"].eq("anomaly").sum()),
        "events_with_pre_event_rows": int(len(early)),
        "normal_with_pre_event_rows": int(early["event_label"].eq("normal").sum()),
        "anomaly_with_pre_event_rows": int(early["event_label"].eq("anomaly").sum()),
        "auc_all_prediction_flag_rate": float(roc_auc_score(
            frame["event_label"].eq("anomaly"), frame["all_flag_rate"]
        )),
        "auc_pre_event_flag_rate": float(roc_auc_score(
            early["event_label"].eq("anomaly"), early["before_flag_rate"]
        )),
        "excluded_event": "72: 1,918 of its 2,017 prediction timestamps overlap event 51",
        "experiment": args.experiment,
        "limitation": "Exploratory ranking only: small, correlated event set; no validated fault detector.",
    }
    suffix = "" if args.experiment == "shared_reference" else "_regime_residual"
    output = MODEL_ROOT / "outputs" / f"farm_A_summary{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "events.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{metrics['events_total']} eventos: {metrics['normal_events']} normais, "
          f"{metrics['anomaly_events']} com anomalia")
    print(f"AUC antes do evento: {metrics['auc_pre_event_flag_rate']:.3f} "
          f"({metrics['events_with_pre_event_rows']} eventos com dados anteriores)")
    print(f"AUC da previsão inteira: {metrics['auc_all_prediction_flag_rate']:.3f}")
    print(f"Saída: {output}")


if __name__ == "__main__":
    main()
