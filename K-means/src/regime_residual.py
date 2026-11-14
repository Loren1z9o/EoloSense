"""Use K-means operating regimes to score deviations in other sensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kmeans_baseline import MODEL_ROOT, PROJECT_ROOT, split_rows
from shared_reference import candidate_features, load_event


def describe(frame: pd.DataFrame, mask: pd.Series) -> dict[str, float | int | None]:
    subset = frame.loc[mask]
    if subset.empty:
        return {"rows": 0, "flagged": 0, "flag_rate": None, "median_score": None}
    return {
        "rows": int(len(subset)),
        "flagged": int(subset["flagged"].sum()),
        "flag_rate": float(subset["flagged"].mean()),
        "median_score": float(subset["score"].median()),
    }


def numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def run(args: argparse.Namespace) -> None:
    event_ids = list(dict.fromkeys(args.events))
    if args.reference_event not in event_ids:
        raise ValueError("O evento de referência deve estar em --events.")
    loaded = {event_id: load_event(args.data_dir, args.farm, event_id) for event_id in event_ids}
    assets = {
        event_id: set(frame["asset_id"].dropna().astype(str))
        for event_id, (_, frame) in loaded.items()
    }
    if any(len(value) != 1 for value in assets.values()) or len(set.union(*assets.values())) != 1:
        raise ValueError(f"Os eventos devem pertencer à mesma turbina: {assets}")
    asset_id = next(iter(assets[event_ids[0]]))
    predictions = {}
    for event_id, (_, frame) in loaded.items():
        _, prediction_mask = split_rows(frame)
        prediction = frame.loc[prediction_mask].copy()
        if prediction.empty or prediction["time_stamp"].duplicated().any():
            raise ValueError(f"Previsão vazia ou com timestamps duplicados: evento {event_id}.")
        predictions[event_id] = prediction
    for index, first in enumerate(event_ids):
        for second in event_ids[index + 1:]:
            overlap = predictions[first]["time_stamp"].isin(predictions[second]["time_stamp"]).sum()
            if overlap:
                raise ValueError(f"Previsões {first} e {second} se sobrepõem em {overlap} linhas.")

    cutoff = min(frame["time_stamp"].min() for frame in predictions.values())
    reference_path, reference = loaded[args.reference_event]
    train_mask, _ = split_rows(reference)
    normal_mask = pd.to_numeric(reference["status_type_id"], errors="coerce").eq(0)
    eligible = reference.loc[
        train_mask & normal_mask & reference["time_stamp"].lt(cutoff)
    ].sort_values("time_stamp").drop_duplicates("time_stamp")
    if len(eligible) < 100:
        raise ValueError("Poucas linhas normais anteriores à primeira previsão.")
    calibration_start = pd.Timestamp(eligible.iloc[int(len(eligible) * args.fit_fraction)]["time_stamp"])
    fit_period = eligible.loc[
        eligible["time_stamp"].lt(calibration_start - pd.Timedelta(args.gap_days, unit="D"))
    ]
    calibration = eligible.loc[eligible["time_stamp"].ge(calibration_start)]
    if len(fit_period) < args.max_k + 2 or len(calibration) < 100:
        raise ValueError("Poucas linhas para ajuste ou calibração.")
    if len(fit_period) > args.max_fit_rows:
        rng = np.random.default_rng(args.seed)
        sample_index = np.sort(rng.choice(fit_period.index.to_numpy(), args.max_fit_rows, replace=False))
        fit_sample = fit_period.loc[sample_index]
    else:
        fit_sample = fit_period

    candidates = candidate_features(reference, reference_path.parent.parent / "feature_description.csv")
    fit_values = numeric_columns(fit_sample, candidates)
    usable = fit_values.columns[
        fit_values.isna().mean().le(args.max_missing)
        & fit_values.nunique(dropna=True).gt(1)
    ].tolist()
    regime_features = [
        col for col in usable if col.startswith("wind_speed_") or col.startswith("power_")
    ]
    residual_features = [col for col in usable if col not in regime_features]
    if len(regime_features) < 2 or len(residual_features) < 2:
        raise ValueError("São necessários ao menos 2 sensores de regime e 2 sensores residuais.")

    regime_imputer = SimpleImputer(strategy="median")
    regime_scaler = StandardScaler()
    x_regime_fit = regime_scaler.fit_transform(
        regime_imputer.fit_transform(fit_values[regime_features])
    )
    diagnostics = []
    for k in range(2, args.max_k + 1):
        if k >= len(x_regime_fit):
            break
        candidate = KMeans(n_clusters=k, n_init=10, random_state=args.seed)
        labels = candidate.fit_predict(x_regime_fit)
        if len(set(labels)) < 2:
            continue
        silhouette = silhouette_score(
            x_regime_fit, labels, sample_size=min(args.silhouette_sample, len(labels)),
            random_state=args.seed,
        )
        diagnostics.append({"k": k, "silhouette": float(silhouette)})
    if not diagnostics:
        raise ValueError("Não foi possível selecionar k para os regimes.")
    best_k = max(diagnostics, key=lambda item: item["silhouette"])["k"]
    model = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed).fit(x_regime_fit)
    fit_regimes = model.predict(x_regime_fit)

    residual_imputer = SimpleImputer(strategy="median")
    x_residual_fit = residual_imputer.fit_transform(fit_values[residual_features])
    global_spread = x_residual_fit.std(axis=0)
    centers = np.stack([
        np.median(x_residual_fit[fit_regimes == cluster], axis=0)
        for cluster in range(best_k)
    ])
    spreads = np.stack([
        x_residual_fit[fit_regimes == cluster].std(axis=0)
        for cluster in range(best_k)
    ])
    spreads = np.maximum(spreads, np.maximum(global_spread * 0.1, 1e-6))

    def score(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        operating = regime_scaler.transform(
            regime_imputer.transform(numeric_columns(frame, regime_features))
        )
        clusters = model.predict(operating)
        residuals = residual_imputer.transform(numeric_columns(frame, residual_features))
        z = (residuals - centers[clusters]) / spreads[clusters]
        # RMS across non-operating sensors; clip extreme single-sensor values.
        scores = np.sqrt(np.mean(np.minimum(z * z, 100.0), axis=1))
        return clusters, scores

    calibration_regimes, calibration_scores = score(calibration)
    thresholds = {}
    for cluster in range(best_k):
        cluster_scores = calibration_scores[calibration_regimes == cluster]
        if len(cluster_scores) < 30:
            raise ValueError(f"Regime {cluster} tem menos de 30 linhas de calibração.")
        thresholds[cluster] = float(np.quantile(cluster_scores, args.quantile, method="lower"))

    event_tag = "_".join(event_ids)
    output = args.output_dir / f"farm_{args.farm}_asset_{asset_id}_events_{event_tag}"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "regime_features": regime_features, "residual_features": residual_features,
        "regime_imputer": regime_imputer, "regime_scaler": regime_scaler,
        "model": model, "residual_imputer": residual_imputer,
        "centers": centers, "spreads": spreads, "thresholds": thresholds,
    }, output / "model.joblib")

    info = pd.read_csv(reference_path.parent.parent / "event_info.csv", sep=";").set_index("event_id")
    results = {}
    fig, axes = plt.subplots(len(event_ids), 1, figsize=(11, 3.5 * len(event_ids)), squeeze=False)
    for row_number, event_id in enumerate(event_ids):
        prediction = predictions[event_id].sort_values("time_stamp").copy()
        prediction["regime"], prediction["score"] = score(prediction)
        prediction["threshold"] = prediction["regime"].map(thresholds)
        prediction["flagged"] = prediction["score"].gt(prediction["threshold"])
        event_info = info.loc[int(event_id)]
        start = pd.Timestamp(event_info["event_start"])
        end = pd.Timestamp(event_info["event_end"])
        results[event_id] = {
            "label": str(event_info["event_label"]),
            "event_start": start.isoformat(), "event_end": end.isoformat(),
            "all": describe(prediction, pd.Series(True, index=prediction.index)),
            "before": describe(prediction, prediction["time_stamp"].lt(start)),
            "during": describe(prediction, prediction["time_stamp"].between(start, end)),
            "after": describe(prediction, prediction["time_stamp"].gt(end)),
            "status_0": describe(prediction, prediction["status_type_id"].eq(0)),
        }
        prediction[[
            "time_stamp", "asset_id", "id", "status_type_id", "regime", "score",
            "threshold", "flagged",
        ]].to_csv(output / f"event_{event_id}_scores.csv", index=False)
        axis = axes[row_number, 0]
        axis.plot(prediction["time_stamp"], prediction["score"], lw=0.8, label="desvio dos sensores")
        axis.plot(prediction["time_stamp"], prediction["threshold"], color="red", ls="--", label="limiar do regime")
        axis.axvspan(start, end, color="orange", alpha=0.15, label="intervalo do evento")
        axis.set_title(f"Evento {event_id}: {event_info['event_label']}")
        axis.set_ylabel("Desvio normalizado")
        axis.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "scores_timeline.png", dpi=160)
    plt.close(fig)

    report = {
        "method": "K-means regimes on wind speed and power; per-regime residual sensor score",
        "farm": args.farm, "asset_id": asset_id,
        "reference_event": args.reference_event, "evaluation_events": event_ids,
        "earliest_prediction": cutoff.isoformat(),
        "fit_rows_available": len(fit_period), "fit_rows_used": len(fit_sample),
        "fit_end": fit_period["time_stamp"].max().isoformat(),
        "calibration_start": calibration_start.isoformat(),
        "calibration_end": calibration["time_stamp"].max().isoformat(),
        "calibration_rows": len(calibration), "gap_days": args.gap_days,
        "regime_features": regime_features, "residual_features": residual_features,
        "selected_k": best_k, "k_diagnostics": diagnostics,
        "thresholds_by_regime": thresholds,
        "calibration_regime_counts": {
            str(cluster): int(np.sum(calibration_regimes == cluster)) for cluster in range(best_k)
        },
        "events": results,
        "limitation": "Exploratory scores, not a validated fault detector.",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Fazenda {args.farm}, turbina {asset_id}: {best_k} regimes; "
          f"{len(fit_sample)} linhas de ajuste, {len(calibration)} de calibração")
    for event_id, result in results.items():
        print(f"Evento {event_id} ({result['label']}): "
              f"{result['all']['flag_rate']:.1%} acima do limiar")
    print(f"Saída: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "outputs/regime_residual")
    parser.add_argument("--farm", choices=["A", "B", "C"], default="A")
    parser.add_argument("--reference-event", default="0")
    parser.add_argument("--events", nargs="+", default=["0", "24"])
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--max-fit-rows", type=int, default=20000)
    parser.add_argument("--max-missing", type=float, default=0.3)
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--silhouette-sample", type=int, default=3000)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fit_fraction < 1 or not 0 < args.quantile < 1:
        parser.error("fit-fraction e quantile devem estar entre 0 e 1.")
    if args.gap_days < 0 or args.max_k < 2 or args.max_fit_rows < 10 or args.silhouette_sample < 2:
        parser.error("Parâmetros de intervalo, k ou amostra inválidos.")
    if not 0 <= args.max_missing < 1:
        parser.error("max-missing deve estar entre 0 e 1.")
    try:
        run(args)
    except ValueError as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
