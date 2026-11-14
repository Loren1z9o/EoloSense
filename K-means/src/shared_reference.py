"""Train one K-means reference and score several events of the same turbine."""

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

from kmeans_baseline import MODEL_ROOT, PROJECT_ROOT, choose_file, split_rows


def distance_to_centroid(model: KMeans, matrix: np.ndarray) -> np.ndarray:
    return model.transform(matrix).min(axis=1)


def load_event(data_dir: Path, farm: str, event_id: str) -> tuple[Path, pd.DataFrame]:
    path = choose_file(data_dir, farm, event_id)
    frame = pd.read_csv(path, sep=";", low_memory=False)
    required = {"time_stamp", "asset_id", "id", "train_test", "status_type_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: colunas ausentes: {sorted(missing)}")
    frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], errors="coerce")
    if frame["time_stamp"].isna().any():
        raise ValueError(f"{path}: timestamps inválidos.")
    return path, frame


def candidate_features(frame: pd.DataFrame, descriptions_path: Path) -> list[str]:
    features = [name for name in frame if name.lower().endswith("_avg")]
    if descriptions_path.exists():
        descriptions = pd.read_csv(descriptions_path, sep=";")
        required = {"sensor_name", "is_angle", "is_counter"}
        if required.issubset(descriptions.columns):
            excluded = set(
                descriptions.loc[
                    descriptions["is_angle"].astype(str).str.lower().eq("true")
                    | descriptions["is_counter"].astype(str).str.lower().eq("true"),
                    "sensor_name",
                ].astype(str)
            )
            features = [name for name in features if name.removesuffix("_avg") not in excluded]
    return features


def describe_segment(frame: pd.DataFrame, mask: pd.Series) -> dict[str, float | int | None]:
    subset = frame.loc[mask]
    if subset.empty:
        return {"rows": 0, "flagged": 0, "flag_rate": None, "median_distance": None}
    return {
        "rows": int(len(subset)),
        "flagged": int(subset["flagged"].sum()),
        "flag_rate": float(subset["flagged"].mean()),
        "median_distance": float(subset["distance_to_centroid"].median()),
    }


def run(args: argparse.Namespace) -> None:
    event_ids = list(dict.fromkeys(args.events))
    if args.reference_event not in event_ids:
        raise ValueError("O evento de referência também deve estar em --events.")
    loaded = {event_id: load_event(args.data_dir, args.farm, event_id) for event_id in event_ids}
    assets = {
        event_id: set(frame["asset_id"].dropna().astype(str))
        for event_id, (_, frame) in loaded.items()
    }
    if any(len(asset_set) != 1 for asset_set in assets.values()) or len(set.union(*assets.values())) != 1:
        raise ValueError(f"Os eventos devem pertencer à mesma turbina: {assets}")

    predictions: dict[str, pd.DataFrame] = {}
    for event_id, (_, frame) in loaded.items():
        _, prediction_mask = split_rows(frame)
        prediction = frame.loc[prediction_mask].copy()
        if prediction.empty:
            raise ValueError(f"Evento {event_id} sem linhas de previsão.")
        if prediction["time_stamp"].duplicated().any():
            raise ValueError(f"Evento {event_id} tem timestamps duplicados na previsão.")
        predictions[event_id] = prediction
    for index, first in enumerate(event_ids):
        for second in event_ids[index + 1:]:
            overlap = predictions[first]["time_stamp"].isin(predictions[second]["time_stamp"]).sum()
            if overlap:
                raise ValueError(f"Previsões dos eventos {first} e {second} se sobrepõem ({overlap} linhas).")

    cutoff = min(frame["time_stamp"].min() for frame in predictions.values())
    reference_path, reference = loaded[args.reference_event]
    train_mask, _ = split_rows(reference)
    normal_mask = pd.to_numeric(reference["status_type_id"], errors="coerce").eq(0)
    eligible = reference.loc[
        train_mask & normal_mask & reference["time_stamp"].lt(cutoff)
    ].sort_values("time_stamp").drop_duplicates("time_stamp")
    if len(eligible) < 100:
        raise ValueError("Menos de 100 linhas normais anteriores às previsões.")
    split_index = int(len(eligible) * args.fit_fraction)
    calibration_start = pd.Timestamp(eligible.iloc[split_index]["time_stamp"])
    fit_period = eligible.loc[
        eligible["time_stamp"].lt(calibration_start - pd.Timedelta(args.gap_days, unit="D"))
    ]
    calibration = eligible.loc[eligible["time_stamp"].ge(calibration_start)]
    if len(fit_period) < max(args.k_values) + 2 or len(calibration) < 30:
        raise ValueError("A divisão temporal deixou poucas linhas para ajuste ou calibração.")
    rng = np.random.default_rng(args.seed)
    if len(fit_period) > args.max_fit_rows:
        sample_indices = np.sort(rng.choice(fit_period.index.to_numpy(), args.max_fit_rows, replace=False))
        fit_sample = fit_period.loc[sample_indices]
    else:
        fit_sample = fit_period

    candidates = candidate_features(reference, reference_path.parent.parent / "feature_description.csv")
    fit_numeric = fit_sample[candidates].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    usable = fit_numeric.columns[
        fit_numeric.isna().mean().le(args.max_missing)
        & fit_numeric.nunique(dropna=True).gt(1)
    ].tolist()
    if len(usable) < 2:
        raise ValueError("Menos de 2 sensores úteis no período de ajuste.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(imputer.fit_transform(fit_numeric[usable]))

    diagnostics = []
    for k in args.k_values:
        if k >= len(x_fit):
            continue
        candidate = KMeans(n_clusters=k, n_init=10, random_state=args.seed)
        labels = candidate.fit_predict(x_fit)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(
            x_fit, labels, sample_size=min(args.silhouette_sample, len(x_fit)),
            random_state=args.seed,
        )
        diagnostics.append({"k": k, "silhouette": float(score), "inertia": float(candidate.inertia_)})
    if not diagnostics:
        raise ValueError("Não foi possível comparar os valores de k.")
    best_k = max(diagnostics, key=lambda item: item["silhouette"])["k"]
    model = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed).fit(x_fit)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        values = frame[usable].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        return scaler.transform(imputer.transform(values))

    calibration_distances = distance_to_centroid(model, transform(calibration))
    threshold = float(np.quantile(calibration_distances, args.quantile, method="lower"))
    event_tag = "_".join(event_ids)
    output = args.output_dir / (
        f"farm_{args.farm}_asset_{next(iter(assets[event_ids[0]]))}_events_{event_tag}"
    )
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "features": usable, "imputer": imputer, "scaler": scaler,
        "model": model, "threshold": threshold,
    }, output / "model.joblib")

    info_path = reference_path.parent.parent / "event_info.csv"
    info = pd.read_csv(info_path, sep=";").set_index("event_id")
    results = {}
    figure, axes = plt.subplots(len(event_ids), 1, figsize=(11, 3.5 * len(event_ids)), squeeze=False)
    for row_number, event_id in enumerate(event_ids):
        prediction = predictions[event_id].sort_values("time_stamp").copy()
        x_prediction = transform(prediction)
        prediction["cluster"] = model.predict(x_prediction)
        prediction["distance_to_centroid"] = distance_to_centroid(model, x_prediction)
        prediction["flagged"] = prediction["distance_to_centroid"].gt(threshold)
        event_info = info.loc[int(event_id)]
        start = pd.to_datetime(event_info["event_start"])
        end = pd.to_datetime(event_info["event_end"])
        before = prediction["time_stamp"].lt(start)
        during = prediction["time_stamp"].between(start, end)
        after = prediction["time_stamp"].gt(end)
        results[event_id] = {
            "label": str(event_info["event_label"]),
            "event_start": start.isoformat(), "event_end": end.isoformat(),
            "all": describe_segment(prediction, pd.Series(True, index=prediction.index)),
            "before": describe_segment(prediction, before),
            "during": describe_segment(prediction, during),
            "after": describe_segment(prediction, after),
            "status_0": describe_segment(prediction, prediction["status_type_id"].eq(0)),
        }
        prediction[[
            "time_stamp", "asset_id", "id", "status_type_id", "cluster",
            "distance_to_centroid", "flagged",
        ]].to_csv(output / f"event_{event_id}_scores.csv", index=False)
        axis = axes[row_number, 0]
        axis.plot(prediction["time_stamp"], prediction["distance_to_centroid"], lw=0.8)
        axis.axhline(threshold, color="red", linestyle="--", label="limiar da calibração")
        axis.axvspan(start, end, color="orange", alpha=0.15, label="intervalo do evento")
        axis.set_title(f"Evento {event_id}: {event_info['event_label']}")
        axis.set_ylabel("Distância ao centroide")
        axis.legend(loc="upper right")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output / "scores_timeline.png", dpi=160)
    plt.close(figure)

    report = {
        "farm": args.farm, "asset_id": next(iter(assets[event_ids[0]])),
        "reference_event": args.reference_event, "evaluation_events": event_ids,
        "earliest_prediction": cutoff.isoformat(),
        "fit_rows_available": len(fit_period), "fit_rows_used": len(fit_sample),
        "fit_end": fit_period["time_stamp"].max().isoformat(),
        "calibration_start": calibration_start.isoformat(),
        "calibration_end": calibration["time_stamp"].max().isoformat(),
        "calibration_rows": len(calibration), "gap_days": args.gap_days,
        "quantile": args.quantile, "threshold": threshold,
        "calibration_flag_rate": float(np.mean(calibration_distances > threshold)),
        "features": usable, "k_diagnostics": diagnostics, "selected_k": best_k,
        "events": results,
        "limitation": "Experimento exploratório; alertas não comprovam diagnóstico ou antecipação de falha.",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modelo único: fazenda {args.farm}, turbina {report['asset_id']}, k={best_k}")
    print(f"Treino: {len(fit_sample)} linhas; calibração: {len(calibration)}; limiar: {threshold:.3f}")
    for event_id, result in results.items():
        print(f"Evento {event_id} ({result['label']}): {result['all']['flag_rate']:.1%} acima do limiar")
    print(f"Saída: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "outputs/shared_reference")
    parser.add_argument("--farm", choices=["A", "B", "C"], default="A")
    parser.add_argument("--reference-event", default="0")
    parser.add_argument("--events", nargs="+", default=["0", "24"])
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--max-fit-rows", type=int, default=20000)
    parser.add_argument("--max-missing", type=float, default=0.3)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--silhouette-sample", type=int, default=3000)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fit_fraction < 1 or not 0 < args.quantile < 1:
        parser.error("fit-fraction e quantile devem estar entre 0 e 1.")
    if args.gap_days < 0 or args.max_fit_rows < 10 or args.silhouette_sample < 2:
        parser.error("gap-days deve ser >= 0; amostras devem ter pelo menos 10/2 linhas.")
    if not args.k_values or min(args.k_values) < 2 or not 0 <= args.max_missing < 1:
        parser.error("Use k >= 2 e max-missing entre 0 e 1.")
    try:
        run(args)
    except ValueError as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
