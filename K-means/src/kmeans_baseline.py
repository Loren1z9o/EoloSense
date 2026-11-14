"""Exploratory K-means baseline for one CARE to Compare event CSV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


EXCLUDE = {"id", "asset_id", "time_stamp", "train_test", "status_type_id", "status_type"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]


def event_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.csv")
        if p.parent.name.lower() == "datasets"
        and re.search(r"wind farm [abc]", str(p), re.IGNORECASE)
    )


def farm_name(path: Path) -> str:
    for part in path.parts:
        match = re.fullmatch(r"Wind Farm ([ABC])", part, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "?"


def choose_file(root: Path, farm: str, event_id: str | None) -> Path:
    files = [p for p in event_files(root) if farm_name(p) == farm]
    if not files:
        raise ValueError(f"Nenhum CSV de eventos da fazenda {farm} em {root}. Extraia o ZIP primeiro.")
    if event_id is None:
        return files[0]
    for path in files:
        if path.stem == event_id:
            return path
    raise ValueError(f"Evento {event_id} não encontrado na fazenda {farm}.")


def split_rows(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "train_test" not in frame:
        raise ValueError("Coluna train_test ausente.")
    values = frame["train_test"].astype(str).str.strip().str.lower()
    train = values.isin({"train", "training"})
    predict = values.isin({"test", "prediction", "predict"})
    unknown = sorted(set(values[~(train | predict)]))
    if unknown:
        raise ValueError(f"Valores desconhecidos em train_test: {unknown}")
    return train, predict


def run(args: argparse.Namespace) -> None:
    path = choose_file(args.data_dir, args.farm, args.event_id)
    frame = pd.read_csv(path, sep=";", low_memory=False)
    train_mask, prediction_mask = split_rows(frame)
    normal_mask = train_mask.copy()
    if "status_type_id" in frame:
        normal_mask &= pd.to_numeric(frame["status_type_id"], errors="coerce").eq(0)
    elif "status_type" in frame:
        normal_mask &= pd.to_numeric(frame["status_type"], errors="coerce").eq(0)
    if not normal_mask.any():
        raise ValueError("Nenhuma linha de treino em operação normal (status 0).")

    feature_names = [
        col for col in frame
        if col.lower().endswith("_avg") and col.lower() not in EXCLUDE
    ]
    descriptions_path = path.parent.parent / "feature_description.csv"
    excluded_by_description: list[str] = []
    if descriptions_path.exists():
        descriptions = pd.read_csv(descriptions_path, sep=";")
        required = {"sensor_name", "is_angle", "is_counter"}
        if required.issubset(descriptions.columns):
            special_sensors = set(
                descriptions.loc[
                    descriptions["is_angle"].astype(str).str.lower().eq("true")
                    | descriptions["is_counter"].astype(str).str.lower().eq("true"),
                    "sensor_name",
                ].astype(str)
            )
            excluded_by_description = [
                col for col in feature_names if col.removesuffix("_avg") in special_sensors
            ]
            feature_names = [col for col in feature_names if col not in excluded_by_description]
    if not feature_names:
        raise ValueError("Nenhuma coluna de sensor terminada em _avg foi encontrada.")
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    fit_pool = numeric.loc[normal_mask]
    missing = fit_pool.isna().mean()
    varying = fit_pool.nunique(dropna=True) > 1
    chosen = missing.index[(missing <= args.max_missing) & varying].tolist()
    if len(chosen) < 2:
        raise ValueError("Menos de 2 sensores úteis após filtrar ausências e constantes.")

    rng = np.random.default_rng(args.seed)
    fit_indices = fit_pool.index.to_numpy()
    if len(fit_indices) > args.max_fit_rows:
        fit_indices = np.sort(rng.choice(fit_indices, args.max_fit_rows, replace=False))
    sampled = numeric.loc[fit_indices, chosen]
    chosen = sampled.columns[sampled.nunique(dropna=True) > 1].tolist()
    if len(chosen) < 2:
        raise ValueError("A amostra de ajuste contém menos de 2 sensores variáveis.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(imputer.fit_transform(numeric.loc[fit_indices, chosen]))
    if len(x_fit) < max(args.k_values) + 1:
        raise ValueError("Poucas linhas de treino para os valores de k solicitados.")

    diagnostics = []
    for k in args.k_values:
        model = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = model.fit_predict(x_fit)
        sample_size = min(args.silhouette_sample, len(x_fit))
        score = silhouette_score(
            x_fit, labels, sample_size=sample_size, random_state=args.seed
        ) if len(set(labels)) > 1 else None
        diagnostics.append({
            "k": k, "inertia": float(model.inertia_),
            "silhouette": None if score is None else float(score),
        })
    valid = [item for item in diagnostics if item["silhouette"] is not None]
    if not valid:
        raise ValueError("K-means produziu apenas um grupo nos valores de k avaliados.")
    best = max(valid, key=lambda item: item["silhouette"])
    model = KMeans(n_clusters=best["k"], random_state=args.seed, n_init=10).fit(x_fit)

    output = args.output_dir / f"farm_{args.farm}_event_{path.stem}"
    output.mkdir(parents=True, exist_ok=True)
    # The projection is fitted on training rows only; prediction rows remain held out.
    projection = PCA(n_components=2, random_state=args.seed).fit(x_fit)
    fit_xy = projection.transform(x_fit)
    fit_labels = model.predict(x_fit)
    plt.figure(figsize=(8, 6))
    plt.scatter(fit_xy[:, 0], fit_xy[:, 1], c=fit_labels, s=3, cmap="tab10", alpha=0.5)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(f"Fazenda {args.farm}, evento {path.stem}: K-means (k={best['k']})")
    plt.tight_layout()
    plt.savefig(output / "clusters_pca.png", dpi=160)
    plt.close()

    def save_assignments(mask: pd.Series, name: str) -> int:
        subset = frame.loc[mask]
        if subset.empty:
            return 0
        x = scaler.transform(imputer.transform(numeric.loc[mask, chosen]))
        result = pd.DataFrame({
            "source_row": subset.index,
            "cluster": model.predict(x),
            "distance_to_centroid": model.transform(x).min(axis=1),
        })
        for column in ("id", "time_stamp", "asset_id", "status_type_id"):
            if column in subset:
                result[column] = subset[column].to_numpy()
        result.to_csv(output / name, index=False)
        return len(result)

    n_train = save_assignments(normal_mask, "train_clusters.csv")
    n_prediction = save_assignments(prediction_mask, "prediction_clusters.csv")
    train_labels = pd.read_csv(output / "train_clusters.csv", usecols=["cluster"])["cluster"]
    train_values = numeric.loc[normal_mask, chosen].reset_index(drop=True)
    profile = train_values.groupby(train_labels).mean()
    profile.insert(0, "rows", train_labels.value_counts().sort_index())
    profile.to_csv(output / "cluster_profile.csv", index_label="cluster")
    report = {
        "source": str(path), "farm": args.farm, "event_id": path.stem,
        "rows_total": len(frame), "rows_train_normal": n_train,
        "rows_prediction": n_prediction, "rows_used_for_fit": len(x_fit),
        "features": chosen, "excluded_angles_and_counters": excluded_by_description,
        "k_diagnostics": diagnostics, "selected_k": best["k"],
        "pca_explained_variance_ratio": projection.explained_variance_ratio_.tolist(),
        "note": "Silhouette e PCA são exploratórios. Clusters não são rótulos de falha.",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Evento: {path}\nSaída: {output}\nk escolhido: {best['k']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "outputs/kmeans")
    parser.add_argument("--farm", choices=["A", "B", "C"], default="A")
    parser.add_argument("--event-id", help="ID do CSV (sem .csv); padrão: primeiro evento da fazenda")
    parser.add_argument("--list-events", action="store_true")
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--max-fit-rows", type=int, default=20000)
    parser.add_argument("--silhouette-sample", type=int, default=3000)
    parser.add_argument("--max-missing", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.list_events:
        for path in event_files(args.data_dir):
            print(f"{farm_name(path)}  {path.stem}  {path}")
        return
    if not args.k_values or min(args.k_values) < 2 or args.max_fit_rows < 3 or args.silhouette_sample < 2:
        parser.error("Use k >= 2, max-fit-rows >= 3 e silhouette-sample >= 2.")
    if not 0 <= args.max_missing < 1:
        parser.error("max-missing deve estar entre 0 e 1 (exclusivo).")
    try:
        run(args)
    except ValueError as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
