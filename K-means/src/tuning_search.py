"""Grouped configuration selection followed by a frozen evaluation."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from kmeans_baseline import PROJECT_ROOT, MODEL_ROOT, event_files, farm_name, split_rows
from shared_reference import candidate_features

KS = [3, 8, 16, 32, 64, 128]
GRID = list(itertools.product(KS, ["euclidean", "dispersion"], [.95, .975, .99], [6, 12, 24]))
OUT = MODEL_ROOT / "outputs/tuning_search"


def config_id(config):
    k, score, quantile, hours = config
    return f"k{k}_{score}_q{quantile}_h{hours}"


def manifest():
    rows = []
    for path in event_files(PROJECT_ROOT / "data"):
        farm = farm_name(path)
        asset = str(pd.read_csv(path, sep=";", usecols=["asset_id"], nrows=1).asset_id.iloc[0])
        info = pd.read_csv(path.parent.parent / "event_info.csv", sep=";")
        label = info.loc[info.event_id.eq(int(path.stem)), "event_label"].iloc[0]
        rows.append(dict(path=str(path), farm=farm, asset=asset, event=path.stem, label=label))
    data = pd.DataFrame(rows)
    data["split"] = "selection"
    for farm, group in data.groupby("farm"):
        assets = sorted(group.asset.unique(), key=lambda a: hashlib.sha256(f"42/{farm}/{a}".encode()).hexdigest())
        held = assets[:int(np.ceil(.3 * len(assets))) ]
        data.loc[data.farm.eq(farm) & data.asset.isin(held), "split"] = "evaluation"
    return data


def evaluate_event(entry, configs):
    path = Path(entry.path)
    header = pd.read_csv(path, sep=";", nrows=0)
    features = candidate_features(header, path.parent.parent / "feature_description.csv")
    frame = pd.read_csv(path, sep=";", usecols=features + ["time_stamp", "train_test", "status_type_id"])
    frame.time_stamp = pd.to_datetime(frame.time_stamp)
    frame = frame.sort_values("time_stamp")
    train, predict = split_rows(frame)
    healthy = frame.loc[train & frame.status_type_id.eq(0)]
    cal_start = healthy.iloc[int(.8 * len(healthy))].time_stamp
    fit = healthy.loc[healthy.time_stamp < cal_start - pd.Timedelta(1, unit="D")]
    cal = healthy.loc[healthy.time_stamp >= cal_start]
    fit = fit.sample(min(12000, len(fit)), random_state=42)
    numeric = fit[features].replace([np.inf, -np.inf], np.nan)
    columns = numeric.columns[(numeric.isna().mean() <= .3) & (numeric.nunique() > 1)].tolist()
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(numeric[columns]))
    def transform(data):
        return scaler.transform(imputer.transform(data[columns].replace([np.inf, -np.inf], np.nan)))
    pred = frame.loc[predict]
    xc, xp = transform(cal), transform(pred)
    valid = np.ones(len(pred), dtype=bool) if entry.farm == "A" else pred.status_type_id.eq(0).to_numpy()
    info = pd.read_csv(path.parent.parent / "event_info.csv", sep=";")
    meta = info.loc[info.event_id.eq(int(entry.event))].iloc[0]
    start, end = pd.Timestamp(meta.event_start), pd.Timestamp(meta.event_end)
    inside = pred.time_stamp.between(start, end).to_numpy()
    early = pred.time_stamp.between(start, start + (end-start)/2).to_numpy()
    anomaly = entry.label == "anomaly"
    rows = []
    for k in sorted({c[0] for c in configs}):
        model = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42,
                               batch_size=1024, max_iter=100).fit(x)
        fit_labels = model.predict(x)
        global_sd = np.maximum(x.std(axis=0), 1e-6)
        spreads = np.stack([
            x[fit_labels == cluster].std(axis=0) if (fit_labels == cluster).sum() >= 20 else global_sd
            for cluster in range(k)
        ])
        spreads = np.maximum(spreads, .1 * global_sd)
        def scores(values, mode):
            labels = model.predict(values)
            residual = values - model.cluster_centers_[labels]
            if mode == "dispersion":
                residual = residual / spreads[labels]
            return np.sqrt(np.sum(residual * residual, axis=1))
        for mode in sorted({c[1] for c in configs if c[0] == k}):
            cs, ps = scores(xc, mode), scores(xp, mode)
            for config in [c for c in configs if c[:2] == (k, mode)]:
                _, _, quantile, hours = config
                threshold = float(np.quantile(cs, quantile))
                signal = pd.Series(np.where(valid, (ps > threshold).astype(float), np.nan),
                                   index=pd.DatetimeIndex(pred.time_stamp))
                alarm = (signal.rolling(f"{hours}h", min_periods=hours*3).mean().to_numpy() >= .25) & valid
                window = inside if anomaly else np.ones(len(pred), dtype=bool)
                rows.append(dict(farm=entry.farm, asset=entry.asset, event=entry.event,
                                 anomaly=anomaly, config=config_id(config), k=k, score=mode,
                                 quantile=quantile, hours=hours, detected=bool((alarm & window).any()),
                                 early_detected=bool((alarm & early).any()) if anomaly else False,
                                 threshold=threshold))
    return rows


def metrics(data):
    summaries = []
    for name, group in data.groupby("config"):
        positive, negative = group.loc[group.anomaly], group.loc[~group.anomaly]
        sensitivity, specificity = positive.detected.mean(), 1-negative.detected.mean()
        first = group.iloc[0]
        summaries.append(dict(config=name, k=int(first.k), score=first.score,
                              quantile=float(first["quantile"]), hours=int(first.hours),
                              sensitivity=float(sensitivity), specificity=float(specificity),
                              balanced_accuracy=float((sensitivity+specificity)/2),
                              tp=int(positive.detected.sum()), fn=int((~positive.detected).sum()),
                              tn=int((~negative.detected).sum()), fp=int(negative.detected.sum()),
                              accepted=bool(sensitivity >= .7 and specificity >= .8)))
    return pd.DataFrame(summaries)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = manifest()
    split.to_csv(OUT / "manifest.csv", index=False)
    protocol = dict(k=KS, scores=["euclidean", "dispersion"], quantiles=[.95,.975,.99],
                    hours=[6,12,24], alarm_fraction=.25, seed=42,
                    split="30% of turbine identities per farm, deterministic SHA256; evaluation groups excluded from selection",
                    selection_rule="specificity >= .8 then maximum sensitivity, specificity, lower k; fallback maximum balanced accuracy",
                    limitation="All farms were previously explored; this is a grouped internal evaluation, not untouched external validation.")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    checkpoint = OUT / "selection_events.csv"
    selection = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    for entry in split.loc[split.split.eq("selection")].itertuples():
        completed = [r for r in selection if r["farm"] == entry.farm and str(r["event"]) == str(entry.event)]
        if len(completed) == len(GRID):
            continue
        selection.extend(evaluate_event(entry, GRID))
        pd.DataFrame(selection).to_csv(OUT / "selection_events.csv", index=False)
        print(f"selection {entry.farm}/{entry.event} done", flush=True)
    ranking = metrics(pd.DataFrame(selection))
    ranking.to_csv(OUT / "selection_ranking.csv", index=False)
    feasible = ranking.loc[ranking.specificity >= .8]
    if len(feasible):
        winner = feasible.sort_values(["sensitivity","specificity","k","config"], ascending=[False,False,True,True]).iloc[0]
    else:
        winner = ranking.sort_values(["balanced_accuracy","k","config"], ascending=[False,True,True]).iloc[0]
    frozen = (int(winner.k), str(winner.score), float(winner["quantile"]), int(winner.hours))
    (OUT / "frozen_winner.json").write_text(winner.to_json(indent=2), encoding="utf-8")
    print(f"FROZEN {config_id(frozen)}", flush=True)
    baseline = (32, "euclidean", .95, 24)
    evaluation = []
    for entry in split.loc[split.split.eq("evaluation")].itertuples():
        evaluation.extend(evaluate_event(entry, list(dict.fromkeys([frozen, baseline]))))
        pd.DataFrame(evaluation).to_csv(OUT / "evaluation_events.csv", index=False)
        print(f"evaluation {entry.farm}/{entry.event} done", flush=True)
    final = metrics(pd.DataFrame(evaluation))
    final.to_csv(OUT / "evaluation_metrics.csv", index=False)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
