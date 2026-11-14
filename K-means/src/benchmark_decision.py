"""Fixed exploratory K-means anomaly protocol; no label-based parameter tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from shared_reference import candidate_features
from kmeans_baseline import PROJECT_ROOT, MODEL_ROOT, event_files, farm_name, split_rows


def main():
    output = MODEL_ROOT / "outputs/benchmark_decision"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in event_files(PROJECT_ROOT / "data"):
        farm = farm_name(path)
        header = pd.read_csv(path, sep=";", nrows=0)
        features = candidate_features(header, path.parent.parent / "feature_description.csv")
        columns = features + ["time_stamp", "asset_id", "train_test", "status_type_id"]
        frame = pd.read_csv(path, sep=";", usecols=columns)
        frame["time_stamp"] = pd.to_datetime(frame["time_stamp"])
        frame = frame.sort_values("time_stamp")
        train, prediction = split_rows(frame)
        normal = frame.loc[train & frame.status_type_id.eq(0)]
        split = int(.8 * len(normal))
        cal_start = normal.iloc[split].time_stamp
        fit = normal.loc[normal.time_stamp < cal_start - pd.Timedelta(1, unit="D")]
        cal = normal.loc[normal.time_stamp >= cal_start]
        fit = fit.sample(min(12000, len(fit)), random_state=42)
        numeric = fit[features].replace([np.inf, -np.inf], np.nan)
        selected = numeric.columns[(numeric.isna().mean() <= .3) & (numeric.nunique() > 1)].tolist()
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x = scaler.fit_transform(imputer.fit_transform(numeric[selected]))
        model = MiniBatchKMeans(n_clusters=32, random_state=42, n_init=3,
                               batch_size=1024, max_iter=100).fit(x)
        def score(data):
            values = data[selected].replace([np.inf, -np.inf], np.nan)
            transformed = scaler.transform(imputer.transform(values))
            return model.transform(transformed).min(axis=1)
        threshold = float(np.quantile(score(cal), .95))
        pred = frame.loc[prediction].copy()
        pred["score"] = score(pred)
        pred["exceedance"] = pred.score > threshold
        # A causal 24-hour persistence rule, fixed before inspecting outcomes.
        valid = pd.Series(True, index=pred.index) if farm == "A" else pred.status_type_id.eq(0)
        signal = pd.Series(np.where(valid, pred.exceedance.astype(float), np.nan),
                           index=pd.DatetimeIndex(pred.time_stamp))
        rolling = signal.rolling("24h", min_periods=72).mean()
        pred["alarm"] = (rolling.to_numpy() >= .25) & valid.to_numpy()
        info = pd.read_csv(path.parent.parent / "event_info.csv", sep=";")
        meta = info.loc[info.event_id.eq(int(path.stem))].iloc[0]
        start, end = pd.Timestamp(meta.event_start), pd.Timestamp(meta.event_end)
        inside = pred.time_stamp.between(start, end)
        anomaly = meta.event_label == "anomaly"
        window = inside if anomaly else pd.Series(True, index=pred.index)
        alarms = pred.loc[window & pred.alarm]
        detected = not alarms.empty
        early_window = pred.time_stamp.between(start, start + (end-start)/2)
        rows.append(dict(farm=farm, event_id=path.stem, asset_id=str(frame.asset_id.iloc[0]),
                         anomaly=bool(anomaly), detected=detected,
                         first_half_detected=bool((pred.alarm & early_window).any()) if anomaly else None,
                         threshold=threshold, fit_rows=len(fit), calibration_rows=len(cal),
                         valid_prediction_rows=int(valid.sum()),
                         first_alarm=str(alarms.time_stamp.iloc[0]) if detected else None))
        pred[["time_stamp", "score", "exceedance", "alarm"]].to_csv(
            output / f"farm_{farm}_event_{path.stem}.csv", index=False)
        pd.DataFrame(rows).to_csv(output / "events.csv", index=False)
        print(f"{farm}/{path.stem}: label={meta.event_label}, alarm={detected}", flush=True)
    summaries = {}
    data = pd.DataFrame(rows)
    for farm in ["A", "B", "C", "BC", "ALL"]:
        sub = data if farm == "ALL" else data.loc[data.farm.isin(list(farm))]
        tn, fp, fn, tp = confusion_matrix(sub.anomaly, sub.detected, labels=[False, True]).ravel()
        sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
        summaries[farm] = dict(events=len(sub), tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp),
                              sensitivity=float(sensitivity), specificity=float(specificity),
                              balanced_accuracy=float((sensitivity+specificity)/2),
                              accepted=bool(sensitivity >= .7 and specificity >= .8))
    (output / "metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
