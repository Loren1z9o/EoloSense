"""Train and reuse an experimental CARE K-means anomaly detector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from kmeans_baseline import MODEL_ROOT, PROJECT_ROOT, choose_file, split_rows
from shared_reference import candidate_features
from alert_episodes import group_alerts

PRIMARY_HOURS = 6


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"time_stamp", "asset_id"}
    if required.difference(frame.columns):
        raise ValueError(f"Colunas obrigatórias: {sorted(required)}")
    frame = frame.copy()
    frame["time_stamp"] = pd.to_datetime(frame.time_stamp, errors="coerce")
    if frame.time_stamp.isna().any() or frame.time_stamp.duplicated().any():
        raise ValueError("Timestamps inválidos ou duplicados dentro do arquivo.")
    if frame.asset_id.isna().any() or frame.asset_id.nunique() != 1:
        raise ValueError("Cada arquivo deve conter exatamente uma turbina identificada.")
    frame = frame.sort_values("time_stamp").reset_index(drop=True)
    if frame.time_stamp.diff().dropna().dt.total_seconds().mod(600).ne(0).any():
        raise ValueError("Os registros devem seguir intervalos de 10 minutos (lacunas são permitidas).")
    return frame


def numeric(frame, features):
    return frame[features].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def train(frame: pd.DataFrame, farm: str, features: list[str], *, k=32, seed=42,
          quantile=.95, hours=PRIMARY_HOURS, alarm_fraction=.25, recovery_hours=6):
    if not .5 < quantile < 1 or hours < 1 or int(hours) != hours or not 0 < alarm_fraction <= 1:
        raise ValueError("Use .5 < quantile < 1, horas inteiras >= 1 e 0 < alarm_fraction <= 1.")
    if recovery_hours <= 0 or recovery_hours * 6 != int(recovery_hours * 6):
        raise ValueError("A recuperação deve ser positiva e múltipla de 10 minutos.")
    frame = validate(frame)
    train_mask, pred_mask = split_rows(frame)
    if not pred_mask.any() or not train_mask.any():
        raise ValueError("São necessárias seções train e prediction no arquivo CARE.")
    if frame.loc[train_mask, "time_stamp"].max() >= frame.loc[pred_mask, "time_stamp"].min():
        raise ValueError("O treino deve terminar antes da seção de previsão do arquivo.")
    if "status_type_id" not in frame:
        raise ValueError("O treino exige status_type_id para identificar operação normal.")
    healthy = frame.loc[train_mask & pd.to_numeric(frame.status_type_id, errors="coerce").eq(0)]
    if len(healthy) < 100:
        raise ValueError("São necessárias pelo menos 100 linhas de treino normal.")
    cal_start = healthy.iloc[int(.8 * len(healthy))].time_stamp
    cutoff = cal_start.to_datetime64().astype("datetime64[ns]") - np.timedelta64(1, "D")
    fit = healthy.loc[healthy.time_stamp.to_numpy(dtype="datetime64[ns]") < cutoff]
    calibration = healthy.loc[healthy.time_stamp >= cal_start]
    if len(fit) < k or len(calibration) < 20:
        raise ValueError("Dados insuficientes após separar ajuste, intervalo de 1 dia e calibração.")
    fit = fit.sample(min(12000, len(fit)), random_state=seed)
    values = numeric(fit, features)
    selected = values.columns[(values.isna().mean() <= .3) & (values.nunique() > 1)].tolist()
    if len(selected) < 2:
        raise ValueError("Menos de dois sensores úteis no ajuste.")
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(values[selected]))
    model = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=1024,
                            max_iter=100, random_state=seed).fit(x)
    cal_values = numeric(calibration, selected)
    cal_values = cal_values.loc[cal_values.isna().mean(axis=1) <= .3]
    if len(cal_values) < 20:
        raise ValueError("Poucas linhas de calibração com sensores suficientes.")
    xc = scaler.transform(imputer.transform(cal_values))
    threshold = float(np.quantile(model.transform(xc).min(axis=1), quantile))
    return dict(version=1, farm=farm, asset_id=str(frame.asset_id.iloc[0]),
                features=selected, imputer=imputer, scaler=scaler, model=model,
                threshold=threshold, threshold_quantile=quantile, hours=hours,
                alarm_fraction=alarm_fraction, max_missing=.3, recovery_hours=recovery_hours,
                training_end=str(frame.loc[train_mask, "time_stamp"].max()),
                fit_rows=len(fit), calibration_rows=len(cal_values), seed=seed,
                calibration_start=str(cal_start), fit_end=str(fit.time_stamp.max()))


def score_rows(bundle: dict, frame: pd.DataFrame, farm: str) -> pd.DataFrame:
    frame = validate(frame)
    if farm != bundle["farm"] or str(frame.asset_id.iloc[0]) != bundle["asset_id"]:
        raise ValueError("O modelo só pode ser usado na mesma fazenda e turbina de treino.")
    # CARE files are anonymized separately; do not transfer references across events.
    if frame.time_stamp.min() <= pd.Timestamp(bundle["training_end"]):
        raise ValueError("A previsão deve ser posterior ao treino, no mesmo eixo temporal.")
    missing = set(bundle["features"]).difference(frame.columns)
    if missing:
        raise ValueError(f"Sensores ausentes no CSV: {sorted(missing)}")
    values = numeric(frame, bundle["features"])
    eligible = values.isna().mean(axis=1) <= bundle["max_missing"]
    if farm != "A":
        if "status_type_id" not in frame:
            raise ValueError("Previsão de B/C exige status_type_id.")
        eligible &= pd.to_numeric(frame.status_type_id, errors="coerce").eq(0)
    columns = [c for c in ("time_stamp", "asset_id", "id") if c in frame]
    result = frame[columns].copy()
    result["eligible"] = eligible
    result["cluster"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    result["score"] = np.nan
    if eligible.any():
        x = bundle["scaler"].transform(bundle["imputer"].transform(values.loc[eligible]))
        result.loc[eligible, "cluster"] = bundle["model"].predict(x)
        result.loc[eligible, "score"] = bundle["model"].transform(x).min(axis=1)
    return result


def predict_stream(bundle, frame, farm, state=None):
    """Carry the rolling context and incident lifecycle across nonoverlapping blocks."""
    fresh = score_rows(bundle, frame, farm)
    state = state or {}
    context = pd.DataFrame(state.get("context", []), columns=["time_stamp", "eligible", "score"])
    if len(context):
        context.time_stamp = pd.to_datetime(context.time_stamp)
        if context.time_stamp.max() >= fresh.time_stamp.min():
            raise ValueError("O estado exige blocos novos, sem repetir registros anteriores.")
        values = pd.concat([context, fresh[["time_stamp", "eligible", "score"]]], ignore_index=True)
    else:
        values = fresh[["time_stamp", "eligible", "score"]].copy()
    ruled = apply_alarm_rule(values, bundle["threshold"], bundle["hours"], bundle["alarm_fraction"])
    for column in ["threshold", "exceedance", "rolling_exceedance_fraction", "alarm"]:
        fresh[column] = ruled[column].iloc[-len(fresh):].to_numpy()
    result, episode_state = group_alerts(fresh, bundle.get("recovery_hours", 6), state.get("episode"))
    cutoff_ns = values.time_stamp.max().value - int(bundle["hours"]) * 3_600_000_000_000
    retained = values.loc[values.time_stamp.to_numpy(dtype="datetime64[ns]").astype(np.int64) > cutoff_ns]
    context_rows = [dict(time_stamp=row.time_stamp.isoformat(), eligible=bool(row.eligible),
                         score=float(row.score) if pd.notna(row.score) else None)
                    for row in retained.itertuples()]
    return result, dict(episode=episode_state, context=context_rows)


def predict(bundle: dict, frame: pd.DataFrame, farm: str) -> pd.DataFrame:
    return predict_stream(bundle, frame, farm)[0]


def apply_alarm_rule(scores, threshold, hours, alarm_fraction):
    """Reuse fixed sensor scores while changing a causal alarm rule."""
    if hours < 1 or int(hours) != hours or not 0 < alarm_fraction <= 1:
        raise ValueError("Horas inteiras >= 1 e 0 < alarm_fraction <= 1 são obrigatórios.")
    result = scores.copy()
    eligible = result.eligible
    result["threshold"] = threshold
    result["exceedance"] = eligible & result.score.gt(threshold)
    signal = pd.Series(np.where(eligible, result.exceedance.astype(float), np.nan),
                       index=pd.DatetimeIndex(result.time_stamp))
    # At 10-minute cadence, half of the expected records must be eligible.
    rolling = signal.rolling(f"{hours}h", min_periods=int(hours) * 3).mean()
    result["rolling_exceedance_fraction"] = rolling.to_numpy()
    result["alarm"] = eligible & result.rolling_exceedance_fraction.ge(alarm_fraction)
    return result


def save_report(bundle, scores, output, metadata=None):
    report = {key: bundle[key] for key in (
        "version", "farm", "asset_id", "features", "threshold", "hours", "alarm_fraction",
        "fit_rows", "calibration_rows", "training_end", "calibration_start", "fit_end", "seed"
    )}
    report.update(k=int(bundle["model"].n_clusters), prediction_rows=len(scores),
                  threshold_quantile=bundle.get("threshold_quantile", .95),
                  eligible_rows=int(scores.eligible.sum()), alarm_rows=int(scores.alarm.sum()),
                  first_alarm=str(scores.loc[scores.alarm, "time_stamp"].min()) if scores.alarm.any() else None,
                  note="Alerta de anomalia experimental; não estima probabilidade, causa ou data de falha.")
    if "notification" in scores:
        report.update(notification_count=int(scores.notification.sum()),
                      recovery_hours=bundle.get("recovery_hours", 6),
                      active_incident_rows=int(scores.incident_active.sum()),
                      selection_accepted=bundle.get("selection_accepted"))
    if metadata is not None:
        anomaly = metadata.event_label == "anomaly"
        window = scores.time_stamp.between(pd.Timestamp(metadata.event_start), pd.Timestamp(metadata.event_end))
        report["event_evaluation"] = dict(
            label=metadata.event_label,
            detected_in_annotated_window=bool((scores.alarm & window).any()) if anomaly else None,
            false_alarm_in_prediction=bool(scores.alarm.any()) if not anomaly else None,
            note="Avaliação de um evento; não é a métrica CARE oficial nem validação de generalização.",
        )
        if "notification" in scores:
            report["event_evaluation"]["new_notification_in_annotated_window"] = bool((scores.notification & window).any()) if anomaly else None
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("train", help="Ajustar em um evento CARE e avaliar sua previsão")
    fit_parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    fit_parser.add_argument("--farm", choices=list("ABC"), required=True)
    fit_parser.add_argument("--event-id", required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--quantile", type=float, default=.95)
    fit_parser.add_argument("--hours", type=int, default=PRIMARY_HOURS)
    fit_parser.add_argument("--alarm-fraction", type=float, default=.25)
    fit_parser.add_argument("--recovery-hours", type=float, default=6)
    score_parser = commands.add_parser("predict", help="Aplicar modelo salvo em novo CSV com separador ;")
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--csv", type=Path, required=True)
    score_parser.add_argument("--farm", choices=list("ABC"), required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument("--state-file", type=Path, help="Persistir janela e episódios entre blocos novos")
    args = parser.parse_args()
    try:
        metadata = None
        if args.command == "train":
            path = choose_file(args.data_dir, args.farm, args.event_id)
            header = pd.read_csv(path, sep=";", nrows=0)
            features = candidate_features(header, path.parent.parent / "feature_description.csv")
            frame = pd.read_csv(path, sep=";", usecols=features + [
                "time_stamp", "asset_id", "id", "status_type_id", "train_test"])
            bundle = train(frame, args.farm, features, quantile=args.quantile,
                           hours=args.hours, alarm_fraction=args.alarm_fraction, recovery_hours=args.recovery_hours)
            _, prediction = split_rows(frame)
            scores = predict(bundle, frame.loc[prediction], args.farm)
            info = pd.read_csv(path.parent.parent / "event_info.csv", sep=";")
            metadata = info.loc[info.event_id.eq(int(args.event_id))].iloc[0]
        else:
            bundle = joblib.load(args.model)
            frame = pd.read_csv(args.csv, sep=";", low_memory=False)
            if "train_test" in frame:
                _, prediction = split_rows(frame)
                frame = frame.loc[prediction]
            if frame.empty:
                raise ValueError("CSV sem linhas de previsão.")
            model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
            state = None
            if args.state_file and args.state_file.exists():
                state = json.loads(args.state_file.read_text(encoding="utf-8"))
                if state.pop("model_sha256", None) != model_hash:
                    raise ValueError("O arquivo de estado pertence a outro modelo; use outro state-file.")
            scores, next_state = predict_stream(bundle, frame, args.farm, state)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "train":
            joblib.dump(bundle, args.output_dir / "model.joblib")
        scores.to_csv(args.output_dir / "predictions.csv", index=False)
        scores.loc[scores.notification].to_csv(args.output_dir / "notifications.csv", index=False)
        save_report(bundle, scores, args.output_dir, metadata)
        if args.command == "predict" and args.state_file:
            next_state["model_sha256"] = model_hash
            args.state_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.state_file.with_suffix(args.state_file.suffix + ".tmp")
            temporary.write_text(json.dumps(next_state, allow_nan=False, indent=2), encoding="utf-8")
            temporary.replace(args.state_file)
        print(f"Saída: {args.output_dir}\nNovos avisos: {int(scores.notification.sum())}"
              f"\nRegistros em alarme (não são avisos repetidos): {int(scores.alarm.sum())}/{len(scores)}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
