"""Retrain the fixed failure_alerts flow on every CARE event and report event metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from failure_alerts import train, predict, save_report
from kmeans_baseline import PROJECT_ROOT, MODEL_ROOT, choose_file, split_rows
from shared_reference import candidate_features


def event_result(scores, meta):
    """Anomaly detection requires an alarm inside the annotation, inclusive."""
    if meta.event_label not in {"normal", "anomaly"}:
        raise ValueError(f"Rótulo desconhecido: {meta.event_label}")
    anomaly = meta.event_label == "anomaly"
    inside = scores.time_stamp.between(pd.Timestamp(meta.event_start), pd.Timestamp(meta.event_end))
    window = inside if anomaly else pd.Series(True, index=scores.index)
    alarms = scores.loc[window & scores.alarm, "time_stamp"]
    early = inside & scores.time_stamp.le(
        pd.Timestamp(meta.event_start) + (pd.Timestamp(meta.event_end) - pd.Timestamp(meta.event_start)) / 2
    )
    return dict(anomaly=anomaly, detected=bool(len(alarms)),
                prediction_rows=len(scores), eligible_rows=int(scores.eligible.sum()),
                alarm_rows=int(scores.alarm.sum()), window_rows=int(window.sum()),
                eligible_window_rows=int((window & scores.eligible).sum()),
                window_with_rolling_coverage=int((window & scores.eligible & scores.rolling_exceedance_fraction.notna()).sum()),
                prediction_days=float((scores.time_stamp.max() - scores.time_stamp.min()).total_seconds() / 86400),
                first_alarm_in_evaluation_window=str(alarms.iloc[0]) if len(alarms) else None,
                first_half_detected=bool((early & scores.alarm).any()) if anomaly else None)


def metrics(data):
    actual, detected = data.anomaly.astype(bool), data.detected.astype(bool)
    tp, fn = int((actual & detected).sum()), int((actual & ~detected).sum())
    tn, fp = int((~actual & ~detected).sum()), int((~actual & detected).sum())
    def ratio(a, b):
        return a / b if b else None
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return dict(events=len(data), tp=tp, fn=fn, tn=tn, fp=fp,
                precision=ratio(tp, tp + fp), sensitivity=recall, specificity=specificity,
                accuracy=ratio(tp + tn, len(data)), f1=ratio(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                events_without_rolling_coverage=int(data.window_with_rolling_coverage.eq(0).sum()))


def write_summary(data, output):
    rows = []
    for split in ["evaluation", "selection", "ALL"]:
        subset = data if split == "ALL" else data.loc[data.split.eq(split)]
        rows.append(dict(split=split, farm="ALL", **metrics(subset)))
        for farm, group in subset.groupby("farm"):
            rows.append(dict(split=split, farm=farm, **metrics(group)))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    held = summary.loc[summary.split.eq("evaluation") & summary.farm.eq("ALL")].iloc[0]
    text = ["# Avaliação do novo fluxo K-means", "",
            "Cada evento foi retreinado com sua própria seção normal de treino. Os rótulos de previsão foram consultados depois do ajuste e da geração dos alertas.", "",
            "Configuração fixa: MiniBatchKMeans K=32, semente 42, P95 da calibração, persistência de 24h, mínimo de 72 registros elegíveis e fração de excedências de 25%. Sem nova busca de parâmetros.", "",
            "## Resultados por evento", "",
            "O resultado principal usa as mesmas turbinas dos 30 eventos reservados anteriormente. Elas não participam da escolha de parâmetros nesta execução. Esse conjunto já foi consultado em experimentos anteriores; é avaliação interna reutilizada, não teste externo intocado.", "",
            "| Conjunto | Fazenda | Eventos | Precisão | Sensibilidade | Especificidade | F1 | Acurácia | Acurácia balanceada |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def percent(value):
        return "indefinida" if pd.isna(value) else f"{value:.1%}"
    for row in summary.itertuples():
        text.append(f"| {row.split} | {row.farm} | {row.events} | {percent(row.precision)} | {percent(row.sensitivity)} | {percent(row.specificity)} | {percent(row.f1)} | {percent(row.accuracy)} | {percent(row.balanced_accuracy)} |")
    text += ["", "## Matriz de confusão — eventos reservados", "",
             "| Rótulo real | Alerta na janela de avaliação | Sem alerta na janela de avaliação |",
             "| --- | ---: | ---: |",
             f"| Anomalia | {int(held.tp)} (VP) | {int(held.fn)} (FN) |",
             f"| Normal | {int(held.fp)} (FP) | {int(held.tn)} (VN) |", "",
             "Para anomalias, um verdadeiro positivo exige alerta dentro da janela anotada (limites inclusivos). Para eventos normais, qualquer alerta na previsão inteira é falso positivo. Não se avalia previsão da data da falha nem a métrica CARE oficial.", "",
             "A janela anotada descreve a anomalia; alerta nessa janela não comprova antecedência em relação a uma data independente de falha. Os períodos normais têm durações diferentes e eventos da mesma turbina são correlacionados. As métricas por evento não são métricas por registro de 10 minutos.", "",
             f"Eventos reservados sem cobertura mínima da janela móvel: {int(held.events_without_rolling_coverage)}. Eles permanecem no denominador, sem serem excluídos silenciosamente. Ausência de alerta com dados inelegíveis não significa operação saudável.", "",
             "O total de 95 eventos é descritivo e inclui os eventos usados anteriormente para exploração. Não é o resultado de um teste independente.", "",
             "## Reprodução e artefatos", "",
             "Na raiz do projeto, execute com um diretório de saída novo:", "",
             "```powershell",
             "& ./.venv/Scripts/python.exe K-means/src/evaluate_failure_alerts.py --output-dir K-means/outputs/failure_alerts_evaluation_nova",
             "```", "",
             "[protocol.json](protocol.json) registra parâmetros, versões e hashes dos scripts; [manifest.csv](manifest.csv) registra a partição e os arquivos locais; [events.csv](events.csv) contém as decisões por evento; [metrics.csv](metrics.csv) contém as métricas. Cada subdiretório de evento contém modelo, previsões e relatório.", ""]
    (output / "RESULTADOS.md").write_text("\n".join(text), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--manifest", type=Path, default=MODEL_ROOT / "outputs/tuning_search/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "outputs/failure_alerts_evaluation")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use um diretório de saída vazio para não misturar execuções.")
    manifest = pd.read_csv(args.manifest, dtype=str)
    if len(manifest) != 95 or manifest.duplicated(["farm", "event"]).any():
        parser.error("O manifesto deve conter os 95 eventos únicos do CARE.")
    if set(manifest.split) != {"selection", "evaluation"}:
        parser.error("Partições desconhecidas no manifesto.")
    if manifest.groupby(["farm", "asset"]).split.nunique().max() != 1:
        parser.error("Uma turbina aparece em mais de uma partição.")
    if manifest.split.eq("evaluation").sum() != 30:
        parser.error("São esperados os 30 eventos reservados anteriormente.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [choose_file(args.data_dir, entry.farm, entry.event) for entry in manifest.itertuples()]
    manifest["path"] = [str(path.resolve()) for path in paths]
    manifest["source_bytes"] = [path.stat().st_size for path in paths]
    manifest["source_mtime_ns"] = [path.stat().st_mtime_ns for path in paths]
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    code_paths = [Path(__file__), Path(__file__).with_name("failure_alerts.py"),
                  Path(__file__).with_name("shared_reference.py"), Path(__file__).with_name("kmeans_baseline.py")]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                    k=32, seed=42, max_fit_rows=12000, calibration_fraction=.2, gap_days=1,
                    quantile=.95, hours=24, alarm_fraction=.25, min_eligible_records=72,
                    max_missing=.3, training_status=0, prediction_status="A: ignored; B/C: 0",
                    tuning=False, split_source=str(args.manifest.resolve()),
                    evaluation="same previously reserved 30 events; reused internal evaluation",
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths})
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, errors = [], []
    for index, entry in enumerate(manifest.itertuples(), 1):
        path = Path(entry.path)
        try:
            header = pd.read_csv(path, sep=";", nrows=0)
            features = candidate_features(header, path.parent.parent / "feature_description.csv")
            frame = pd.read_csv(path, sep=";", usecols=features + [
                "time_stamp", "asset_id", "id", "status_type_id", "train_test"])
            # Preserve the historical baseline when the main profile changes.
            bundle = train(frame, entry.farm, features, hours=24)
            if bundle["asset_id"] != entry.asset:
                raise ValueError("Identidade da turbina difere do manifesto.")
            _, prediction = split_rows(frame)
            scores = predict(bundle, frame.loc[prediction], entry.farm)
            info = pd.read_csv(path.parent.parent / "event_info.csv", sep=";")
            meta = info.loc[info.event_id.eq(int(entry.event))].iloc[0]
            if meta.event_label != entry.label:
                raise ValueError("Rótulo difere do manifesto anterior.")
            output = args.output_dir / f"farm_{entry.farm}_event_{entry.event}"
            output.mkdir()
            joblib.dump(bundle, output / "model.joblib")
            scores.to_csv(output / "predictions.csv", index=False)
            save_report(bundle, scores, output, meta)
            rows.append(dict(farm=entry.farm, asset=entry.asset, event=entry.event, split=entry.split,
                             **event_result(scores, meta)))
            pd.DataFrame(rows).to_csv(args.output_dir / "events.csv", index=False)
            print(f"[{index}/95] {entry.farm}/{entry.event}: {meta.event_label}, detected={rows[-1]['detected']}", flush=True)
        except (ValueError, KeyError, IndexError) as exc:
            errors.append(dict(farm=entry.farm, event=entry.event, error=str(exc)))
            (args.output_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{index}/95] ERRO {entry.farm}/{entry.event}: {exc}", flush=True)
    if errors:
        parser.exit(1, f"{len(errors)} eventos com erro; métricas finais não publicadas. Consulte errors.json.\n")
    write_summary(pd.DataFrame(rows), args.output_dir)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
