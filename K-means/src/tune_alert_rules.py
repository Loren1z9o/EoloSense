"""Select stricter alarm rules on development turbines, then freeze and evaluate."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_failure_alerts import event_result, metrics
from failure_alerts import apply_alarm_rule, numeric, save_report
from kmeans_baseline import MODEL_ROOT, split_rows


CONFIGS = list(itertools.product([.95, .99], [24, 48], [.25, .5]))
BASELINE = (.95, 24, .25)


def config_id(config):
    q, hours, fraction = config
    return f"q{q}_h{hours}_f{fraction}"


def rank_configurations(rows):
    summary = []
    for config, group in pd.DataFrame(rows).groupby("config"):
        first = group.iloc[0]
        summary.append(dict(config=config, quantile=float(first["quantile"]),
                            hours=int(first.hours), fraction=float(first.fraction), **metrics(group)))
    ranking = pd.DataFrame(summary)
    ranking["eligible_for_selection"] = ranking.sensitivity.ge(.4) & ranking.precision.notna()
    ranking = ranking.sort_values(["eligible_for_selection", "precision", "sensitivity", "specificity", "config"],
                                  ascending=[False, False, False, False, True]).reset_index(drop=True)
    if not ranking.eligible_for_selection.any():
        raise ValueError("Nenhuma configuração atingiu sensibilidade de 40% no desenvolvimento.")
    return ranking


def read_scores(source, entry):
    path = source / f"farm_{entry.farm}_event_{entry.event}" / "predictions.csv"
    scores = pd.read_csv(path, parse_dates=["time_stamp"])
    if scores.eligible.dtype != bool or scores.alarm.dtype != bool:
        raise ValueError(f"Elegibilidade/alerta inválidos em {path}")
    return scores


def metadata(entry):
    info = pd.read_csv(Path(entry.path).parent.parent / "event_info.csv", sep=";")
    return info.loc[info.event_id.eq(int(entry.event))].iloc[0]


def prepare_reference(source, entry, quantiles=(.95, .99)):
    path = Path(entry.path)
    stat = path.stat()
    if stat.st_size != int(entry.source_bytes) or stat.st_mtime_ns != int(entry.source_mtime_ns):
        raise ValueError(f"O dataset mudou desde a avaliação base: {path}")
    directory = source / f"farm_{entry.farm}_event_{entry.event}"
    bundle = joblib.load(directory / "model.joblib")
    if bundle["farm"] != entry.farm or bundle["asset_id"] != entry.asset:
        raise ValueError("Identidade do modelo não corresponde ao manifesto.")
    columns = bundle["features"] + ["time_stamp", "train_test", "status_type_id"]
    frame = pd.read_csv(path, sep=";", usecols=columns)
    frame.time_stamp = pd.to_datetime(frame.time_stamp)
    train_mask, _ = split_rows(frame)
    cal = frame.loc[train_mask & frame.status_type_id.eq(0)
                    & frame.time_stamp.ge(pd.Timestamp(bundle["calibration_start"]))]
    values = numeric(cal, bundle["features"])
    values = values.loc[values.isna().mean(axis=1) <= bundle["max_missing"]]
    if len(values) != bundle["calibration_rows"]:
        raise ValueError("A calibração não reproduziu a quantidade de linhas da referência.")
    x = bundle["scaler"].transform(bundle["imputer"].transform(values))
    distances = bundle["model"].transform(x).min(axis=1)
    if not np.isclose(np.quantile(distances, .95), bundle["threshold"], rtol=1e-10):
        raise ValueError("O P95 da calibração não reproduz o limiar original.")
    bundle["calibration_thresholds"] = {q: float(np.quantile(distances, q)) for q in quantiles}
    return bundle


def apply_config(bundle, scores, config):
    q, hours, fraction = config
    return apply_alarm_rule(scores, bundle["calibration_thresholds"][q], hours, fraction)


def write_report(output, ranking, results, chosen):
    summary = []
    data = pd.DataFrame(results)
    for (split, config), group in data.groupby(["split", "config"]):
        summary.append(dict(split=split, config=config, farm="ALL", **metrics(group)))
        for farm, subset in group.groupby("farm"):
            summary.append(dict(split=split, config=config, farm=farm, **metrics(subset)))
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "metrics.csv", index=False)
    chosen_id = config_id(chosen)
    held = summary.loc[summary.split.eq("evaluation") & summary.farm.eq("ALL")]
    winner = held.loc[held.config.eq(chosen_id)].iloc[0]
    lines = ["# Teste de alertas mais exigentes", "",
             "Foram reutilizados os 95 K-means da avaliação base. Centros, sensores, imputadores e padronizadores permaneceram fixos. Os limiares P95 e P99 foram reproduzidos/calculados somente na calibração normal de cada arquivo.", "",
             "As oito combinações de P95/P99, 24/48 horas e fração de excedências de 25%/50% foram comparadas apenas nos 65 eventos de desenvolvimento. A regra foi registrada antes da execução: exigir sensibilidade >=40%, maximizar precisão, desempatar por sensibilidade, especificidade e identificador. Esse piso é uma escolha experimental, não um requisito operacional fornecido pelo usuário.", "",
             f"Configuração congelada: **{chosen_id}**, sempre K=32. A escolha foi salva antes de calcular os resultados nos 30 eventos reservados. Somente a escolhida e o baseline foram avaliados nesses eventos.", "",
             "## Desenvolvimento — oito configurações", "",
             "| Configuração | Precisão | Sensibilidade | Especificidade | Alertas verdadeiros | Falsos alertas | Elegível |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in ranking.itertuples():
        precision = f"{row.precision:.1%}" if pd.notna(row.precision) else "indefinida"
        lines.append(f"| {row.config} | {precision} | {row.sensitivity:.1%} | {row.specificity:.1%} | {row.tp} | {row.fp} | {row.eligible_for_selection} |")
    lines += ["", "## Avaliação interna reutilizada — 30 eventos", "",
              "| Configuração | Precisão | Sensibilidade | Especificidade | F1 | Acurácia | VP | FN | VN | FP |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in held.itertuples():
        lines.append(f"| {row.config} | {row.precision:.1%} | {row.sensitivity:.1%} | {row.specificity:.1%} | {row.f1:.1%} | {row.accuracy:.1%} | {row.tp} | {row.fn} | {row.tn} | {row.fp} |")
    lines += ["", f"Na escolhida, foram detectadas {int(winner.tp)}/15 anomalias; {int(winner.fn)} não foram detectadas. Dos 15 normais, {int(winner.fp)} geraram falso alerta e {int(winner.tn)} não geraram alerta.", "",
              "Aumentar a fração de excedências pode eliminar alertas mantendo a mesma janela/limiar. Uma janela maior, porém, pode redistribuir os alertas no tempo e não garante menos falsos positivos por evento. Os resultados medem alertas na janela anotada de anomalia; nos normais, qualquer alerta na previsão conta como falso positivo. Não são métricas CARE oficiais nem previsão da data de falha.", "",
              "Os 30 eventos já foram consultados em experimentos anteriores. Eventos da mesma turbina são correlacionados e os períodos normais têm durações diferentes. O resultado não é validação externa intocada e não comprova generalização. Não se escolheu outra configuração após consultar a avaliação.", "",
              "## Arquivos", "",
              "[protocol.json](protocol.json): grade e regra prévia; [selection_ranking.csv](selection_ranking.csv): ranking; [frozen_config.json](frozen_config.json): escolha anterior à avaliação; [selection_events.csv](selection_events.csv): decisões no desenvolvimento; [events.csv](events.csv): escolhida em 95 eventos e baseline nos 30 reservados; [metrics.csv](metrics.csv): métricas por conjunto/fazenda. Cada pasta de evento contém o modelo escolhido, previsões e relatório.", "",
              "Para reproduzir a seleção, use um diretório novo:", "",
              "```powershell",
              "& ./.venv/Scripts/python.exe K-means/src/tune_alert_rules.py --output-dir K-means/outputs/stricter_alerts_nova",
              "```", ""]
    (output / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(held.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=MODEL_ROOT / "outputs/failure_alerts_evaluation")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "outputs/stricter_alerts")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use um diretório de saída vazio.")
    manifest = pd.read_csv(args.source_dir / "manifest.csv", dtype=str)
    if len(manifest) != 95 or manifest.duplicated(["farm", "event"]).any():
        parser.error("Manifesto inválido.")
    if set(manifest.split) != {"selection", "evaluation"} or manifest.split.eq("evaluation").sum() != 30:
        parser.error("Partição inválida.")
    if manifest.groupby(["farm", "asset"]).split.nunique().max() != 1:
        parser.error("Turbinas compartilhadas entre conjuntos.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    protocol = dict(grid=[dict(quantile=q, hours=h, fraction=f) for q, h, f in CONFIGS],
                    min_development_sensitivity=.4,
                    selection_rule="maximum precision among sensitivity>=.4; ties: sensitivity, specificity, config ID",
                    baseline=config_id(BASELINE), k=32, seed=42, source_dir=str(args.source_dir.resolve()),
                    source_protocol_sha256=hashlib.sha256((args.source_dir / "protocol.json").read_bytes()).hexdigest(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                        Path(__file__), Path(__file__).with_name("failure_alerts.py"), Path(__file__).with_name("evaluate_failure_alerts.py")]},
                    limitation="reused internal evaluation; not untouched external validation")
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    selection = []
    development = manifest.loc[manifest.split.eq("selection")]
    for index, entry in enumerate(development.itertuples(), 1):
        bundle = prepare_reference(args.source_dir, entry)
        scores = read_scores(args.source_dir, entry)
        meta = metadata(entry)
        directory = args.output_dir / f"farm_{entry.farm}_event_{entry.event}"
        directory.mkdir()
        joblib.dump(bundle, directory / "model.joblib")
        for config in CONFIGS:
            predicted = apply_config(bundle, scores, config)
            selection.append(dict(farm=entry.farm, asset=entry.asset, event=entry.event,
                                  config=config_id(config), quantile=config[0], hours=config[1], fraction=config[2],
                                  **event_result(predicted, meta)))
        pd.DataFrame(selection).to_csv(args.output_dir / "selection_events.csv", index=False)
        print(f"selection [{index}/65] {entry.farm}/{entry.event}", flush=True)
    ranking = rank_configurations(selection)
    ranking.to_csv(args.output_dir / "selection_ranking.csv", index=False)
    first = ranking.iloc[0]
    chosen = (float(first["quantile"]), int(first.hours), float(first.fraction))
    (args.output_dir / "frozen_config.json").write_text(json.dumps(dict(
        config=config_id(chosen), quantile=chosen[0], hours=chosen[1], alarm_fraction=chosen[2],
        min_development_sensitivity=.4, development_precision=float(first.precision),
        development_sensitivity=float(first.sensitivity)), indent=2), encoding="utf-8")
    print(f"FROZEN: {config_id(chosen)}", flush=True)
    results = []
    for index, entry in enumerate(manifest.itertuples(), 1):
        directory = args.output_dir / f"farm_{entry.farm}_event_{entry.event}"
        if entry.split == "selection":
            bundle = joblib.load(directory / "model.joblib")
        else:
            bundle = prepare_reference(args.source_dir, entry)
            directory.mkdir()
        scores, meta = read_scores(args.source_dir, entry), metadata(entry)
        configs = list(dict.fromkeys([chosen, BASELINE])) if entry.split == "evaluation" else [chosen]
        for config in configs:
            predicted = apply_config(bundle, scores, config)
            results.append(dict(split=entry.split, farm=entry.farm, asset=entry.asset, event=entry.event,
                                config=config_id(config), **event_result(predicted, meta)))
            if config == chosen:
                chosen_bundle = dict(bundle, threshold=bundle["calibration_thresholds"][chosen[0]],
                                     threshold_quantile=chosen[0], hours=chosen[1], alarm_fraction=chosen[2])
                joblib.dump(chosen_bundle, directory / "model.joblib")
                predicted.to_csv(directory / "predictions.csv", index=False)
                save_report(chosen_bundle, predicted, directory, meta)
        pd.DataFrame(results).to_csv(args.output_dir / "events.csv", index=False)
        print(f"chosen [{index}/95] {entry.farm}/{entry.event}", flush=True)
    write_report(args.output_dir, ranking, results, chosen)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
