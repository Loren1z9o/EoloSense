"""Recall-first CARE experiment with explicit false-notification and activity budgets."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from alert_episodes import group_alerts
from evaluate_failure_alerts import event_result, metrics
from failure_alerts import apply_alarm_rule, save_report
from kmeans_baseline import MODEL_ROOT
from tune_alert_rules import config_id, metadata, prepare_reference, read_scores

QUANTILES = [.90, .925, .95, .975]
CONFIGS = list(itertools.product(QUANTILES, [6, 12, 24], [.1, .25]))
# Grade ampla (2º experimento): inclui regras mais exigentes, que só fazem sentido quando a
# especificidade entra no critério de escolha. Contém a grade padrão inteira.
WIDE_QUANTILES = [.90, .925, .95, .975, .99]
WIDE_CONFIGS = list(itertools.product(WIDE_QUANTILES, [6, 12, 24, 48, 72], [.1, .25, .5]))
GRIDS = {"standard": (QUANTILES, CONFIGS), "wide": (WIDE_QUANTILES, WIDE_CONFIGS)}
BASELINE = (.95, 24, .25)
RECOVERY_HOURS = 6
# sensitivity: maior sensibilidade de avisos dentro do orçamento (1º experimento).
# balanced: maior acurácia balanceada por evento dentro do orçamento (2º experimento). Com o
# critério de sensibilidade, a especificidade não entra na escolha e tende a zero.
CRITERIA = ("sensitivity", "balanced")
SELECTION_TEXT = {
    "sensitivity": "max notification sensitivity within both normal-data budgets; tie F2, fewer false notices, "
                   "less active time, ID; fallback least budget violation, not accepted",
    "balanced": "max event-level balanced accuracy (sensitivity + specificity) / 2 within both normal-data budgets; "
                "tie sensitivity, fewer false notices, less active time, ID; fallback least budget violation, not accepted",
}


def apply_config(bundle, scores, config):
    q, hours, fraction = config
    ruled = apply_alarm_rule(scores, bundle["calibration_thresholds"][q], hours, fraction)
    return group_alerts(ruled, RECOVERY_HOURS)[0]


def measure(scores, meta):
    raw = event_result(scores, meta)
    # Count real notices, not repeated 10-minute rows or a stretched active state.
    notified = event_result(scores.assign(alarm=scores.notification), meta)
    known = scores.eligible & scores.rolling_exceedance_fraction.notna()
    return dict(**notified, signal_detected=raw["detected"],
                signal_alarm_rows=int(scores.alarm.sum()),
                notifications=int(scores.notification.sum()),
                monitoring_days=float(known.sum() / 144),
                monitored_rows=int(known.sum()),
                active_monitored_rows=int((known & scores.incident_active).sum()))


def summarize(data):
    result = metrics(data)
    normal = data.loc[~data.anomaly.astype(bool)]
    positive = data.loc[data.anomaly.astype(bool)]
    days = normal.monitoring_days.sum()
    rows = normal.monitored_rows.sum()
    rate = float(normal.notifications.sum() / days * 30) if days else float("inf")
    duty = float(normal.active_monitored_rows.sum() / rows) if rows else 1.
    tp, fn, fp = result["tp"], result["fn"], result["fp"]
    result.update(false_notifications=int(normal.notifications.sum()),
                  normal_monitoring_days=float(days), false_notifications_per_30_days=rate,
                  normal_incident_active_fraction=duty,
                  total_notifications=int(data.notifications.sum()),
                  signal_sensitivity=float(positive.signal_detected.mean()) if len(positive) else None,
                  f2=5 * tp / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else None)
    return result


def rank_candidates(rows, max_false_per_30_days=2., max_active_fraction=.25, criterion="sensitivity"):
    ranked = []
    for name, group in pd.DataFrame(rows).groupby("config"):
        first = group.iloc[0]
        ranked.append(dict(config=name, quantile=float(first["quantile"]), hours=int(first.hours),
                           fraction=float(first.fraction), **summarize(group)))
    ranking = pd.DataFrame(ranked)
    ranking["within_budget"] = ranking.false_notifications_per_30_days.le(max_false_per_30_days) & ranking.normal_incident_active_fraction.le(max_active_fraction)
    ranking["budget_violation"] = ((ranking.false_notifications_per_30_days / max_false_per_30_days - 1).clip(lower=0)
                                    + (ranking.normal_incident_active_fraction / max_active_fraction - 1).clip(lower=0))
    if criterion == "sensitivity":
        order = ["within_budget", "budget_violation", "sensitivity", "f2"]
    elif criterion == "balanced":
        order = ["within_budget", "budget_violation", "balanced_accuracy", "sensitivity"]
    else:
        raise ValueError(f"Critério desconhecido: {criterion}")
    order += ["false_notifications_per_30_days", "normal_incident_active_fraction", "config"]
    return ranking.sort_values(order, ascending=[False, True, False, False, True, True, True],
                               na_position="last").reset_index(drop=True)


def _pct(value):
    return "indefinida" if value is None or pd.isna(value) else f"{value:.1%}"


def write_results(output, rows, ranking, frozen):
    data, summary = pd.DataFrame(rows), []
    for (split, name), group in data.groupby(["split", "config"]):
        summary.append(dict(split=split, config=name, farm="ALL", **summarize(group)))
        for farm, subset in group.groupby("farm"):
            summary.append(dict(split=split, config=name, farm=farm, **summarize(subset)))
    result = pd.DataFrame(summary)
    result.to_csv(output / "metrics.csv", index=False)
    held = result.loc[result.split.eq("evaluation") & result.farm.eq("ALL")]
    text = ["# Detecção de falhas com controle de notificações", "",
            "Experimento com os K-means K=32 existentes. O pré-processamento e os centros foram reutilizados; apenas os limiares da calibração e as regras causais de alerta foram alterados. Não se usaram rótulos de previsão para recalibrar distâncias.", "",
            "## Critério definido antes dos resultados", "",
            f"Grade: {frozen.get('grid_size', len(CONFIGS))} combinações ({frozen.get('grid', 'standard')}) de percentil × janela × fração de excedências. Recuperação: {RECOVERY_HOURS} horas consecutivas de dados elegíveis, com cobertura de janela, sem sinal de alarme. Lacunas/inelegibilidade interrompem a contagem de recuperação.", "",
            f"Limites experimentais no desenvolvimento: até {frozen['max_false_per_30_days']} notificações falsas por 30 dias monitorados de trechos normais e incidente ativo em até {frozen['max_active_fraction']:.0%} dos registros monitorados normais. Esses limites não são requisitos operacionais fornecidos pelo usuário.", "",
            ("Entre regras dentro dos limites, maximizar a acurácia balanceada por evento ((sensibilidade + especificidade) / 2); desempatar por sensibilidade, menor taxa de avisos falsos, menor tempo ativo e identificador."
             if frozen.get("criterion") == "balanced" else
             "Entre regras dentro dos limites, maximizar a sensibilidade de notificações; desempatar por F2, menor taxa de avisos falsos, menor tempo ativo e identificador.")
            + " Se nenhuma atender, registrar a menor violação dos limites como candidata não aprovada; não relaxar os limites silenciosamente.", "",
            f"Configuração congelada: **{frozen['config']}**. Atendeu aos limites no desenvolvimento: **{frozen['accepted_development']}**. Foi congelada antes de calcular os resultados nos 30 eventos reservados.", "",
            "## Notificações e episódios", "",
            "`alarm` é a indicação por registro de 10 minutos; `notification` é um aviso na abertura de um episódio; `incident_active` acompanha o episódio aberto; `recovered` marca a recuperação. Não há repetição de avisos durante um episódio aberto, nem lembretes periódicos. Isso não confirma a investigação ou a resolução física de uma falha.", "",
            "A sensibilidade principal exige uma nova notificação dentro da janela anotada da anomalia. Uma linha de alarme repetida ou um incidente antigo ainda aberto não conta como nova detecção. Também é reportada a sensibilidade do sinal bruto para comparação com os experimentos anteriores. Avisos anteriores à janela anotada ficam fora da métrica principal, mesmo que pudessem ser úteis na prática.", "",
            "## Avaliação interna reutilizada — 30 eventos", "",
            "| Regra | Sensibilidade de avisos | Sensibilidade do sinal | Especificidade | Precisão de avisos | Acurácia balanceada | VP | FN | Eventos normais com aviso | Avisos falsos/30 dias normais | Tempo ativo normal |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in held.itertuples():
        text.append(f"| {row.config} | {row.sensitivity:.1%} | {row.signal_sensitivity:.1%} | {_pct(row.specificity)} | {_pct(row.precision)} | {_pct(row.balanced_accuracy)} | {row.tp} | {row.fn} | {row.fp} | {row.false_notifications_per_30_days:.2f} | {row.normal_incident_active_fraction:.1%} |")
    text += ["", "## Desenvolvimento — todas as regras (ordem do ranking)", "",
             "| Regra | Sensibilidade de avisos | Especificidade | Precisão | Acurácia balanceada | Avisos falsos/30 dias normais | Tempo ativo normal | Dentro dos limites |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in ranking.itertuples():
        text.append(f"| {row.config} | {row.sensitivity:.1%} | {_pct(row.specificity)} | {_pct(row.precision)} | {_pct(row.balanced_accuracy)} | {row.false_notifications_per_30_days:.2f} | {row.normal_incident_active_fraction:.1%} | {row.within_budget} |")
    text += ["", "## Limites da avaliação", "",
             "As divisões anteriores por turbina foram mantidas (65 eventos de desenvolvimento e 30 de avaliação). Os 30 eventos já foram consultados; isso é avaliação interna reutilizada, não teste externo intocado. Não se escolheu outra regra depois da avaliação. Os limites são aplicados na seleção, não garantias sobre novas turbinas.", "",
             "As taxas por 30 dias usam a soma dos registros elegíveis com cobertura de janela nos arquivos normais, na cadência de 10 minutos. Eventos anonimizados não são unidos por timestamp; a taxa é por exposição de trechos avaliados, não uma garantia de frequência mensal numa frota real. Os episódios iniciam independentes em cada arquivo CARE. Não se supõe que arquivos da mesma turbina tenham eixos temporais contínuos.", "",
             "Campos ausentes e lacunas não são tratados como recuperação nem como operação saudável. Tempo sem dados suficientes não entra na exposição monitorada. Eventos sem cobertura permanecem no denominador das métricas de detecção.", "",
             "## Reprodução e uso", "",
             "```powershell",
             "& ./.venv/Scripts/python.exe K-means/src/tune_detection.py --output-dir K-means/outputs/detection_first_nova",
             "```", "",
             "[protocol.json](protocol.json) registra a regra e hashes; [frozen_config.json](frozen_config.json) registra a candidata; [selection_ranking.csv](selection_ranking.csv) contém o ranking; [events.csv](events.csv) e [metrics.csv](metrics.csv) contêm decisões e métricas. Cada pasta de evento contém modelo, predictions.csv, notifications.csv e report.json. O modelo escolhido é experimental; consulte a aceitação e as métricas antes de adotá-lo.", ""]
    (output / "RESULTADOS.md").write_text("\n".join(text), encoding="utf-8")
    print(held.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=MODEL_ROOT / "outputs/failure_alerts_evaluation")
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "outputs/detection_first")
    parser.add_argument("--max-false-per-30-days", type=float, default=2.)
    parser.add_argument("--max-active-fraction", type=float, default=.25)
    parser.add_argument("--criterion", choices=CRITERIA, default="sensitivity",
                        help="sensitivity (1º experimento) ou balanced (acurácia balanceada por evento)")
    parser.add_argument("--grid", choices=list(GRIDS), default="standard",
                        help="standard (24 regras) ou wide (75 regras, inclui P99, janelas de 48/72 h e fração 50%%)")
    args = parser.parse_args()
    quantiles, grid_configs = GRIDS[args.grid]
    if args.max_false_per_30_days <= 0 or not 0 < args.max_active_fraction < 1:
        parser.error("Os limites devem ser positivos; a fração ativa deve ser menor que 1.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use uma saída vazia.")
    manifest = pd.read_csv(args.source_dir / "manifest.csv", dtype=str)
    if (len(manifest) != 95 or manifest.duplicated(["farm", "event"]).any()
            or set(manifest.split) != {"selection", "evaluation"}
            or manifest.split.eq("evaluation").sum() != 30
            or manifest.groupby(["farm", "asset"]).split.nunique().max() != 1):
        parser.error("Manifesto/partição inválidos.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    protocol = dict(quantiles=quantiles, hours=sorted({c[1] for c in grid_configs}), fractions=sorted({c[2] for c in grid_configs}),
                    grid=args.grid, criterion=args.criterion, recovery_hours=RECOVERY_HOURS,
                    max_false_per_30_days=args.max_false_per_30_days, max_active_fraction=args.max_active_fraction,
                    selection=SELECTION_TEXT[args.criterion],
                    detection="new notification inside annotation; normal event: any notification",
                    source_dir=str(args.source_dir.resolve()),
                    source_protocol_sha256=hashlib.sha256((args.source_dir / "protocol.json").read_bytes()).hexdigest(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                        Path(__file__), Path(__file__).with_name("failure_alerts.py"), Path(__file__).with_name("alert_episodes.py"),
                        Path(__file__).with_name("tune_alert_rules.py"), Path(__file__).with_name("evaluate_failure_alerts.py")]})
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    selection = []
    development = manifest.loc[manifest.split.eq("selection")]
    for i, entry in enumerate(development.itertuples(), 1):
        bundle = prepare_reference(args.source_dir, entry, quantiles)
        scores, meta = read_scores(args.source_dir, entry), metadata(entry)
        directory = args.output_dir / f"farm_{entry.farm}_event_{entry.event}"
        directory.mkdir()
        joblib.dump(bundle, directory / "model.joblib")
        for config in grid_configs:
            predicted = apply_config(bundle, scores, config)
            selection.append(dict(farm=entry.farm, asset=entry.asset, event=entry.event,
                                  config=config_id(config), quantile=config[0], hours=config[1], fraction=config[2],
                                  **measure(predicted, meta)))
        pd.DataFrame(selection).to_csv(args.output_dir / "selection_events.csv", index=False)
        print(f"selection [{i}/65] {entry.farm}/{entry.event}", flush=True)
    ranking = rank_candidates(selection, args.max_false_per_30_days, args.max_active_fraction, args.criterion)
    ranking.to_csv(args.output_dir / "selection_ranking.csv", index=False)
    winner = ranking.iloc[0]
    chosen = (float(winner["quantile"]), int(winner.hours), float(winner.fraction))
    frozen = dict(config=config_id(chosen), quantile=chosen[0], hours=chosen[1], alarm_fraction=chosen[2],
                  recovery_hours=RECOVERY_HOURS, accepted_development=bool(winner.within_budget),
                  max_false_per_30_days=args.max_false_per_30_days, max_active_fraction=args.max_active_fraction,
                  criterion=args.criterion, grid=args.grid, grid_size=len(grid_configs),
                  development_sensitivity=float(winner.sensitivity),
                  development_specificity=None if pd.isna(winner.specificity) else float(winner.specificity),
                  development_balanced_accuracy=None if pd.isna(winner.balanced_accuracy) else float(winner.balanced_accuracy))
    (args.output_dir / "frozen_config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"FROZEN {frozen}", flush=True)
    results = []
    for i, entry in enumerate(manifest.itertuples(), 1):
        directory = args.output_dir / f"farm_{entry.farm}_event_{entry.event}"
        if entry.split == "selection":
            bundle = joblib.load(directory / "model.joblib")
        else:
            bundle = prepare_reference(args.source_dir, entry, quantiles)
            directory.mkdir()
        scores, meta = read_scores(args.source_dir, entry), metadata(entry)
        configs = list(dict.fromkeys([chosen, BASELINE])) if entry.split == "evaluation" else [chosen]
        for config in configs:
            predicted = apply_config(bundle, scores, config)
            results.append(dict(split=entry.split, farm=entry.farm, asset=entry.asset, event=entry.event,
                                config=config_id(config), **measure(predicted, meta)))
            if config == chosen:
                updated = dict(bundle, threshold=bundle["calibration_thresholds"][chosen[0]],
                               threshold_quantile=chosen[0], hours=chosen[1], alarm_fraction=chosen[2],
                               recovery_hours=RECOVERY_HOURS, selection_accepted=bool(winner.within_budget))
                joblib.dump(updated, directory / "model.joblib")
                predicted.to_csv(directory / "predictions.csv", index=False)
                predicted.loc[predicted.notification].to_csv(directory / "notifications.csv", index=False)
                save_report(updated, predicted, directory, meta)
        pd.DataFrame(results).to_csv(args.output_dir / "events.csv", index=False)
        print(f"chosen [{i}/95] {entry.farm}/{entry.event}", flush=True)
    write_results(args.output_dir, results, ranking, frozen)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
