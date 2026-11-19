"""3º experimento: variantes do ESCORE dos modelos de comportamento normal, sem retreinar.

Motivação: no diagnóstico de AUC por evento (Comum/src/diagnostico_auc_eventos.py), o maior escore
de cada evento quase não ordena falha acima de normal (AUC ~0,5). Trocar só a regra de alerta
(2º experimento) não resolve isso. Aqui muda o próprio escore, com hipóteses físicas:

  * sinal:     "abs" = |resíduo| (atual)  ·  "pos" = só resíduo positivo (componente MAIS QUENTE
               que o esperado; falha costuma aquecer, e resfriamento/queda de leitura gera falso aviso)
  * agregação: "max" = maior resíduo entre os alvos (atual)  ·  "top2" = média dos 2 maiores
               (um único sensor ruidoso não basta)
  * tempo:     "1h" = média móvel causal de 1 h (atual)  ·  "med24h" = mediana causal de 24 h
               por cima (falhas evoluem em dias; picos isolados somem)

São 8 variantes. Os modelos (model.joblib) são os da execução de origem (padrão: principal); as
previsões são refeitas com o mesmo pré-processamento. A variante abs/max/1h reproduz o escore
salvo, o que é conferido evento a evento.

Escolha, SÓ nos eventos de desenvolvimento, em duas etapas (para não testar centenas de regras
no mesmo conjunto pequeno):
  1. variante: maior AUC por evento do maior escore / limiar P99 da calibração normal;
  2. regra de alerta para essa variante: maior acurácia balanceada dentro do orçamento
     (5 avisos falsos / 30 dias, 25% de tempo ativo), grade de 45 regras.
A escolha é congelada e só então os eventos reservados são avaliados, uma vez.

Uso (raiz do projeto; precisa do CARE em data/ e dos modelos de outputs/principal):
    python Comum/src/estudo_escore.py --jobs 4
    python Comum/src/estudo_escore.py --models linear ensembles --farms A    # parcial
    python Comum/src/estudo_escore.py --avaliar-apenas                       # refaz só a seleção

Resultados em Comum/outputs/<--run>/ (padrão: escore_variantes).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import nbm
from nbm import PROJECT_ROOT, SMOOTHING, add_engineered, load_event, predict_z, split_event
from treinar_modelos import FAMILIES, event_dir, event_meta, load_manifest, pct

from alert_episodes import group_alerts  # noqa: E402  (K-means/src)
from failure_alerts import apply_alarm_rule  # noqa: E402
from tune_detection import RECOVERY_HOURS, measure, rank_candidates, summarize  # noqa: E402

SIGNS = ("abs", "pos")
AGGREGATIONS = ("max", "top2")
TIMES = ("1h", "med24h")
VARIANTS = [f"{s}_{a}_{t}" for s, a, t in itertools.product(SIGNS, AGGREGATIONS, TIMES)]
CURRENT = "abs_max_1h"
QUANTILES = (.95, .975, .99, .995, .999)
AUC_QUANTILE = .99
RULES = list(itertools.product(QUANTILES, [6, 24, 72], [.1, .25, .5]))
MEDIAN_WINDOW, MEDIAN_MIN_ROWS = "24h", 72


# --- Escores --------------------------------------------------------------------------
def smoothed_residuals(bundle, frame, farm, require_status):
    """Mesmo cálculo de nbm.score_frame até a média móvel de 1 h, devolvendo todos os alvos."""
    observed, predicted, raw = predict_z(bundle, frame)
    residual = (observed - predicted) / bundle["sigma"]
    used = frame[bundle["inputs"] + bundle["targets"]].apply(pd.to_numeric, errors="coerce")
    eligible = (used.isna().mean(axis=1) <= bundle["max_missing"]).to_numpy() & ~np.isnan(raw).all(axis=1)
    if require_status and farm != "A":
        eligible &= pd.to_numeric(frame.status_type_id, errors="coerce").eq(0).to_numpy()
    residual[~eligible] = np.nan
    smooth = (pd.DataFrame(residual, index=pd.DatetimeIndex(frame.time_stamp))
              .rolling(SMOOTHING, min_periods=3).mean().to_numpy())
    return eligible, smooth


def variant_scores(time_stamp, eligible, smooth):
    """Tabela com as 8 variantes por registro (NaN onde não há escore)."""
    out = {"time_stamp": np.asarray(time_stamp)}
    index = pd.DatetimeIndex(time_stamp)
    for sign in SIGNS:
        values = np.abs(smooth) if sign == "abs" else np.clip(smooth, 0, None)
        filled = np.where(np.isnan(values), -np.inf, values)
        ordered = -np.sort(-filled, axis=1)
        best = ordered[:, 0]
        second = ordered[:, 1] if ordered.shape[1] > 1 else ordered[:, 0]
        second = np.where(np.isfinite(second), second, best)
        for aggregation, value in (("max", best), ("top2", (best + second) / 2)):
            base = np.where(eligible & np.isfinite(value), value, np.nan)
            out[f"{sign}_{aggregation}_1h"] = base
            daily = pd.Series(base, index=index).rolling(MEDIAN_WINDOW, min_periods=MEDIAN_MIN_ROWS).median()
            out[f"{sign}_{aggregation}_med24h"] = np.where(eligible, daily.to_numpy(), np.nan)
    table = pd.DataFrame(out)
    table["eligible"] = np.asarray(eligible, dtype=bool)
    return table


def rescore_event(entry, families, source_run, root):
    """Lê o CSV uma vez e refaz os escores das famílias pedidas a partir dos modelos salvos."""
    todo = [f for f in families if not (event_dir(root / f, entry) / "limiares.json").exists()]
    if not todo:
        return []
    log = []
    with threadpool_limits(limits=1):
        frame, _, _ = load_event(Path(entry.path))
        parts = split_event(frame)
        for family in todo:
            source = event_dir(PROJECT_ROOT / FAMILIES[family][0] / "outputs" / source_run, entry)
            output = event_dir(root / family, entry)
            output.mkdir(parents=True, exist_ok=True)
            bundle = joblib.load(source / "model.joblib")
            layout = nbm.Layout(bundle["inputs"], bundle["engineered"], [], {})
            full = add_engineered(frame, layout)
            cal = full.loc[parts["calibration"].index]
            pred = full.loc[parts["prediction"].index]
            cal_table = variant_scores(cal.time_stamp.to_numpy(), *smoothed_residuals(bundle, cal, entry.farm, False))
            table = variant_scores(pred.time_stamp.to_numpy(), *smoothed_residuals(bundle, pred, entry.farm, True))
            thresholds = {}
            for variant in VARIANTS:
                values = cal_table.loc[cal_table.eligible, variant].dropna()
                if len(values) < 50:
                    raise ValueError(f"{family} {entry.farm}/{entry.event}: poucas linhas de calibração em {variant}")
                thresholds[variant] = {str(q): float(np.quantile(values, q)) for q in QUANTILES}
            check = None
            saved = source / "scores.csv"
            if saved.exists():
                old = pd.read_csv(saved).score.to_numpy(float)
                check = bool(len(old) == len(table) and np.allclose(old, table[CURRENT], equal_nan=True))
            table.to_csv(output / "escores.csv", index=False)
            (output / "limiares.json").write_text(json.dumps(dict(
                thresholds=thresholds, reproduz_scores_salvos=check, modelo=str(source / "model.joblib")),
                indent=2), encoding="utf-8")
            log.append(f"{family:9s} {entry.farm}/{entry.event}: ok"
                       + ("" if check is None else f" · escore atual reproduzido: {check}"))
    return log


# --- Seleção e avaliação -------------------------------------------------------------
def rule_id(variant, rule):
    q, hours, fraction = rule
    return f"{variant}|q{q}_h{hours}_f{fraction}"


def apply_rule(table, thresholds, variant, rule):
    q, hours, fraction = rule
    scores = pd.DataFrame(dict(time_stamp=pd.to_datetime(table.time_stamp), score=table[variant],
                               eligible=table.eligible & table[variant].notna()))
    ruled = apply_alarm_rule(scores, thresholds[variant][str(q)], hours, fraction)
    return group_alerts(ruled, RECOVERY_HOURS)[0]


def event_auc(rows, column):
    data = pd.DataFrame(rows)
    data = data.loc[np.isfinite(data[column])]
    if data.anomaly.nunique() < 2:
        return None
    return float(roc_auc_score(data.anomaly, data[column]))


def evaluate_family(family, root, manifest, max_false, max_active):
    cache = {}
    for entry in manifest.itertuples():
        directory = event_dir(root / family, entry)
        table = pd.read_csv(directory / "escores.csv")
        table["eligible"] = table.eligible.astype(str).str.lower().eq("true")
        info = json.loads((directory / "limiares.json").read_text(encoding="utf-8"))
        cache[(entry.farm, entry.event)] = (entry, table, info["thresholds"], event_meta(entry.path, entry.event),
                                            info["reproduz_scores_salvos"])
    reproduced = [c[4] for c in cache.values() if c[4] is not None]
    # Etapa 1: AUC por evento de cada variante, só no desenvolvimento.
    peaks = []
    for entry, table, thresholds, meta, _ in cache.values():
        for variant in VARIANTS:
            values = table.loc[table.eligible, variant].dropna()
            peaks.append(dict(split=entry.split, farm=entry.farm, event=entry.event, variant=variant,
                              anomaly=meta.event_label == "anomaly",
                              peak=float(values.max() / thresholds[variant][str(AUC_QUANTILE)]) if len(values) else np.nan))
    peaks = pd.DataFrame(peaks)
    peaks.to_csv(root / family / "picos_por_evento.csv", index=False)
    development = peaks.loc[peaks.split.eq("selection")]
    variant_table = pd.DataFrame([dict(variant=v, auc_desenvolvimento=event_auc(g.to_dict("records"), "peak"))
                                  for v, g in development.groupby("variant")])
    # Empate favorece o escore atual: só se troca de escore se a variante for estritamente melhor.
    variant_table["nao_atual"] = variant_table.variant.ne(CURRENT)
    variant_table = (variant_table.sort_values(["auc_desenvolvimento", "nao_atual", "variant"],
                                               ascending=[False, True, True], na_position="last")
                     .drop(columns="nao_atual").reset_index(drop=True))
    variant = variant_table.variant.iloc[0]
    # Etapa 2: regra para essa variante, só no desenvolvimento.
    selection = []
    for entry, table, thresholds, meta, _ in cache.values():
        if entry.split != "selection":
            continue
        for rule in RULES:
            selection.append(dict(farm=entry.farm, asset=entry.asset, event=entry.event, config=rule_id(variant, rule),
                                  quantile=rule[0], hours=rule[1], fraction=rule[2],
                                  **measure(apply_rule(table, thresholds, variant, rule), meta)))
    ranking = rank_candidates(selection, max_false, max_active, "balanced")
    winner = ranking.iloc[0]
    rule = (float(winner["quantile"]), int(winner.hours), float(winner.fraction))
    frozen = dict(family=family, variant=variant, config=rule_id(variant, rule), quantile=rule[0], hours=rule[1],
                  alarm_fraction=rule[2], recovery_hours=RECOVERY_HOURS, accepted_development=bool(winner.within_budget),
                  max_false_per_30_days=max_false, max_active_fraction=max_active,
                  development_auc=float(variant_table.auc_desenvolvimento.iloc[0]),
                  development_sensitivity=float(winner.sensitivity),
                  development_specificity=None if pd.isna(winner.specificity) else float(winner.specificity),
                  development_balanced_accuracy=None if pd.isna(winner.balanced_accuracy)
                  else float(winner.balanced_accuracy),
                  saved_scores_reproduced=f"{sum(reproduced)}/{len(reproduced)}")
    (root / family).mkdir(parents=True, exist_ok=True)
    (root / family / "escolha_congelada.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    variant_table.to_csv(root / family / "variantes_desenvolvimento.csv", index=False)
    ranking.to_csv(root / family / "regras_desenvolvimento.csv", index=False)
    # Só agora os reservados: a escolha congelada e, como referência, o escore atual com a mesma etapa 2.
    reference_rule = rule_for(cache, CURRENT, max_false, max_active)
    results = []
    for entry, table, thresholds, meta, _ in cache.values():
        if entry.split != "evaluation":
            continue
        for name, (v, r) in (("escolhida", (variant, rule)), ("escore atual", (CURRENT, reference_rule))):
            predicted = apply_rule(table, thresholds, v, r)
            results.append(dict(papel=name, config=rule_id(v, r), farm=entry.farm, asset=entry.asset,
                                event=entry.event, label=meta.event_label, **measure(predicted, meta)))
            if name == "escolhida":
                predicted.to_csv(event_dir(root / family, entry) / "predictions.csv", index=False)
    events = pd.DataFrame(results)
    events.to_csv(root / family / "eventos_reservados.csv", index=False)
    held = {name: dict(config=g.config.iloc[0], **summarize(g)) for name, g in events.groupby("papel")}
    held_peaks = peaks.loc[peaks.split.eq("evaluation")]
    for name, v in (("escolhida", variant), ("escore atual", CURRENT)):
        held[name]["auc"] = event_auc(held_peaks.loc[held_peaks.variant.eq(v)].to_dict("records"), "peak")
    return frozen, variant_table, held


def rule_for(cache, variant, max_false, max_active):
    rows = []
    for entry, table, thresholds, meta, _ in cache.values():
        if entry.split == "selection":
            for rule in RULES:
                rows.append(dict(config=rule_id(variant, rule), quantile=rule[0], hours=rule[1], fraction=rule[2],
                                 **measure(apply_rule(table, thresholds, variant, rule), meta)))
    best = rank_candidates(rows, max_false, max_active, "balanced").iloc[0]
    return float(best["quantile"]), int(best.hours), float(best.fraction)


def write_report(root, outcomes, farms, source_run):
    lines = ["# 3º experimento — variantes do escore (sem retreino)", "",
             f"Gerado por `Comum/src/estudo_escore.py` em {time.strftime('%d/%m/%Y %H:%M')}. Fazendas: {farms}. "
             f"Modelos de `outputs/{source_run}`; previsões refeitas, nenhum modelo retreinado.", "",
             "Escolha só no desenvolvimento: (1) variante com maior AUC por evento do pico do escore / P99 da "
             "calibração; (2) regra de alerta com maior acurácia balanceada dentro do orçamento. Congelada antes "
             "dos reservados. A linha \"escore atual\" passa pela mesma etapa 2 com o escore de hoje (abs_max_1h), "
             "para isolar o efeito da troca de escore.", "",
             "## Eventos reservados", "",
             "| Modelo | Escore | Regra | AUC por evento | Sensibilidade | Especificidade | Precisão | "
             "Acurácia balanceada | VP | FN | VN | FP | Avisos falsos / 30 dias |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = []
    for family, (frozen, variants, held) in outcomes.items():
        for name in ("escore atual", "escolhida"):
            h = held[name]
            variant, rule = h["config"].split("|")
            auc = "—" if h["auc"] is None else f"{h['auc']:.3f}"
            label = FAMILIES[family][1] + (" · referência" if name != "escolhida" else " · **escolhida**")
            lines.append(f"| {label} | {variant} | {rule} | {auc} | "
                         f"{pct(h['sensitivity'])} | {pct(h['specificity'])} | {pct(h['precision'])} | "
                         f"{pct(h['balanced_accuracy'])} | {h['tp']} | {h['fn']} | {h['tn']} | {h['fp']} | "
                         f"{h['false_notifications_per_30_days']:.2f} |")
            rows.append(dict(modelo=family, papel=name, escore=variant, regra=rule, **{k: v for k, v in h.items()
                                                                                       if k != "config"}))
    pd.DataFrame(rows).to_csv(root / "comparacao.csv", index=False)
    lines += ["", "Critério experimental do projeto: sensibilidade ≥ 70% e especificidade ≥ 80%.", "",
              "## Desenvolvimento: AUC por evento de cada variante (etapa 1)", "",
              "| Variante | " + " | ".join(FAMILIES[f][1] for f in outcomes) + " |",
              "| --- | " + " | ".join("---:" for _ in outcomes) + " |"]
    for variant in VARIANTS:
        cells = []
        for family, (_, variants, _) in outcomes.items():
            value = variants.set_index("variant").auc_desenvolvimento.get(variant)
            cells.append("—" if value is None or pd.isna(value) else f"{value:.3f}")
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines += ["", "## Desenvolvimento: regra congelada", "",
              "| Modelo | Escolha | Sensibilidade | Especificidade | Acurácia balanceada | Dentro do orçamento | "
              "Escore atual reproduzido |", "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for family, (frozen, _, _) in outcomes.items():
        lines.append(f"| {FAMILIES[family][1]} | {frozen['config'].replace('|', ' · ')} | {pct(frozen['development_sensitivity'])} | "
                     f"{pct(frozen['development_specificity'])} | {pct(frozen['development_balanced_accuracy'])} | "
                     f"{'sim' if frozen['accepted_development'] else 'não'} | {frozen['saved_scores_reproduced']} |")
    lines += ["", "## Como ler", "",
              "- **AUC por evento**: fração dos pares (anomalia, normal) em que o evento com anomalia tem o maior pico "
              "do escore (dividido pelo P99 da sua calibração normal). 0,5 = não ordena; 1 = ordena perfeitamente.",
              "- Variantes: `abs`/`pos` = |resíduo| ou só resíduo positivo (mais quente que o esperado); `max`/`top2` = "
              "maior alvo ou média dos 2 maiores; `1h`/`med24h` = média móvel de 1 h ou, por cima, mediana de 24 h.",
              "- Os reservados já foram consultados em experimentos anteriores (avaliação interna reutilizada). "
              "Nenhuma escolha deste experimento usou esses eventos. Com 30 eventos, cada um vale 3,3 pontos.",
              "- O K-means não entra: o escore dele é distância a protótipos, não resíduo por sensor.", ""]
    (root / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--models", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--farms", default="ABC")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--origem", default="principal", help="Execução com os model.joblib (outputs/<origem>)")
    parser.add_argument("--run", default="escore_variantes", help="Pasta de saída em Comum/outputs/")
    parser.add_argument("--max-false-per-30-days", type=float, default=5.)
    parser.add_argument("--max-active-fraction", type=float, default=.25)
    parser.add_argument("--avaliar-apenas", action="store_true", help="Não refaz escores; só seleção e avaliação")
    args = parser.parse_args()
    farms = "".join(sorted(set(args.farms.upper())))
    if not farms or set(farms) - set("ABC"):
        parser.error("--farms aceita só A, B e C.")
    root = PROJECT_ROOT / "Comum/outputs" / args.run
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.data_dir, farms)
    for family in args.models:
        origin = PROJECT_ROOT / FAMILIES[family][0] / "outputs" / args.origem
        missing = [f"{e.farm}/{e.event}" for e in manifest.itertuples()
                   if not (event_dir(origin, e) / "model.joblib").exists()]
        if missing:
            parser.error(f"{family}: faltam modelos em {origin} ({missing[:3]}...). Treine antes com treinar_modelos.py.")
    protocol = dict(farms=farms, origem=args.origem, variants=VARIANTS, quantiles=list(QUANTILES),
                    rules=[f"q{q}_h{h}_f{f}" for q, h, f in RULES], auc_quantile=AUC_QUANTILE,
                    max_false_per_30_days=args.max_false_per_30_days, max_active_fraction=args.max_active_fraction,
                    selection="etapa 1: variante pela AUC por evento no desenvolvimento; etapa 2: regra pela acurácia "
                              "balanceada no desenvolvimento, dentro do orçamento; congelada antes dos reservados",
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), Path(nbm.__file__)]})
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.to_csv(root / "manifest.csv", index=False)
    if not args.avaliar_apenas:
        started = time.time()
        tasks = (delayed(rescore_event)(entry, args.models, args.origem, root) for entry in manifest.itertuples())
        for done, lines in enumerate(Parallel(n_jobs=args.jobs, return_as="generator_unordered")(tasks), 1):
            for line in lines:
                print(f"[{done}/{len(manifest)}] {line}", flush=True)
        print(f"Escores refeitos em {(time.time() - started) / 60:.1f} min", flush=True)
    outcomes = {}
    for family in args.models:
        outcomes[family] = evaluate_family(family, root, manifest, args.max_false_per_30_days,
                                           args.max_active_fraction)
        frozen = outcomes[family][0]
        print(f"{family}: escolha congelada {frozen['config']} · AUC dev {frozen['development_auc']:.3f} · "
              f"escore atual reproduzido {frozen['saved_scores_reproduced']}", flush=True)
    write_report(root, outcomes, farms, args.origem)


if __name__ == "__main__":
    main()
