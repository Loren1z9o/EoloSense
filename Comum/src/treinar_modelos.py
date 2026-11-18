"""Treina e avalia Árvore de Decisão, Ensembles, SVM e o baseline linear no CARE to Compare.

Mesmo protocolo do K-means (pasta K-means/), para que os quatro modelos sejam comparáveis:
  * mesma partição por turbina: 65 eventos de desenvolvimento e 30 reservados;
  * cada evento tem seu próprio modelo, treinado só na seção de treino com status 0;
  * limiar = percentil da calibração (últimos 20% do treino normal, após 1 dia de intervalo);
  * mesma regra causal de alerta, mesmos episódios/avisos e mesmas métricas;
  * a regra de alerta (percentil, janela, fração) é escolhida SÓ nos 65 eventos de
    desenvolvimento, com orçamento de avisos falsos, e congelada antes dos 30 reservados.

Uso (na raiz do projeto, Git Bash ou PowerShell):
    python Comum/src/treinar_modelos.py                     # tudo: 4 modelos, 95 eventos
    python Comum/src/treinar_modelos.py --farms A --jobs 4  # rápido: só a fazenda A
    python Comum/src/treinar_modelos.py --models arvore ensembles
    python Comum/src/treinar_modelos.py --compare-only       # refaz só a comparação

2º experimento (sem retreinar): escolhe a regra pela acurácia balanceada, numa grade mais
ampla, reaproveitando os escores salvos de outra execução:
    python Comum/src/treinar_modelos.py --reusar principal --run acuracia_balanceada \
        --criterio acuracia-balanceada --grade ampla --kmeans-run detection_acuracia_balanceada

A execução é retomável: eventos já treinados são pulados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import nbm
from nbm import PROJECT_ROOT, load_event, score_frame, split_rows, train_nbm

from alert_episodes import group_alerts  # noqa: E402  (K-means/src)
from failure_alerts import apply_alarm_rule  # noqa: E402
from kmeans_baseline import choose_file, event_files, farm_name  # noqa: E402
from tune_detection import BASELINE, CONFIGS, GRIDS, RECOVERY_HOURS, measure, rank_candidates, summarize  # noqa: E402
from tune_alert_rules import config_id  # noqa: E402

FAMILIES = {
    "arvore": ("Arvore-de-Decisao", "Árvore de Decisão"),
    "ensembles": ("Ensembles", "Ensembles"),
    "svm": ("SVM", "SVM (SVR)"),
    "linear": ("Regressao-Linear", "Regressão Linear (baseline)"),
}
COLORS = {"K-means": "#2a78d6", "arvore": "#eb6834", "ensembles": "#1baf7a", "svm": "#eda100", "linear": "#e87ba4"}
INK, MUTED, CRITICAL, SURFACE = "#0b0b0b", "#52514e", "#d03b3b", "#fcfcfb"
# Nomes da linha de comando → nomes internos de tune_detection.py
CRITERIOS = {"sensibilidade": "sensitivity", "acuracia-balanceada": "balanced"}
GRADES = {"padrao": "standard", "ampla": "wide"}
SELECAO = {
    "sensitivity": "maior sensibilidade de avisos dentro do orçamento no desenvolvimento; congelada antes dos reservados",
    "balanced": "maior acurácia balanceada por evento ((sensibilidade + especificidade) / 2) dentro do orçamento no "
                "desenvolvimento; congelada antes dos reservados",
}
KMEANS_RUN = "detection_balanced"


# --- Partição -------------------------------------------------------------------------
def build_manifest(data_dir: Path) -> pd.DataFrame:
    """Reproduz a regra de K-means/src/tuning_search.py: 30% das turbinas de cada fazenda (SHA256, semente 42)."""
    rows = []
    for path in event_files(data_dir):
        farm = farm_name(path)
        asset = str(pd.read_csv(path, sep=";", usecols=["asset_id"], nrows=1).asset_id.iloc[0])
        info = pd.read_csv(path.parent.parent / "event_info.csv", sep=";")
        label = info.loc[info.event_id.eq(int(path.stem)), "event_label"].iloc[0]
        rows.append(dict(path=str(path), farm=farm, asset=asset, event=path.stem, label=label))
    if not rows:
        raise SystemExit(f"Nenhum CSV do CARE em {data_dir}. Extraia o ZIP em data/CARE_To_Compare/.")
    data = pd.DataFrame(rows)
    data["split"] = "selection"
    for farm, group in data.groupby("farm"):
        assets = sorted(group.asset.unique(), key=lambda a: hashlib.sha256(f"42/{farm}/{a}".encode()).hexdigest())
        held = assets[:int(np.ceil(.3 * len(assets)))]
        data.loc[data.farm.eq(farm) & data.asset.isin(held), "split"] = "evaluation"
    return data


def load_manifest(data_dir: Path, farms: str) -> pd.DataFrame:
    manifest = build_manifest(data_dir)
    previous = PROJECT_ROOT / "K-means/outputs/tuning_search/manifest.csv"
    if previous.exists():
        old = pd.read_csv(previous, dtype=str)[["farm", "event", "split"]]
        merged = manifest.merge(old, on=["farm", "event"], how="outer", suffixes=("", "_kmeans"), indicator=True)
        if (merged._merge != "both").any() or merged.split.ne(merged.split_kmeans).any():
            raise SystemExit("A partição recalculada difere da usada pelo K-means; confira os dados.")
        print("Partição conferida com K-means/outputs/tuning_search/manifest.csv", flush=True)
    manifest = manifest.loc[manifest.farm.isin(list(farms))].reset_index(drop=True)
    manifest["path"] = [str(choose_file(data_dir, e.farm, e.event)) for e in manifest.itertuples()]
    return manifest


def event_meta(path: Path, event: str):
    info = pd.read_csv(Path(path).parent.parent / "event_info.csv", sep=";")
    return info.loc[info.event_id.eq(int(event))].iloc[0]


# --- Treino por evento ------------------------------------------------------------------
def event_dir(root: Path, entry) -> Path:
    return root / "eventos" / f"farm_{entry.farm}_event_{entry.event}"


def process_event(entry, families, roots):
    """Lê o CSV uma vez e treina as famílias pedidas. Idempotente (retoma execuções)."""
    todo = [f for f in families if not (event_dir(roots[f], entry) / "report.json").exists()]
    if not todo:
        return []
    log = []
    with threadpool_limits(limits=1):
        frame, features, descriptions = load_event(Path(entry.path))
        if str(frame.asset_id.iloc[0]) != entry.asset:
            raise ValueError("Turbina do arquivo difere do manifesto.")
        meta = event_meta(entry.path, entry.event)
        _, prediction = split_rows(frame)
        for family in todo:
            output = event_dir(roots[family], entry)
            output.mkdir(parents=True, exist_ok=True)
            try:
                trained, full = train_nbm(frame, features, descriptions, family, entry.farm, entry.asset)
                bundle = trained.bundle
                scores = score_frame(bundle, full.loc[prediction], entry.farm)
                joblib.dump(bundle, output / "model.joblib", compress=3)
                scores.to_csv(output / "scores.csv", index=False)
                report = dict(farm=entry.farm, event=entry.event, asset=entry.asset, split=entry.split,
                              label=meta.event_label, family=family, chosen=bundle["chosen"],
                              inputs=bundle["inputs"], engineered=[e[0] for e in bundle["engineered"]],
                              targets=bundle["targets"], target_descriptions=bundle["target_descriptions"],
                              target_r2_ridge=bundle["target_r2_ridge"], tuning=bundle["tuning"],
                              pruning=bundle["pruning"], val_z_rmse=bundle["val_z_rmse"],
                              calibration_thresholds={str(k): v for k, v in bundle["calibration_thresholds"].items()},
                              calibration_rows=bundle["calibration_rows"], fit_rows=bundle["fit_rows"],
                              regression=trained.regression, importances=bundle["importances"],
                              seconds=bundle["seconds"], prediction_rows=len(scores),
                              eligible_prediction_rows=int(scores.eligible.sum()))
                (output / "error.json").unlink(missing_ok=True)
                (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                log.append(f"{family:9s} {entry.farm}/{entry.event}: {bundle['chosen']} "
                           f"| R² médio {trained.regression['mean_r2']:.3f} | {bundle['seconds']}s")
            except (ValueError, KeyError, IndexError) as exc:
                (output / "error.json").write_text(json.dumps(dict(error=str(exc), trace=traceback.format_exc()),
                                                              indent=2, ensure_ascii=False), encoding="utf-8")
                log.append(f"{family:9s} {entry.farm}/{entry.event}: ERRO {exc}")
    return log


# --- Regra de alerta, seleção e avaliação --------------------------------------------------
def read_event(root, entry):
    directory = event_dir(root, entry)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(directory / "scores.csv", parse_dates=["time_stamp"])
    scores["eligible"] = scores.eligible.astype(bool)
    thresholds = {float(k): v for k, v in report["calibration_thresholds"].items()}
    return report, scores, thresholds


def apply_config(scores, thresholds, config):
    quantile, hours, fraction = config
    ruled = apply_alarm_rule(scores[["time_stamp", "eligible", "score"]], thresholds[quantile], hours, fraction)
    return group_alerts(ruled, RECOVERY_HOURS)[0]


def select_and_evaluate(family, root, manifest, max_false, max_active, criterion="sensitivity", grid="standard",
                        source=None):
    """Escolhe a regra só no desenvolvimento e avalia nos reservados.

    `source` é a pasta da execução com os escores já calculados (padrão: a própria `root`). Com outra pasta,
    nada é retreinado: só a regra de alerta é refeita e os resultados são gravados em `root`.
    """
    source = root if source is None else source
    configs = GRIDS[grid][1]
    cache = {}
    for entry in manifest.itertuples():
        cache[(entry.farm, entry.event)] = (entry, *read_event(source, entry), event_meta(entry.path, entry.event))
    selection = []
    for entry, report, scores, thresholds, meta in (cache[(e.farm, e.event)] for e in manifest.itertuples()
                                                    if e.split == "selection"):
        missing = {c[0] for c in configs} - set(thresholds)
        if missing:
            raise SystemExit(f"{family} {entry.farm}/{entry.event}: limiares {sorted(missing)} não foram salvos.")
        for config in configs:
            selection.append(dict(farm=entry.farm, asset=entry.asset, event=entry.event,
                                  config=config_id(config), quantile=config[0], hours=config[1], fraction=config[2],
                                  **measure(apply_config(scores, thresholds, config), meta)))
    pd.DataFrame(selection).to_csv(root / "selection_events.csv", index=False)
    ranking = rank_candidates(selection, max_false, max_active, criterion)
    ranking.to_csv(root / "selection_ranking.csv", index=False)
    winner = ranking.iloc[0]
    chosen = (float(winner["quantile"]), int(winner.hours), float(winner.fraction))
    frozen = dict(config=config_id(chosen), quantile=chosen[0], hours=chosen[1], alarm_fraction=chosen[2],
                  recovery_hours=RECOVERY_HOURS, accepted_development=bool(winner.within_budget),
                  max_false_per_30_days=max_false, max_active_fraction=max_active,
                  criterion=criterion, grid=grid, grid_size=len(configs),
                  scores_from=str(source.relative_to(PROJECT_ROOT)) if source.is_relative_to(PROJECT_ROOT) else str(source),
                  development_sensitivity=float(winner.sensitivity),
                  development_specificity=None if pd.isna(winner.specificity) else float(winner.specificity),
                  development_balanced_accuracy=None if pd.isna(winner.balanced_accuracy)
                  else float(winner.balanced_accuracy))
    (root / "frozen_config.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    # Só agora, com a regra congelada, os eventos reservados são consultados.
    results, regression = [], []
    for entry, report, scores, thresholds, meta in cache.values():
        configs = list(dict.fromkeys([chosen, BASELINE])) if entry.split == "evaluation" else [chosen]
        for config in configs:
            predicted = apply_config(scores, thresholds, config)
            results.append(dict(split=entry.split, farm=entry.farm, asset=entry.asset, event=entry.event,
                                config=config_id(config), label=meta.event_label, event_start=str(meta.event_start),
                                event_end=str(meta.event_end), **measure(predicted, meta)))
            if config == chosen:
                event_dir(root, entry).mkdir(parents=True, exist_ok=True)
                predicted.assign(top_sensor=scores.top_sensor, top_residual=scores.top_residual).to_csv(
                    event_dir(root, entry) / "predictions.csv", index=False)
        regression.append(dict(split=entry.split, farm=entry.farm, event=entry.event, label=report["label"],
                               chosen=report["chosen"], targets=len(report["targets"]),
                               val_z_rmse=report["val_z_rmse"], cal_z_rmse=report["regression"]["z_rmse"],
                               mean_mae=report["regression"]["mean_mae"], mean_rmse=report["regression"]["mean_rmse"],
                               mean_r2=report["regression"]["mean_r2"], seconds=report["seconds"]))
    events = pd.DataFrame(results)
    events.to_csv(root / "events.csv", index=False)
    pd.DataFrame(regression).to_csv(root / "regression.csv", index=False)
    summary = []
    for (split, name), group in events.groupby(["split", "config"]):
        summary.append(dict(split=split, config=name, farm="ALL", **summarize(group)))
        for farm, subset in group.groupby("farm"):
            summary.append(dict(split=split, config=name, farm=farm, **summarize(subset)))
    metrics = pd.DataFrame(summary)
    metrics.to_csv(root / "metrics.csv", index=False)
    return frozen, ranking, metrics, events, pd.DataFrame(regression)


# --- Relatórios ---------------------------------------------------------------------------
def pct(value):
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


def chosen_counts(regression):
    names = regression.chosen.str.replace(r" ccp_alpha=.*", "", regex=True)
    return names.str.split(" ").str[0].value_counts().to_dict()


def write_results(family, root, frozen, ranking, metrics, regression, args):
    folder, title = FAMILIES[family]
    held = metrics.loc[metrics.split.eq("evaluation") & metrics.farm.eq("ALL")].set_index("config")
    main = held.loc[frozen["config"]] if frozen["config"] in held.index else None
    text = [f"# {title} — resultados", "",
            f"Gerado por `Comum/src/treinar_modelos.py` em {time.strftime('%d/%m/%Y %H:%M')}. "
            f"Fazendas: {', '.join(sorted(regression.farm.unique()))}; eventos: {len(regression)}.", "",
            "## 1. O modelo prevê bem o comportamento normal?", "",
            "Regressão multi-saída das temperaturas a partir das condições de operação, medida na **calibração** "
            "(últimos 20% do treino normal, dados não vistos no ajuste). R² perto de 1 = o modelo entende a turbina saudável.", "",
            "| Conjunto | Eventos | R² médio (mediana) | MAE médio (mediana) | RMSE padronizado (mediana) |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for split, group in [("Todos", regression), *regression.groupby("split")]:
        text.append(f"| {split} | {len(group)} | {group.mean_r2.median():.3f} | {group.mean_mae.median():.2f} | "
                    f"{group.cal_z_rmse.median():.3f} |")
    text += ["", f"Configuração escolhida na validação interna, por evento: {chosen_counts(regression)}.", "",
             "## 2. O alerta pega a falha a tempo?", "",
             f"Critério de escolha: **{SELECAO[frozen.get('criterion', 'sensitivity')]}** "
             f"({frozen.get('grid_size', len(CONFIGS))} regras testadas).", "",
             f"Regra congelada no desenvolvimento: **{frozen['config']}** (percentil {frozen['quantile']}, janela de "
             f"{frozen['hours']} h, {frozen['alarm_fraction']:.0%} de excedências, recuperação de {frozen['recovery_hours']} h). "
             f"Dentro do orçamento (≤ {frozen['max_false_per_30_days']} avisos falsos por 30 dias e ≤ "
             f"{frozen['max_active_fraction']:.0%} de tempo ativo): **{'sim' if frozen['accepted_development'] else 'não'}**.", "",
             "### Eventos reservados (não usados em nenhuma escolha)", "",
             "| Regra | Sensibilidade (novo aviso na janela) | Sensibilidade do sinal | Especificidade | Precisão | "
             "Acurácia balanceada | VP | FN | VN | FP | Avisos falsos / 30 dias | Tempo ativo normal |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in held.iterrows():
        label = f"**{name}** (congelada)" if name == frozen["config"] else f"{name} (regra-base do protocolo)"
        text.append(f"| {label} | {pct(row.sensitivity)} | {pct(row.signal_sensitivity)} | {pct(row.specificity)} | "
                    f"{pct(row.precision)} | {pct(row.balanced_accuracy)} | {row.tp} | {row.fn} | {row.tn} | {row.fp} | "
                    f"{row.false_notifications_per_30_days:.2f} | {pct(row.normal_incident_active_fraction)} |")
    text += ["", "### Desenvolvimento (5 melhores regras do ranking)", "",
             "| Regra | Sensibilidade | Especificidade | Precisão | Acurácia balanceada | Avisos falsos / 30 dias | "
             "Tempo ativo normal | Dentro do orçamento |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in ranking.head(5).itertuples():
        text.append(f"| {row.config} | {pct(row.sensitivity)} | {pct(row.specificity)} | {pct(row.precision)} | "
                    f"{pct(row.balanced_accuracy)} | {row.false_notifications_per_30_days:.2f} | "
                    f"{pct(row.normal_incident_active_fraction)} | {row.within_budget} |")
    text += ["", "## Como ler", "",
             "- **Sensibilidade**: dos eventos com anomalia, quantos receberam um aviso novo dentro da janela anotada.",
             "- **Especificidade**: dos eventos normais, quantos passaram sem nenhum aviso.",
             "- As métricas são por evento (não por registro de 10 min) e não são o CARE score oficial.",
             "- Os 30 eventos reservados já foram consultados em experimentos do K-means: é avaliação interna "
             "reutilizada, não teste externo intocado. Nenhuma escolha deste modelo usou esses eventos.",
             "- Cada evento tem seu próprio modelo; os timestamps do CARE são anonimizados por arquivo.", "",
             "## Arquivos", "",
             "`metrics.csv` (todas as métricas), `events.csv` (decisão por evento), `regression.csv` (qualidade da "
             "regressão por evento), `selection_ranking.csv`, `frozen_config.json`, `protocol.json` e "
             "`eventos/farm_X_event_N/` com `model.joblib`, `scores.csv`, `predictions.csv` e `report.json` "
             "(alvos, entradas, grade testada, importâncias).", ""]
    if frozen.get("scores_from") and Path(PROJECT_ROOT / frozen["scores_from"]).resolve() != root.resolve():
        text += [f"Esta execução não retreinou nada: `model.joblib`, `scores.csv` e `report.json` estão em "
                 f"`{frozen['scores_from']}/eventos/`. Aqui ficam só a nova regra e o `predictions.csv` de cada evento.", ""]
    (root / "RESULTADOS.md").write_text("\n".join(text), encoding="utf-8")
    return main


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#e8e7e2", linewidth=.6)
    ax.set_axisbelow(True)


def plot_timelines(family, root, manifest, frozen, source=None):
    source = root if source is None else source
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    held = manifest.loc[manifest.split.eq("evaluation")]
    picks = pd.concat([held.loc[held.label.eq("anomaly")].head(3), held.loc[held.label.eq("normal")].head(1)])
    if picks.empty:
        return
    figure, axes = plt.subplots(len(picks), 1, figsize=(10, 2.3 * len(picks)), facecolor=SURFACE, squeeze=False)
    for ax, entry in zip(axes[:, 0], picks.itertuples()):
        report, scores, thresholds = read_event(source, entry)
        meta = event_meta(entry.path, entry.event)
        predicted = apply_config(scores, thresholds, (frozen["quantile"], frozen["hours"], frozen["alarm_fraction"]))
        style(ax)
        if meta.event_label == "anomaly":
            ax.axvspan(pd.Timestamp(meta.event_start), pd.Timestamp(meta.event_end), color="#f0efec", lw=0,
                       label="janela anotada da anomalia")
        ax.plot(scores.time_stamp, scores.score, color=COLORS[family], lw=1, label="escore (maior resíduo suavizado)")
        ax.axhline(thresholds[frozen["quantile"]], color=MUTED, lw=1.2, ls="--", label="limiar da calibração")
        warn = predicted.loc[predicted.notification]
        ax.scatter(warn.time_stamp, scores.loc[warn.index, "score"], color=CRITICAL, s=36, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5, label="aviso emitido")
        top = scores.loc[predicted.alarm, "top_sensor"].mode()
        detail = f" · sensor mais desviado nos alarmes: {top.iloc[0]}" if len(top) else ""
        ax.set_title(f"Fazenda {entry.farm} · evento {entry.event} · {meta.event_label}{detail}",
                     loc="left", fontsize=9, color=INK)
    axes[0, 0].legend(loc="upper left", fontsize=7, frameon=False, ncol=4)
    figure.suptitle(f"{FAMILIES[family][1]}: escore nos eventos reservados (regra {frozen['config']})",
                    x=.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout()
    (root / "figuras").mkdir(exist_ok=True)
    figure.savefig(root / "figuras/linha_do_tempo_reservados.png", dpi=150)
    plt.close(figure)


def kmeans_row(farms, kmeans_run=KMEANS_RUN):
    """Linha do K-means (pasta K-means/outputs/<kmeans_run>) no mesmo escopo de fazendas, se existir."""
    base = PROJECT_ROOT / "K-means/outputs" / kmeans_run
    if not (base / "metrics.csv").exists():
        return None
    frozen = json.loads((base / "frozen_config.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(base / "metrics.csv")
    scope = "ALL" if len(farms) == 3 else (farms if len(farms) == 1 else None)
    if scope is None:
        return None
    row = metrics.loc[metrics.split.eq("evaluation") & metrics.farm.eq(scope) & metrics.config.eq(frozen["config"])]
    if row.empty:
        return None
    values = row.iloc[0].drop(labels=["split", "farm", "config"], errors="ignore").to_dict()
    return dict(modelo="K-means", chave="K-means", config=frozen["config"], r2=None, **values)


def collect_rows(run, farms):
    """Junta todos os modelos já avaliados nesta execução (--run), mesmo que treinados em comandos separados."""
    scope = "ALL" if len(farms) == 3 else farms
    rows = []
    for family, (folder, title) in FAMILIES.items():
        root = PROJECT_ROOT / folder / "outputs" / run
        needed = [root / n for n in ("frozen_config.json", "metrics.csv", "regression.csv", "protocol.json")]
        if not all(p.exists() for p in needed):
            continue
        if json.loads(needed[3].read_text(encoding="utf-8")).get("farms") != farms:
            continue
        frozen = json.loads(needed[0].read_text(encoding="utf-8"))
        metrics, regression = pd.read_csv(needed[1]), pd.read_csv(needed[2])
        row = metrics.loc[metrics.split.eq("evaluation") & metrics.farm.eq(scope) & metrics.config.eq(frozen["config"])]
        if row.empty:
            continue
        values = row.iloc[0].drop(labels=["split", "farm", "config"]).to_dict()
        rows.append(dict(modelo=title, chave=family, config=frozen["config"],
                         r2=float(regression.loc[regression.split.eq("evaluation"), "mean_r2"].median()), **values))
    return rows


def write_comparison(rows, farms, run, kmeans_run=KMEANS_RUN):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = PROJECT_ROOT / "Comum/outputs" / run
    output.mkdir(parents=True, exist_ok=True)
    scope = "ALL" if len(farms) == 3 else farms
    km = kmeans_row(farms, kmeans_run)
    if km is not None:
        rows = [km] + rows
    (output / "comparacao.json").write_text(json.dumps(dict(kmeans_run=kmeans_run if km is not None else None),
                                                       indent=2), encoding="utf-8")
    table = pd.DataFrame(rows)
    keep = ["modelo", "config", "r2", "sensitivity", "signal_sensitivity", "specificity", "precision",
            "balanced_accuracy", "f2", "tp", "fn", "tn", "fp", "false_notifications_per_30_days",
            "normal_incident_active_fraction"]
    table[[c for c in keep if c in table]].to_csv(output / "comparacao.csv", index=False)
    text = ["# Comparação dos modelos — eventos reservados", "",
            f"Fazendas: {scope}. Cada modelo teve sua regra de alerta escolhida apenas no desenvolvimento.", "",
            "| Modelo | R² normal (mediana) | Sensibilidade | Especificidade | Precisão | Acurácia balanceada | "
            "Avisos falsos / 30 dias | Regra |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in table.itertuples():
        r2 = "—" if row.r2 is None or pd.isna(row.r2) else f"{row.r2:.3f}"
        text.append(f"| {row.modelo} | {r2} | {pct(row.sensitivity)} | {pct(row.specificity)} | {pct(row.precision)} | "
                    f"{pct(row.balanced_accuracy)} | {row.false_notifications_per_30_days:.2f} | {row.config} |")
    if km is None:
        text += ["", f"K-means ausente: não há resultados em `K-means/outputs/{kmeans_run}` (veja "
                 "K-means/DETECCAO_E_ALERTAS.md; use `--kmeans-run` para apontar outra pasta)."]
    else:
        text += ["", f"K-means: resultados de `K-means/outputs/{kmeans_run}`."]
    text += ["", "Critério experimental do projeto para aceitar um detector: sensibilidade ≥ 70% e especificidade ≥ 80%.", ""]
    (output / "COMPARACAO.md").write_text("\n".join(text), encoding="utf-8")

    measures = [("sensitivity", "Sensibilidade"), ("specificity", "Especificidade"),
                ("precision", "Precisão"), ("false_notifications_per_30_days", "Avisos falsos / 30 dias")]
    figure, axes = plt.subplots(1, 4, figsize=(12, .55 * len(table) + 1.4), facecolor=SURFACE, sharey=True)
    for ax, (column, label) in zip(axes, measures):
        style(ax)
        ax.grid(axis="y", visible=False)
        values = table[column].astype(float).fillna(0)
        colors = [COLORS.get(k, MUTED) for k in table.chave]
        ax.barh(table.modelo, values, color=colors, height=.6)
        for y, value in enumerate(values):
            ax.text(value, y, f" {value:.0%}" if column != "false_notifications_per_30_days" else f" {value:.1f}",
                    va="center", fontsize=8, color=INK)
        ax.set_title(label, loc="left", fontsize=9, color=INK)
        if column != "false_notifications_per_30_days":
            ax.set_xlim(0, 1.15)
        ax.invert_yaxis() if ax is axes[0] else None
    figure.suptitle(f"Modelos nos eventos reservados ({scope})", x=.01, ha="left", fontsize=11, color=INK)
    figure.tight_layout()
    figure.savefig(output / "comparacao.png", dpi=150)
    plt.close(figure)
    print("\n".join(text), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--models", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--farms", default="ABC", help="Ex.: A, AB ou ABC (padrão)")
    parser.add_argument("--jobs", type=int, default=1, help="Eventos em paralelo (1 processo por evento)")
    parser.add_argument("--run", default="principal", help="Nome da execução (subpasta de outputs)")
    parser.add_argument("--max-false-per-30-days", type=float, default=5.)
    parser.add_argument("--max-active-fraction", type=float, default=.25)
    parser.add_argument("--criterio", choices=list(CRITERIOS), default="sensibilidade",
                        help="Como escolher a regra de alerta no desenvolvimento: sensibilidade (1º experimento) ou "
                             "acuracia-balanceada (sensibilidade e especificidade juntas)")
    parser.add_argument("--grade", choices=list(GRADES), default="padrao",
                        help="padrao (24 regras) ou ampla (75 regras: inclui P99, janelas de 48/72 h e fração 50%%)")
    parser.add_argument("--reusar", metavar="RUN",
                        help="Não treina: reaproveita modelos e escores de outputs/RUN e refaz só a regra de alerta, "
                             "gravando em outputs/<--run>")
    parser.add_argument("--kmeans-run", default=KMEANS_RUN,
                        help=f"Pasta K-means/outputs/<nome> usada na comparação (padrão: {KMEANS_RUN})")
    parser.add_argument("--compare-only", action="store_true",
                        help="Só refaz a comparação com os resultados já salvos (não precisa do dataset)")
    args = parser.parse_args()
    farms = "".join(sorted(set(args.farms.upper())))
    if not farms or set(farms) - set("ABC"):
        parser.error("--farms aceita só A, B e C.")
    if args.reusar and args.reusar == args.run:
        parser.error("--reusar precisa de um --run diferente, para não sobrescrever o experimento original.")
    criterion, grid = CRITERIOS[args.criterio], GRADES[args.grade]
    if args.compare_only:
        compare(farms, args.run, args.kmeans_run)
        return
    manifest = load_manifest(args.data_dir, farms)
    print(f"{len(manifest)} eventos ({manifest.split.eq('evaluation').sum()} reservados) · modelos: {args.models}", flush=True)
    roots = {f: PROJECT_ROOT / FAMILIES[f][0] / "outputs" / args.run for f in args.models}
    if args.reusar:
        sources = reuse_sources(args, manifest, farms, criterion, grid, roots)
    else:
        sources = train_all(args, manifest, farms, criterion, grid, roots)

    for family, root in roots.items():
        source = sources.get(family)
        if source is None:
            continue
        missing = [f"{e.farm}/{e.event}" for e in manifest.itertuples()
                   if not (event_dir(source, e) / "report.json").exists()]
        if missing:
            print(f"{family}: {len(missing)} eventos sem modelo ({missing[:5]}...). Veja error.json; "
                  "métricas não publicadas para não esconder falhas.", flush=True)
            continue
        frozen, ranking, metrics, _, regression = select_and_evaluate(
            family, root, manifest, args.max_false_per_30_days, args.max_active_fraction, criterion, grid, source)
        write_results(family, root, frozen, ranking, metrics, regression, args)
        plot_timelines(family, root, manifest, frozen, source)
        print(f"{family}: regra {frozen['config']} · resultados em {root / 'RESULTADOS.md'}", flush=True)
    compare(farms, args.run, args.kmeans_run)


def train_all(args, manifest, farms, criterion, grid, roots):
    code = [Path(__file__), Path(nbm.__file__)] + [nbm.KMEANS_SRC / n for n in (
        "alert_episodes.py", "failure_alerts.py", "tune_detection.py", "shared_reference.py")]
    protocol = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, farms=farms, seed=nbm.SEED, max_fit_rows=nbm.MAX_FIT_ROWS,
                    calibration_fraction=nbm.CALIBRATION_FRACTION, validation_fraction=nbm.VALIDATION_FRACTION,
                    gap="1 dia", max_targets=nbm.MAX_TARGETS, min_target_r2=nbm.MIN_TARGET_R2,
                    smoothing=nbm.SMOOTHING, svr_rows=nbm.SVR_ROWS, quantiles=list(nbm.QUANTILES),
                    alert_grid=[config_id(c) for c in GRIDS[grid][1]], baseline=config_id(BASELINE),
                    max_false_per_30_days=args.max_false_per_30_days, max_active_fraction=args.max_active_fraction,
                    criterion=criterion, grid=grid, selection=SELECAO[criterion],
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code})
    for family, root in roots.items():
        root.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(root / "manifest.csv", index=False)
        (root / "protocol.json").write_text(json.dumps(dict(protocol, family=family), indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    started = time.time()
    tasks = (delayed(process_event)(entry, args.models, roots) for entry in manifest.itertuples())
    for done, lines in enumerate(Parallel(n_jobs=args.jobs, return_as="generator_unordered")(tasks), 1):
        for line in lines:
            print(f"[{done}/{len(manifest)}] {line}", flush=True)
    print(f"Treino concluído em {(time.time() - started) / 60:.1f} min", flush=True)
    return dict(roots)


def reuse_sources(args, manifest, farms, criterion, grid, roots):
    """Confere que a execução de origem tem a mesma partição e fazendas; grava o protocolo do novo experimento."""
    sources = {}
    for family, root in roots.items():
        source = PROJECT_ROOT / FAMILIES[family][0] / "outputs" / args.reusar
        if not (source / "protocol.json").exists():
            print(f"{family}: sem outputs/{args.reusar}; pulado.", flush=True)
            continue
        original = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
        if original.get("farms") != farms:
            raise SystemExit(f"{family}: outputs/{args.reusar} foi treinado com as fazendas {original.get('farms')}, "
                             f"não {farms}.")
        old = pd.read_csv(source / "manifest.csv", dtype=str)[["farm", "event", "split"]]
        new = manifest[["farm", "event", "split"]].astype(str)
        if not old.sort_values(["farm", "event"]).reset_index(drop=True).equals(
                new.sort_values(["farm", "event"]).reset_index(drop=True)):
            raise SystemExit(f"{family}: a partição de outputs/{args.reusar} difere da atual.")
        root.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(root / "manifest.csv", index=False)
        protocol = dict(original, alert_grid=[config_id(c) for c in GRIDS[grid][1]],
                        max_false_per_30_days=args.max_false_per_30_days, max_active_fraction=args.max_active_fraction,
                        criterion=criterion, grid=grid, selection=SELECAO[criterion], reusa_run=args.reusar,
                        note=f"Modelos e escores de outputs/{args.reusar} (sem retreino); só a regra de alerta mudou.")
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
        sources[family] = source
    return sources


def compare(farms, run, kmeans_run=KMEANS_RUN):
    rows = collect_rows(run, farms)
    if rows:
        write_comparison(rows, farms, run, kmeans_run)
    else:
        print("Nenhum modelo avaliado nesta execução para comparar.", flush=True)


if __name__ == "__main__":
    main()
