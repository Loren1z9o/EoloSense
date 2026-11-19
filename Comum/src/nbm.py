"""Modelos de comportamento normal (NBM) para o CARE to Compare.

Ideia: aprender, somente com operação normal, como os sensores de temperatura
respondem às condições de operação (vento, potência, rotação, temperatura
ambiente). Na previsão, o resíduo padronizado (real - previsto) vira o escore de
anomalia. O escore entra na MESMA regra de alerta e de episódios do K-means, para
que os quatro modelos sejam comparáveis nos mesmos eventos.

Nada aqui usa os rótulos de falha: entradas, alvos e hiperparâmetros são
escolhidos apenas com dados de treino normais (status 0), em validação
cronológica interna.
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KMEANS_SRC = PROJECT_ROOT / "K-means" / "src"
if str(KMEANS_SRC) not in sys.path:
    sys.path.insert(0, str(KMEANS_SRC))

from kmeans_baseline import split_rows  # noqa: E402
from shared_reference import candidate_features  # noqa: E402

# --- Parâmetros fixos do protocolo (iguais ao K-means onde se sobrepõem) -------
SEED = 42
MAX_FIT_ROWS = 12_000          # mesmo teto de linhas de ajuste do K-means
CALIBRATION_FRACTION = .2      # últimos 20% do treino normal calibram o limiar
GAP = pd.Timedelta(1, unit="D")  # 1 dia entre ajuste e calibração/validação
VALIDATION_FRACTION = .2       # últimos 20% do ajuste escolhem hiperparâmetros
MAX_VALIDATION_ROWS = 4_000
MAX_MISSING = .3
MAX_TARGETS = 10
MIN_TARGET_R2 = .3
SVR_ROWS = 3_000               # SVR é O(n²): treina numa amostra
QUANTILES = (.90, .925, .95, .975, .99)
SIGMA_FLOOR = .05
SMOOTHING = "1h"             # média móvel causal dos resíduos

ROLE_PATTERNS = {
    "vento": re.compile(r"wind\s*speed"),
    "potencia": re.compile(r"\bpower\b"),
    "rotacao": re.compile(r"(rotor|generator|shaft|hub)\s*(speed|rpm|revolution)|rotational speed|\brpm\b"),
    "ambiente": re.compile(r"(ambient|outside|outdoor|external)\b.*temp|temp\w*\s.*\b(ambient|outside|outdoor)"),
}
NOT_ACTIVE_POWER = re.compile(r"reactive|apparent|setpoint|set point|limit|available|possible|demand|"
                              r"reference|factor|consum|curtail|\bcos\b|phi")
TEMPERATURE = re.compile(r"temp")
MAX_PER_ROLE = 2


# --- Leitura e papéis dos sensores -------------------------------------------------
def load_event(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    path = Path(path)
    description_path = path.parent.parent / "feature_description.csv"
    header = pd.read_csv(path, sep=";", nrows=0)
    features = candidate_features(header, description_path)
    meta = [c for c in ("time_stamp", "asset_id", "train_test", "status_type_id") if c in header]
    frame = pd.read_csv(path, sep=";", usecols=features + meta)
    frame["time_stamp"] = pd.to_datetime(frame.time_stamp)
    if frame.time_stamp.duplicated().any():
        raise ValueError("Timestamps duplicados no arquivo.")
    frame = frame.sort_values("time_stamp").reset_index(drop=True)
    descriptions = {}
    if description_path.exists():
        table = pd.read_csv(description_path, sep=";")
        if {"sensor_name", "description"}.issubset(table.columns):
            descriptions = dict(zip(table.sensor_name.astype(str), table.description.fillna("").astype(str)))
    return frame, features, {f: descriptions.get(f.removesuffix("_avg"), "") for f in features}


def sensor_role(name: str, description: str) -> str | None:
    text = description.lower()
    base = name.lower()
    if text:
        if ROLE_PATTERNS["ambiente"].search(text):
            return "ambiente"
        if ROLE_PATTERNS["vento"].search(text):
            return "vento"
        if ROLE_PATTERNS["rotacao"].search(text):
            return "rotacao"
        if ROLE_PATTERNS["potencia"].search(text) and not NOT_ACTIVE_POWER.search(text):
            return "potencia"
        return None
    # Sem descrição: só os prefixos explícitos do CARE.
    if base.startswith("wind_speed_"):
        return "vento"
    if base.startswith("power_"):
        return "potencia"
    return None


@dataclass
class Layout:
    inputs: list[str]
    engineered: list[tuple[str, str, str]]  # (nova coluna, coluna base, janela)
    candidates: list[str]
    roles: dict[str, str]

    @property
    def x_columns(self) -> list[str]:
        return self.inputs + [name for name, _, _ in self.engineered]


def plan_layout(fit: pd.DataFrame, features: list[str], descriptions: dict[str, str]) -> Layout:
    values = fit[features].apply(pd.to_numeric, errors="coerce")
    usable = [f for f in features if values[f].isna().mean() <= MAX_MISSING and values[f].nunique() > 1]
    roles = {f: sensor_role(f, descriptions.get(f, "")) for f in usable}
    inputs, engineered = [], []
    for role in ("vento", "potencia", "rotacao", "ambiente"):
        members = sorted((f for f in usable if roles[f] == role), key=lambda f: (values[f].isna().mean(), f))
        inputs += members[:MAX_PER_ROLE]
        if members and role != "ambiente":
            # Inércia térmica: média causal recente da carga e do vento.
            windows = ("1h", "6h") if role != "vento" else ("1h",)
            engineered += [(f"{members[0]}__media_{w}", members[0], w) for w in windows]
    if not any(roles[f] in {"vento", "potencia"} for f in inputs):
        raise ValueError("Não encontrei sensores de vento ou potência para usar como condição de operação.")
    rest = [f for f in usable if roles[f] is None]
    temperatures = [f for f in rest if TEMPERATURE.search(descriptions.get(f, "").lower())]
    candidates = temperatures if len(temperatures) >= 3 else rest
    if len(candidates) < 1:
        raise ValueError("Nenhum sensor disponível como alvo.")
    return Layout(inputs, engineered, candidates, {f: roles[f] or "" for f in inputs})


def add_engineered(frame: pd.DataFrame, layout: Layout) -> pd.DataFrame:
    """Médias móveis CAUSAIS (só olham para trás) calculadas no arquivo inteiro."""
    out = frame.copy()
    index = pd.DatetimeIndex(out.time_stamp)
    for name, base, window in layout.engineered:
        series = pd.Series(pd.to_numeric(out[base], errors="coerce").to_numpy(), index=index)
        out[name] = series.rolling(window, min_periods=1).mean().to_numpy()
    return out


# --- Divisão temporal (sem vazamento) ----------------------------------------------
def split_event(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_mask, pred_mask = split_rows(frame)
    if not train_mask.any() or not pred_mask.any():
        raise ValueError("O arquivo precisa das seções de treino e de previsão.")
    if frame.loc[train_mask, "time_stamp"].max() >= frame.loc[pred_mask, "time_stamp"].min():
        raise ValueError("O treino deve terminar antes da previsão.")
    healthy = frame.loc[train_mask & pd.to_numeric(frame.status_type_id, errors="coerce").eq(0)]
    if len(healthy) < 500:
        raise ValueError("Menos de 500 linhas de treino normal.")
    cal_start = healthy.time_stamp.iloc[int((1 - CALIBRATION_FRACTION) * len(healthy))]
    fit = healthy.loc[healthy.time_stamp < cal_start - GAP]
    calibration = healthy.loc[healthy.time_stamp >= cal_start]
    val_start = fit.time_stamp.iloc[int((1 - VALIDATION_FRACTION) * len(fit))]
    fit_train = fit.loc[fit.time_stamp < val_start - GAP]
    fit_val = fit.loc[fit.time_stamp >= val_start]
    if min(len(fit_train), len(fit_val), len(calibration)) < 100:
        raise ValueError("Dados normais insuficientes após separar ajuste, validação e calibração.")
    return dict(fit=fit, fit_train=fit_train, fit_val=fit_val, calibration=calibration,
                prediction=frame.loc[pred_mask], cal_start=cal_start, val_start=val_start)


def sample(frame: pd.DataFrame, rows: int, seed: int = SEED) -> pd.DataFrame:
    return frame.sample(rows, random_state=seed).sort_values("time_stamp") if len(frame) > rows else frame


def matrices(frame: pd.DataFrame, x_columns, targets):
    x = frame[x_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
    y = frame[targets].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
    return x, y


def complete_targets(x, y):
    keep = ~np.isnan(y).any(axis=1)
    return x[keep], y[keep]


def choose_targets(train, val, x_columns, candidates):
    """Ranqueia candidatos pelo R² de uma Ridge (sem rótulos de falha) e fica com os mais previsíveis."""
    x_tr, _ = matrices(train, x_columns, [])
    x_va, _ = matrices(val, x_columns, [])
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    x_tr = scaler.fit_transform(imputer.fit_transform(x_tr))
    x_va = scaler.transform(imputer.transform(x_va))
    scores = {}
    for target in candidates:
        y_tr = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
        y_va = pd.to_numeric(val[target], errors="coerce").to_numpy(float)
        a, b = ~np.isnan(y_tr), ~np.isnan(y_va)
        if a.sum() < 100 or b.sum() < 50 or np.nanstd(y_va) == 0:
            continue
        model = Ridge(alpha=1.).fit(x_tr[a], y_tr[a])
        residual = y_va[b] - model.predict(x_va[b])
        scores[target] = 1 - residual.var() / y_va[b].var()
    ranked = sorted(scores, key=lambda t: (-scores[t], t))
    chosen = [t for t in ranked if scores[t] >= MIN_TARGET_R2][:MAX_TARGETS]
    if len(chosen) < 3:
        chosen = ranked[:3]
    if not chosen:
        raise ValueError("Nenhum alvo previsível a partir das condições de operação.")
    return chosen, {t: float(scores[t]) for t in ranked}


# --- Famílias de modelos ------------------------------------------------------------
def _tree_estimator(**params):
    return DecisionTreeRegressor(random_state=SEED, **params)


def candidate_grid(family: str) -> list[tuple[str, object, int | None]]:
    """(rótulo, estimador sem treino, limite de linhas de treino)."""
    imputed = lambda est, scale=False: make_pipeline(  # noqa: E731
        SimpleImputer(strategy="median"), *( [StandardScaler()] if scale else [] ), est)
    if family == "linear":
        return [(f"ridge alpha={a}", imputed(Ridge(alpha=a), scale=True), None) for a in (.1, 1., 10., 100.)]
    if family == "arvore":
        return [(f"arvore max_depth={d} min_samples_leaf={m}",
                 imputed(_tree_estimator(max_depth=d, min_samples_leaf=m)), None)
                for d in (6, 10, 14, None) for m in (10, 40, 120)]
    if family == "ensembles":
        grid = []
        for leaf in (5, 20):
            for feats in (.5, 1.):
                grid.append((f"random_forest min_samples_leaf={leaf} max_features={feats}",
                             imputed(RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf, max_features=feats,
                                                           max_samples=.7, random_state=SEED, n_jobs=1)), None))
                grid.append((f"extra_trees min_samples_leaf={leaf} max_features={feats}",
                             imputed(ExtraTreesRegressor(n_estimators=100, min_samples_leaf=leaf, max_features=feats,
                                                         random_state=SEED, n_jobs=1)), None))
        for leaves in (15, 31):
            grid.append((f"gradient_boosting max_leaf_nodes={leaves}",
                         imputed(MultiOutputRegressor(HistGradientBoostingRegressor(
                             learning_rate=.08, max_iter=400, max_leaf_nodes=leaves, min_samples_leaf=40,
                             l2_regularization=1., early_stopping=True, validation_fraction=.15,
                             n_iter_no_change=20, random_state=SEED))), None))
        return grid
    if family == "svm":
        return [(f"svr rbf C={c} epsilon={e}",
                 imputed(MultiOutputRegressor(SVR(kernel="rbf", C=c, epsilon=e, gamma="scale", cache_size=500)),
                         scale=True), SVR_ROWS)
                for c in (1., 10., 100.) for e in (.05, .2)]
    raise ValueError(f"Família desconhecida: {family}")


def finalize(label: str, estimator):
    """Na versão final, florestas ganham mais árvores (menos variância, mesma configuração)."""
    final = clone(estimator)
    last = final.steps[-1][1]
    if isinstance(last, (RandomForestRegressor, ExtraTreesRegressor)):
        last.set_params(n_estimators=300)
    return final


def z_rmse(y_true_z, y_pred_z):
    return float(np.sqrt(np.mean((y_true_z - y_pred_z) ** 2, axis=0)).mean())


def prune_tree(estimator, x_tr, y_tr, x_va, y_va):
    """Poda por custo-complexidade (ccp_alpha), escolhida na validação."""
    tree = estimator.steps[-1][1]
    imputer = estimator.steps[0][1]
    path = tree.cost_complexity_pruning_path(imputer.transform(x_tr), y_tr)
    alphas = np.unique(np.quantile(path.ccp_alphas[:-1], np.linspace(0, .98, 12))) if len(path.ccp_alphas) > 1 else [0.]
    rows = []
    for alpha in alphas:
        candidate = clone(estimator).set_params(**{f"{estimator.steps[-1][0]}__ccp_alpha": float(alpha)})
        candidate.fit(x_tr, y_tr)
        rows.append((z_rmse(y_va, candidate.predict(x_va)), -float(alpha), float(alpha), candidate))
    best = min(rows, key=lambda r: (round(r[0], 4), r[1]))  # empate -> poda maior (árvore mais simples)
    return best[3], best[2], [dict(ccp_alpha=r[2], val_z_rmse=r[0]) for r in rows]


def tune(family, x_tr, y_tr, x_va, y_va, seed=SEED, candidate_options=None):
    rng = np.random.default_rng(seed)
    table, best = [], None
    for order, (label, estimator, limit) in enumerate(candidate_options if candidate_options is not None else candidate_grid(family)):
        xs, ys = x_tr, y_tr
        if limit and len(xs) > limit:
            pick = rng.choice(len(xs), limit, replace=False)
            xs, ys = xs[pick], ys[pick]
        start = time.perf_counter()
        model = clone(estimator).fit(xs, ys)
        error = z_rmse(y_va, model.predict(x_va))
        table.append(dict(candidate=label, val_z_rmse=error, seconds=round(time.perf_counter() - start, 2)))
        if best is None or error < best[0] - 1e-4:
            best = (error, order, label, estimator, limit)
    return best, table


@dataclass
class TrainedNBM:
    bundle: dict
    regression: dict


def train_nbm(frame, features, descriptions, family, farm, asset_id, *,
              layout_planner=None, target_selector=None, candidate_options=None, max_fit_rows=MAX_FIT_ROWS):
    started = time.perf_counter()
    parts = split_event(frame)
    layout = (layout_planner or plan_layout)(parts["fit_train"], features, descriptions)
    full = add_engineered(frame, layout)
    view = {k: full.loc[v.index] for k, v in parts.items() if isinstance(v, pd.DataFrame)}
    fit_train = sample(view["fit_train"], int(max_fit_rows * (1 - VALIDATION_FRACTION)))
    fit_val = sample(view["fit_val"], MAX_VALIDATION_ROWS)
    targets, target_r2 = (target_selector or choose_targets)(fit_train, fit_val, layout.x_columns, layout.candidates)

    x_tr, y_tr = complete_targets(*matrices(fit_train, layout.x_columns, targets))
    x_va, y_va = complete_targets(*matrices(fit_val, layout.x_columns, targets))
    y_scaler = StandardScaler().fit(y_tr)
    yz_tr, yz_va = y_scaler.transform(y_tr), y_scaler.transform(y_va)

    (best_error, _, label, estimator, limit), table = tune(family, x_tr, yz_tr, x_va, yz_va, candidate_options=candidate_options)
    pruning = None
    if family == "arvore":
        pruned, alpha, pruning = prune_tree(clone(estimator).fit(x_tr, yz_tr), x_tr, yz_tr, x_va, yz_va)
        estimator = clone(pruned)
        label = f"{label} ccp_alpha={alpha:.2e}"
    # Desvio dos resíduos em dados NÃO vistos (validação), usado para padronizar o escore.
    probe = finalize(label, estimator)
    rng = np.random.default_rng(SEED)
    xs, ys = x_tr, yz_tr
    if limit and len(xs) > limit:
        pick = rng.choice(len(xs), limit, replace=False)
        xs, ys = xs[pick], ys[pick]
    probe.fit(xs, ys)
    sigma = np.maximum(np.std(yz_va - probe.predict(x_va), axis=0), SIGMA_FLOOR)
    val_error = z_rmse(yz_va, probe.predict(x_va))

    # Modelo final: mesmo hiperparâmetro, reajustado com todo o trecho de ajuste.
    fit_all = sample(view["fit"], max_fit_rows)
    x_fit, y_fit = complete_targets(*matrices(fit_all, layout.x_columns, targets))
    if limit and len(x_fit) > limit:
        pick = rng.choice(len(x_fit), limit, replace=False)
        x_fit, y_fit = x_fit[pick], y_fit[pick]
    model = finalize(label, estimator).fit(x_fit, y_scaler.transform(y_fit))

    bundle = dict(version=1, family=family, farm=farm, asset_id=str(asset_id), chosen=label,
                  model=model, y_scaler=y_scaler, sigma=sigma, inputs=layout.inputs,
                  engineered=layout.engineered, x_columns=layout.x_columns, targets=targets,
                  target_descriptions={t: descriptions.get(t, "") for t in targets},
                  input_roles=layout.roles, target_r2_ridge=target_r2, tuning=table, pruning=pruning,
                  val_z_rmse=val_error, max_missing=MAX_MISSING, seed=SEED,
                  calibration_start=str(parts["cal_start"]), validation_start=str(parts["val_start"]),
                  training_end=str(frame.loc[split_rows(frame)[0], "time_stamp"].max()),
                  fit_rows=len(x_fit), seconds=None)
    cal_scores = score_frame(bundle, view["calibration"], farm, require_status=False)
    eligible = cal_scores.loc[cal_scores.eligible, "score"].dropna()
    if len(eligible) < 50:
        raise ValueError("Poucas linhas elegíveis na calibração.")
    bundle["calibration_thresholds"] = {q: float(np.quantile(eligible, q)) for q in QUANTILES}
    bundle["calibration_rows"] = int(len(eligible))
    bundle["importances"] = importances(model, layout.x_columns)
    regression = regression_metrics(bundle, view["calibration"])
    bundle["seconds"] = round(time.perf_counter() - started, 1)
    return TrainedNBM(bundle, regression), full


def predict_z(bundle, frame):
    x, y = matrices(frame, bundle["x_columns"], bundle["targets"])
    predicted = bundle["model"].predict(x)
    observed = (y - bundle["y_scaler"].mean_) / bundle["y_scaler"].scale_
    return observed, predicted, y


def score_frame(bundle, frame, farm, require_status=True):
    """Escore de anomalia de cada registro de 10 minutos.

    1. resíduo padronizado de cada alvo: (real - previsto) / desvio dos resíduos na validação;
    2. média móvel CAUSAL de 1 h por alvo (tira o ruído de leitura, mantém desvios persistentes);
    3. escore = maior |resíduo suavizado| entre os alvos. Falhas costumam afetar um componente;
       a média entre todos os sensores diluiria esse sinal.
    Também guarda qual sensor mais desviou, para interpretar o alerta.
    """
    observed, predicted, raw = predict_z(bundle, frame)
    residual = (observed - predicted) / bundle["sigma"]
    used = frame[bundle["inputs"] + bundle["targets"]].apply(pd.to_numeric, errors="coerce")
    eligible = (used.isna().mean(axis=1) <= bundle["max_missing"]).to_numpy() & ~np.isnan(raw).all(axis=1)
    if require_status and farm != "A":
        eligible &= pd.to_numeric(frame.status_type_id, errors="coerce").eq(0).to_numpy()
    residual[~eligible] = np.nan
    smooth = (pd.DataFrame(residual, index=pd.DatetimeIndex(frame.time_stamp))
              .rolling(SMOOTHING, min_periods=3).mean().to_numpy())
    absolute = np.where(np.isnan(smooth), -np.inf, np.abs(smooth))
    position = absolute.argmax(axis=1)
    has_value = np.isfinite(absolute.max(axis=1))
    ok = eligible & has_value
    top = np.array(bundle["targets"], dtype=object)[position]
    return pd.DataFrame(dict(time_stamp=frame.time_stamp.to_numpy(), eligible=ok,
                             score=np.where(ok, absolute.max(axis=1), np.nan),
                             top_sensor=np.where(ok, top, None),
                             top_residual=np.where(ok, np.take_along_axis(smooth, position[:, None], 1)[:, 0], np.nan)))


def regression_metrics(bundle, frame):
    """MAE/RMSE/R² por alvo, em unidades originais, na calibração (dados normais não vistos)."""
    observed, predicted, raw = predict_z(bundle, frame)
    pred_raw = bundle["y_scaler"].inverse_transform(predicted)
    rows = []
    for j, target in enumerate(bundle["targets"]):
        ok = ~np.isnan(raw[:, j])
        error = raw[ok, j] - pred_raw[ok, j]
        variance = raw[ok, j].var()
        rows.append(dict(target=target, description=bundle["target_descriptions"].get(target, ""),
                         mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())),
                         r2=float(1 - (error ** 2).mean() / variance) if variance else None))
    table = pd.DataFrame(rows)
    return dict(per_target=rows, mean_mae=float(table.mae.mean()), mean_rmse=float(table.rmse.mean()),
                mean_r2=float(table.r2.dropna().mean()) if table.r2.notna().any() else None,
                z_rmse=float(np.mean([np.sqrt(np.nanmean((observed[:, j] - predicted[:, j]) ** 2))
                                      for j in range(observed.shape[1])])))


def importances(model, columns):
    last = model.steps[-1][1]
    values = getattr(last, "feature_importances_", None)
    if values is None and isinstance(last, MultiOutputRegressor):
        return None
    if values is None:
        return None
    return {c: float(v) for c, v in sorted(zip(columns, values), key=lambda p: -p[1])}
