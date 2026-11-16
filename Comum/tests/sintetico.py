"""Gera um mini-CARE sintético com o MESMO formato de arquivos do dataset real.

Serve só para testar o código ponta a ponta sem os 5,5 GB do Zenodo. Os números
não têm relação com o desempenho nos dados reais.

Estrutura gerada:
    <raiz>/CARE_To_Compare/Wind Farm A/datasets/<id>.csv   (separador ;)
    <raiz>/CARE_To_Compare/Wind Farm A/event_info.csv
    <raiz>/CARE_To_Compare/Wind Farm A/feature_description.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = [
    # nome, descrição, is_angle, is_counter
    ("wind_speed_3", "Wind speed", False, False),
    ("power_29", "Active power", False, False),
    ("reactive_power_27", "Reactive power", False, False),
    ("sensor_0", "Ambient temperature", False, False),
    ("sensor_1", "Rotor speed", False, False),
    ("sensor_2", "Pitch angle", True, False),
    ("sensor_5", "Operating hours counter", False, True),
    ("sensor_11", "Gearbox bearing temperature", False, False),
    ("sensor_12", "Gearbox oil temperature", False, False),
    ("sensor_13", "Generator bearing temperature", False, False),
    ("sensor_14", "Nacelle temperature", False, False),
    ("sensor_15", "Hydraulic oil temperature", False, False),
]


def lag(signal, alpha):
    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


def event_frame(rng, asset, start, train_days, pred_days, anomaly, fault_sensor="sensor_11", drift=7.):
    n_train, n_pred = train_days * 144, pred_days * 144
    n = n_train + n_pred
    t = pd.date_range(start, periods=n, freq="10min")
    hours = np.arange(n) / 6
    wind = np.clip(7 + 3 * np.sin(2 * np.pi * hours / 37) + lag(rng.normal(0, 2.5, n), .05) * 2, 0, 25)
    power = np.clip((wind - 3) ** 3 / 600, 0, 1)
    rotor = np.where(wind > 3, 9 + 6 * power, 0) + rng.normal(0, .2, n)
    ambient = 14 + 6 * np.sin(2 * np.pi * hours / 24) + 4 * np.sin(2 * np.pi * hours / (24 * 30))
    load = lag(power, .03)
    offset = rng.normal(0, 1.5)
    temps = {
        "sensor_11": 35 + offset + .6 * ambient + 25 * load,
        "sensor_12": 40 + offset + .5 * ambient + 18 * load,
        "sensor_13": 30 + offset + .7 * ambient + 22 * lag(power, .05),
        "sensor_14": 18 + .8 * ambient + 6 * load,
        "sensor_15": 28 + .6 * ambient + 5 * load,
    }
    for key in temps:
        temps[key] = temps[key] + rng.normal(0, .6, n)
    event_start = event_end = None
    if anomaly:
        begin = n_train + int(.25 * n_pred)
        end = n_train + int(.85 * n_pred)
        ramp = np.zeros(n)
        ramp[begin:end] = np.linspace(0, drift, end - begin)
        ramp[end:] = drift
        temps[fault_sensor] = temps[fault_sensor] + ramp
        event_start, event_end = t[begin], t[end - 1]
    frame = pd.DataFrame({"time_stamp": t.strftime("%Y-%m-%d %H:%M:%S"), "asset_id": asset, "id": np.arange(n),
                          "train_test": np.where(np.arange(n) < n_train, "train", "prediction")})
    status = np.zeros(n, dtype=int)
    stops = rng.choice(n_train, size=n_train // 40, replace=False)
    status[stops] = 4
    frame["status_type_id"] = status
    values = {"wind_speed_3": wind, "power_29": power, "reactive_power_27": power * .2 + rng.normal(0, .01, n),
              "sensor_0": ambient, "sensor_1": rotor, "sensor_2": rng.uniform(0, 360, n),
              "sensor_5": np.cumsum(np.ones(n)), **temps}
    for name, *_ in SENSORS:
        frame[f"{name}_avg"] = values[name]
        frame[f"{name}_max"] = values[name] + abs(rng.normal(0, .5, n))
        frame[f"{name}_min"] = values[name] - abs(rng.normal(0, .5, n))
        frame[f"{name}_std"] = abs(rng.normal(0, .3, n))
    # Alguns ausentes, como no dado real.
    holes = rng.choice(n, size=n // 200, replace=False)
    frame.loc[holes, "sensor_12_avg"] = np.nan
    return frame, event_start, event_end


def build(root: Path, assets=6, events_per_asset=2, train_days=40, pred_days=10, seed=7) -> Path:
    rng = np.random.default_rng(seed)
    farm = Path(root) / "CARE_To_Compare" / "Wind Farm A"
    (farm / "datasets").mkdir(parents=True, exist_ok=True)
    info, event_id = [], 0
    for asset in range(assets):
        for k in range(events_per_asset):
            anomaly = (asset + k) % 2 == 0
            frame, start, end = event_frame(rng, asset * 10, f"2022-0{1 + k}-01", train_days, pred_days, anomaly)
            frame.to_csv(farm / "datasets" / f"{event_id}.csv", sep=";", index=False)
            prediction = pd.to_datetime(frame.loc[frame.train_test.eq("prediction"), "time_stamp"])
            info.append(dict(asset=asset * 10, event_id=event_id,
                             event_label="anomaly" if anomaly else "normal",
                             event_start=str(start if anomaly else prediction.min()),
                             event_end=str(end if anomaly else prediction.max()),
                             event_description="sintetico"))
            event_id += 1
    pd.DataFrame(info).to_csv(farm / "event_info.csv", sep=";", index=False)
    pd.DataFrame([dict(sensor_name=n, statistics_type="average,minimum,maximum,std_dev", description=d,
                       unit="", is_angle=a, is_counter=c) for n, d, a, c in SENSORS]
                 ).to_csv(farm / "feature_description.csv", sep=";", index=False)
    return Path(root)


if __name__ == "__main__":
    import sys
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "data_sintetico")))
