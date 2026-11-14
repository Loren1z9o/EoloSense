"""Causal incident lifecycle: notify once, rearm after observed recovery."""
from __future__ import annotations

import pandas as pd


def group_alerts(scores, recovery_hours=6, state=None):
    if recovery_hours <= 0 or recovery_hours * 6 != int(recovery_hours * 6):
        raise ValueError("A recuperação deve ser positiva e múltipla de 10 minutos.")
    state = dict(state or {"active": False, "clear_records": 0, "episode_id": 0, "last_timestamp": None})
    result = scores.copy()
    notifications, active, identifiers, recoveries = [], [], [], []
    previous = pd.Timestamp(state["last_timestamp"]) if state["last_timestamp"] else None
    for row in result.itertuples():
        now = pd.Timestamp(row.time_stamp)
        if previous is not None:
            if now <= previous:
                raise ValueError("O estado exige novos registros posteriores ao último processado.")
            if now.value - previous.value != 600_000_000_000:
                state["clear_records"] = 0
        known = bool(row.eligible) and pd.notna(row.rolling_exceedance_fraction)
        notification = recovered = False
        if known and row.alarm:
            state["clear_records"] = 0
            if not state["active"]:
                state["active"] = True
                state["episode_id"] += 1
                notification = True
        elif state["active"]:
            state["clear_records"] = state["clear_records"] + 1 if known else 0
            if state["clear_records"] >= recovery_hours * 6:
                state["active"] = False
                state["clear_records"] = 0
                recovered = True
        notifications.append(notification)
        active.append(state["active"])
        identifiers.append(state["episode_id"] if state["active"] or recovered else pd.NA)
        recoveries.append(recovered)
        previous = now
    if len(result):
        state["last_timestamp"] = previous.isoformat()
    result["notification"] = notifications
    result["incident_active"] = active
    result["episode_id"] = pd.array(identifiers, dtype="Int64")
    result["recovered"] = recoveries
    return result, state
