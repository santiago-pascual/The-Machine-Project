from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard_alert_rules import ALERT_COLUMNS


def load_alert_history(path: str = "alert_history.csv") -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame(columns=ALERT_COLUMNS)
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    for col in ALERT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in {"resolved", "acknowledged", "occurrences", "duration_days"} else 0
    return df[ALERT_COLUMNS]


def _duration_days(first_seen: object, last_seen: object) -> int:
    start = pd.to_datetime(first_seen, errors="coerce")
    end = pd.to_datetime(last_seen, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return 0
    return max(0, int((end - start).days))


def update_alert_history(
    current_alerts: pd.DataFrame,
    history_path: str = "alert_history.csv",
    active_path: str = "active_alerts.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = load_alert_history(history_path)
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    current = current_alerts.copy() if current_alerts is not None else pd.DataFrame(columns=ALERT_COLUMNS)

    for col in ALERT_COLUMNS:
        if col not in current.columns:
            current[col] = "" if col not in {"resolved", "acknowledged", "occurrences", "duration_days"} else 0
    current = current[ALERT_COLUMNS]

    current_ids = set(current["id"].astype(str)) if not current.empty else set()
    hist_by_id = {str(row["id"]): row for _, row in history.iterrows()} if not history.empty else {}
    rows: list[dict] = []

    for _, row in current.iterrows():
        alert = row.to_dict()
        alert_id = str(alert["id"])
        previous = hist_by_id.get(alert_id)
        if previous is not None:
            alert["first_seen"] = previous.get("first_seen", alert.get("first_seen", now))
            alert["acknowledged"] = previous.get("acknowledged", False)
            alert["acknowledged_date"] = previous.get("acknowledged_date", "")
            try:
                alert["occurrences"] = int(float(previous.get("occurrences", 0))) + 1
            except Exception:
                alert["occurrences"] = 1
        else:
            alert["first_seen"] = alert.get("first_seen") or now
            alert["occurrences"] = 1
        alert["last_seen"] = now
        alert["timestamp"] = now
        alert["resolved"] = False
        alert["resolved_date"] = ""
        alert["status"] = "OPEN"
        alert["duration_days"] = _duration_days(alert["first_seen"], alert["last_seen"])
        rows.append(alert)

    for _, row in history.iterrows():
        alert_id = str(row.get("id", ""))
        if alert_id in current_ids:
            continue
        alert = row.to_dict()
        if str(alert.get("resolved", "")).lower() not in {"true", "1"}:
            alert["resolved"] = True
            alert["status"] = "RESOLVED"
            alert["resolved_date"] = now
            alert["last_seen"] = now
            alert["duration_days"] = _duration_days(alert.get("first_seen", now), now)
        rows.append(alert)

    out = pd.DataFrame(rows, columns=ALERT_COLUMNS).drop_duplicates("id", keep="first")
    active = out[out["resolved"].astype(str).str.lower().isin(["false", "0", ""])].copy()
    out.to_csv(history_path, index=False)
    active.to_csv(active_path, index=False)
    return out, active
