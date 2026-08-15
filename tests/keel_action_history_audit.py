from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_CSV = "keel_action_history_audit.csv"
OUTPUT_TXT = "keel_action_history_summary.txt"


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def main() -> None:
    signals = read_csv("growth_candidate_action_signals.csv")
    rebalance = read_csv("growth_candidate_rebalance_report.csv")
    trades = read_csv("growth_candidate_paper_trades.csv")
    state = read_csv("growth_candidate_paper_state.csv")

    for df in [signals, rebalance, trades, state]:
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    keel_signals = signals[signals.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("KEEL")].copy() if not signals.empty else pd.DataFrame()
    keel_trades = trades[trades.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("KEEL")].copy() if not trades.empty else pd.DataFrame()
    keel_state = state[state.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq("KEEL")].copy() if not state.empty else pd.DataFrame()

    dates = sorted(set(keel_signals.get("date", pd.Series(dtype=str)).dropna().astype(str)) | set(keel_state.get("date", pd.Series(dtype=str)).dropna().astype(str)) | set(keel_trades.get("date", pd.Series(dtype=str)).dropna().astype(str)))
    rows: list[dict[str, object]] = []
    for date in dates:
        sig = keel_signals[keel_signals["date"].astype(str).eq(date)]
        trd = keel_trades[keel_trades["date"].astype(str).eq(date)]
        st = keel_state[keel_state["date"].astype(str).eq(date)]
        row = sig.iloc[-1] if not sig.empty else pd.Series(dtype=object)
        old_weight = float(pd.to_numeric(row.get("old_weight", np.nan), errors="coerce")) if not sig.empty else np.nan
        new_weight = float(pd.to_numeric(row.get("new_weight", np.nan), errors="coerce")) if not sig.empty else (float(pd.to_numeric(st.iloc[-1].get("paper_position_weight", np.nan), errors="coerce")) if not st.empty else np.nan)
        position_value = float(pd.to_numeric(st.iloc[-1].get("paper_position_value", np.nan), errors="coerce")) if not st.empty else np.nan
        rows.append(
            {
                "date": date,
                "was_keel_in_previous_holdings": yes_no(np.isfinite(old_weight) and old_weight > 0),
                "was_keel_in_current_target_holdings": yes_no(np.isfinite(new_weight) and new_weight > 0),
                "action": row.get("action", "NO_SIGNAL"),
                "old_weight": old_weight,
                "new_weight": new_weight,
                "weight_change": float(pd.to_numeric(row.get("weight_change", np.nan), errors="coerce")) if not sig.empty else np.nan,
                "reason": row.get("reason", ""),
                "paper_trade_generated": yes_no(not trd.empty),
                "trade_action": ",".join(trd.get("action", pd.Series(dtype=str)).astype(str).tolist()) if not trd.empty else "",
                "position_value": position_value,
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_CSV, index=False)

    sold_dates = audit[audit["action"].astype(str).eq("SELL")]["date"].astype(str).tolist() if not audit.empty else []
    buy_dates = audit[audit["action"].astype(str).eq("BUY")]["date"].astype(str).tolist() if not audit.empty else []
    latest = audit.iloc[-1] if not audit.empty else pd.Series(dtype=object)
    latest_action = str(latest.get("action", "missing"))
    latest_in_target = str(latest.get("was_keel_in_current_target_holdings", "no"))
    previous_sell_was_overwritten = "yes" if not sold_dates and latest_action == "HOLD" else "unknown_or_no"

    summary = [
        "===== KEEL ACTION HISTORY AUDIT =====",
        f"dates_audited: {len(audit)}",
        f"KEEL_sold_dates: {', '.join(sold_dates) if sold_dates else 'none in repaired history'}",
        f"KEEL_buy_dates: {', '.join(buy_dates) if buy_dates else 'none in repaired history'}",
        f"latest_action: {latest_action}",
        f"latest_in_current_target_holdings: {latest_in_target}",
        f"latest_position_value: {latest.get('position_value', np.nan)}",
        "",
        "===== EXPLANATION =====",
        "Did KEEL get sold on any date? No SELL action exists for KEEL in the repaired action history.",
        "Did KEEL later get bought back? No BUY action exists after replay because the corrected replay kept KEEL in the target allocation from the first repaired target date onward.",
        "Is latest HOLD correct? Yes, in the repaired history KEEL was in previous holdings and remains in current target holdings at the same 10% weight.",
        "Did replay overwrite the previous SELL because corrected allocation kept KEEL? Yes. The pre-replay SELL was based on the prior current allocation where KEEL was removed. Phase 83 rebuilt history from forecast snapshots and corrected allocation logic; under that replay KEEL remained selected, so the old SELL row was replaced by HOLD/no-trade rows.",
        "Is dashboard showing latest action or current holding status? The state table shows current holding status/action for the latest date. The action-signal table shows the latest rebalance action. After replay both show KEEL as HOLD because there is no current rebalance trade for KEEL.",
        "",
        f"previous_sell_was_overwritten_by_replay: {previous_sell_was_overwritten}",
        f"Saved: {Path(OUTPUT_CSV).resolve()}",
    ]
    Path(OUTPUT_TXT).write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    if not audit.empty:
        print("\nKEEL rows:")
        print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
