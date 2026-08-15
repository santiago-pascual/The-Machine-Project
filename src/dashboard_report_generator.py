
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from dashboard_components import metric_card, source_caption
from dashboard_data_layer import load_all
from dashboard_report_export import export_report
from dashboard_report_templates import build_report_bundle


def generate_sample_reports(data: dict[str, pd.DataFrame] | None = None) -> dict[str, dict[str, str]]:
    if data is None:
        data, _ = load_all()
    outputs = {}
    audit_rows = []
    for report_type, basename in [("daily", "sample_daily_report"), ("weekly", "sample_weekly_report"), ("monthly", "sample_monthly_report")]:
        bundle = build_report_bundle(data, report_type=report_type)
        paths = export_report(bundle, basename)
        outputs[report_type] = paths
        audit_rows.append({
            "report_type": report_type,
            "status": bundle.status,
            "start_date": bundle.start_date,
            "end_date": bundle.end_date,
            "pdf": paths["pdf"],
            "html": paths["html"],
            "markdown": paths["markdown"],
            "warnings": "; ".join(bundle.warnings),
            "official_sources": len(bundle.source_audit),
            "generated_at": bundle.generated_at,
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv("report_generation_audit.csv", index=False)
    final_status = "report_generator_warning" if audit["status"].astype(str).str.contains("warning", case=False).any() else "report_generator_pass"
    report = [
        "===== PHASE 118 INSTITUTIONAL REPORT GENERATOR =====",
        f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"final_status: {final_status}",
        "read_only: True",
        "model_changed: False",
        "optimizer_changed: False",
        "paper_changed: False",
        "parameters_changed: False",
        "orders_sent: False",
        "",
        audit.to_string(index=False),
    ]
    Path("phase118_report_generator_report.txt").write_text("\n".join(report), encoding="utf-8")
    return outputs


def render_report_generator(st, data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Institutional Report Generator")
    st.caption("Read-only official reporting. Generates publication-style PDF, HTML and Markdown from official namespace files only.")
    outputs = generate_sample_reports(data)
    audit = pd.read_csv("report_generation_audit.csv") if Path("report_generation_audit.csv").exists() else pd.DataFrame()
    cols = st.columns(4)
    with cols[0]: metric_card(st, "Reports", str(len(outputs)), "daily / weekly / monthly")
    with cols[1]: metric_card(st, "Status", "PASS" if not audit.empty and not audit["status"].astype(str).str.contains("fail", case=False).any() else "CHECK", "read-only")
    with cols[2]: metric_card(st, "PDF Export", "enabled", "native minimal PDF")
    with cols[3]: metric_card(st, "Namespace", "official", "no debug mixing")
    if not audit.empty:
        st.dataframe(audit, width="stretch", hide_index=True)
    for report_type, paths in outputs.items():
        st.markdown(f"#### {report_type.title()} Report")
        st.write(paths)
    source_caption(st, "official namespace CSVs + dashboard_report_generator.py", "read-only")


def main() -> None:
    data, _ = load_all()
    outputs = generate_sample_reports(data)
    print("===== INSTITUTIONAL REPORT GENERATOR =====")
    for report_type, paths in outputs.items():
        print(report_type, paths)
    if Path("report_generation_audit.csv").exists():
        print(pd.read_csv("report_generation_audit.csv").to_string(index=False))


if __name__ == "__main__":
    main()
