from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "original_data.xlsx"
OUTPUT_DIR = ROOT / "maturity_quantity_quality_outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_FILE = OUTPUT_DIR / "report.txt"
DOCX_FILE = ROOT / "Maturity, quantity, and integration interpretation.docx"

NUMERIC_COLUMNS = [
    "Awareness Score",
    "Number of Tools Adopted",
    "Integration Score",
    "Performance Score",
    "Maturity Score",
]


def parse_assigned_numeric(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"Assigned\s*([0-9.]+)", text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        return None


def normalize_tool(tool: str) -> str:
    text = str(tool).strip().strip(".")
    replacements = {
        "AMI/Smart meters": "AMI",
        "IoT": "IoT sensors",
        "Sensors": "IoT sensors",
        "Automated devices": "IoT sensors",
        "Data analytics": "Analytics",
        "Billing systems": "Billing",
        "Distribution systems": "Distribution",
        "Quality monitoring": "Quality",
    }
    return replacements.get(text, text)


def split_tools(text: object) -> list[str]:
    if pd.isna(text):
        return []
    parts = re.split(r";|,", str(text))
    tools = [normalize_tool(part) for part in parts if str(part).strip()]
    return [tool for tool in tools if tool]


def load_data() -> pd.DataFrame:
    raw = pd.read_excel(SOURCE_FILE, header=None)
    frame = raw.iloc[3:].copy().reset_index(drop=True)
    frame.columns = raw.iloc[2]
    frame = frame.loc[:, [column for column in frame.columns if not pd.isna(column)]]
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].ffill()
    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].map(parse_assigned_numeric)
    frame["row_id"] = np.arange(1, len(frame) + 1)
    frame["is_adopted"] = frame["Adoption Status"].astype(str).str.strip().eq("Adopted")
    frame["integration_pure_full"] = frame["System Integration Level"].astype(str).str.strip().eq("Full")
    frame["tool_list"] = frame["Digital Tools Adopted"].map(split_tools)
    frame["tool_count_listed"] = frame["tool_list"].map(len)
    return frame


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Number of Tools Adopted", "Integration Score", "Maturity Score", "Performance Score"]
    return df[cols].corr().round(3)


def contradiction_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adopted = df[df["is_adopted"]].copy()
    low_full = adopted[(adopted["Maturity Score"] <= 1) & (adopted["integration_pure_full"])].copy()
    high_not_full = adopted[(adopted["Maturity Score"] >= 3) & (~adopted["integration_pure_full"])].copy()
    keep = [
        "Institution",
        "Digital Tools Adopted",
        "Number of Tools Adopted",
        "System Integration Level",
        "Integration Score",
        "Technological Maturity",
        "Maturity Score",
        "Performance Score",
    ]
    return low_full[keep], high_not_full[keep]


def maturity_integration_summary(df: pd.DataFrame) -> pd.DataFrame:
    adopted = df[df["is_adopted"]].copy()
    summary = (
        adopted.groupby(["Maturity Score", "System Integration Level"], dropna=False)
        .agg(
            institutions=("Institution", "size"),
            mean_quantity=("Number of Tools Adopted", "mean"),
            mean_performance=("Performance Score", "mean"),
        )
        .reset_index()
        .sort_values(["Maturity Score", "System Integration Level"])
    )
    summary["mean_quantity"] = summary["mean_quantity"].round(2)
    summary["mean_performance"] = summary["mean_performance"].round(2)
    return summary


def performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    adopted = df[df["is_adopted"]].copy()
    rows = []
    for maturity, sub in adopted.groupby("Maturity Score"):
        rows.append(
            {
                "maturity_score": maturity,
                "n": len(sub),
                "mean_quantity": round(float(sub["Number of Tools Adopted"].mean()), 2),
                "mean_integration": round(float(sub["Integration Score"].mean()), 2),
                "mean_performance": round(float(sub["Performance Score"].mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("maturity_score")


def tool_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    adopted = df[df["is_adopted"]].copy()
    long_rows: list[dict[str, object]] = []
    for _, row in adopted.iterrows():
        for tool in row["tool_list"]:
            long_rows.append(
                {
                    "tool": tool,
                    "institution": row["Institution"],
                    "quantity": row["Number of Tools Adopted"],
                    "integration_score": row["Integration Score"],
                    "maturity_score": row["Maturity Score"],
                    "performance_score": row["Performance Score"],
                }
            )
    long_df = pd.DataFrame(long_rows)
    summary = (
        long_df.groupby("tool")
        .agg(
            mentions=("tool", "size"),
            mean_quantity=("quantity", "mean"),
            mean_integration=("integration_score", "mean"),
            mean_maturity=("maturity_score", "mean"),
            mean_performance=("performance_score", "mean"),
        )
        .reset_index()
        .sort_values(["mentions", "mean_maturity"], ascending=[False, False])
    )
    for column in ["mean_quantity", "mean_integration", "mean_maturity", "mean_performance"]:
        summary[column] = summary[column].round(2)
    return summary

def safe_label(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def heatmap_svg(summary: pd.DataFrame, path: Path) -> None:
    width, height = 980, 560
    margin_left, margin_top = 180, 95
    cell_w, cell_h = 145, 78
    parts = svg_header(
        width,
        height,
        "Figure 1. Maturity by integration status",
        "Cells show institution count and mean performance among adopted cases",
    )
    maturity_levels = sorted(summary["Maturity Score"].dropna().unique())
    integration_levels = [
        "Full",
        "Partial → Full",
        "Partial",
        "None → Full",
        "None → Partial",
    ]
    max_n = max(int(summary["institutions"].max()), 1)
    for idx, level in enumerate(integration_levels):
        x = margin_left + idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{margin_top-16}" text-anchor="middle" font-size="12" font-family="Arial">{safe_label(level)}</text>'
        )
    for ridx, maturity in enumerate(maturity_levels):
        y = margin_top + ridx * cell_h + cell_h / 2 + 5
        parts.append(
            f'<text x="{margin_left-14}" y="{y:.1f}" text-anchor="end" font-size="12" font-family="Arial">Maturity {maturity:.0f}</text>'
        )
        for cidx, level in enumerate(integration_levels):
            x0 = margin_left + cidx * cell_w
            y0 = margin_top + ridx * cell_h
            row = summary[
                (summary["Maturity Score"] == maturity) & (summary["System Integration Level"] == level)
            ]
            if row.empty:
                count = 0
                perf = None
            else:
                count = int(row.iloc[0]["institutions"])
                perf = float(row.iloc[0]["mean_performance"])
            intensity = count / max_n
            fill = f"rgb({245 - int(70 * intensity)},{241 - int(40 * intensity)},{230 - int(120 * intensity)})"
            parts.append(f'<rect x="{x0}" y="{y0}" width="{cell_w-10}" height="{cell_h-10}" rx="8" fill="{fill}" stroke="#d7d1c4"/>')
            parts.append(
                f'<text x="{x0 + (cell_w-10)/2:.1f}" y="{y0+30:.1f}" text-anchor="middle" font-size="16" font-family="Georgia">{count}</text>'
            )
            perf_text = "NA" if perf is None else f"Perf {perf:.2f}"
            parts.append(
                f'<text x="{x0 + (cell_w-10)/2:.1f}" y="{y0+53:.1f}" text-anchor="middle" font-size="11" font-family="Arial" fill="#444">{perf_text}</text>'
            )
    legend_x, legend_y = width - 220, height - 65
    for i in range(5):
        intensity = i / 4
        fill = f"rgb({245 - int(70 * intensity)},{241 - int(40 * intensity)},{230 - int(120 * intensity)})"
        parts.append(f'<rect x="{legend_x + i*28}" y="{legend_y}" width="24" height="16" fill="{fill}" stroke="#d7d1c4"/>')
    parts.append(f'<text x="{legend_x}" y="{legend_y-8}" font-size="11" font-family="Arial">Higher shading = more institutions</text>')
    save_svg(path, parts)


def tool_map_svg(tool_summary: pd.DataFrame, path: Path) -> None:
    top = tool_summary[tool_summary["mentions"] >= 2].copy().sort_values("mentions", ascending=False).head(12)
    width, height = 940, 560
    margin = 85
    parts = svg_header(width, height, None)
    x = top["mean_maturity"].to_numpy(float)
    y = top["mean_integration"].to_numpy(float)
    x_lo, x_hi = max(0.0, float(x.min()) - 0.2), min(4.2, float(x.max()) + 0.2)
    y_lo, y_hi = max(0.0, float(y.min()) - 0.15), min(2.1, float(y.max()) + 0.15)

    def x_map(v: float) -> float:
        return margin + (v - x_lo) * (width - 2 * margin) / max(1e-9, x_hi - x_lo)

    def y_map(v: float) -> float:
        return height - margin - (v - y_lo) * (height - 2 * margin) / max(1e-9, y_hi - y_lo)

    parts.extend(
        [
            f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        ]
    )
    x_grid_ticks = np.arange(math.ceil(x_lo * 2) / 2, math.floor(x_hi * 2) / 2 + 0.001, 0.5)
    y_grid_ticks = np.arange(math.ceil(y_lo * 4) / 4, math.floor(y_hi * 4) / 4 + 0.001, 0.25)
    for tick in x_grid_ticks:
        tx = x_map(float(tick))
        parts.append(f'<line x1="{tx:.1f}" y1="{margin}" x2="{tx:.1f}" y2="{height-margin}" stroke="#e2ddd3" stroke-width="1"/>')
    for tick in y_grid_ticks:
        ty = y_map(float(tick))
        parts.append(f'<line x1="{margin}" y1="{ty:.1f}" x2="{width-margin}" y2="{ty:.1f}" stroke="#e2ddd3" stroke-width="1"/>')
    for _, row in top.iterrows():
        cx = x_map(float(row["mean_maturity"]))
        cy = y_map(float(row["mean_integration"]))
        radius = 7 + 1.6 * math.sqrt(float(row["mentions"]))
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="#4e79a7" fill-opacity="0.7" stroke="#1f3a5f" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{cx+10:.1f}" y="{cy-8:.1f}" font-size="11" font-family="Arial">{safe_label(row["tool"])}</text>'
        )
    x_major_ticks = np.arange(math.ceil(x_lo), math.floor(x_hi) + 0.001, 1.0)
    y_major_ticks = np.arange(math.ceil(y_lo * 2) / 2, math.floor(y_hi * 2) / 2 + 0.001, 0.5)
    for tick in x_major_ticks:
        tx = x_map(float(tick))
        parts.append(f'<line x1="{tx:.1f}" y1="{height-margin}" x2="{tx:.1f}" y2="{height-margin+6}" stroke="#222"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height-margin+20}" text-anchor="middle" font-size="11">{tick:.0f}</text>')
    for tick in y_major_ticks:
        ty = y_map(float(tick))
        parts.append(f'<line x1="{margin-6}" y1="{ty:.1f}" x2="{margin}" y2="{ty:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{margin-10}" y="{ty+4:.1f}" text-anchor="end" font-size="11">{tick:.1f}</text>')
    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13" font-family="Arial">Average maturity score (quality)</text>')
    parts.append(f'<text x="24" y="{height/2}" transform="rotate(-90 24 {height/2})" text-anchor="middle" font-size="13" font-family="Arial">Average integration score</text>')
    save_svg(path, parts)


    for idx, (label, left_val, right_val) in enumerate(rows):
        ry = y0 + header_h + idx * row_h
        fill = "#fcfaf5" if idx % 2 == 0 else "#fffdfa"
        parts.append(f'<rect x="{x0}" y="{ry}" width="{col0+col1+col2}" height="{row_h}" fill="{fill}" stroke="#ebe4d8"/>')
        parts.append(f'<line x1="{x0+col0}" y1="{ry}" x2="{x0+col0}" y2="{ry+row_h}" stroke="#e3d7c3"/>')
        parts.append(f'<line x1="{x0+col0+col1}" y1="{ry}" x2="{x0+col0+col1}" y2="{ry+row_h}" stroke="#d5e6df"/>')
        parts.append(f'<text x="{x0+18}" y="{ry+29}" font-size="12.5" font-family="Arial">{safe_label(label)}</text>')
        if left_val:
            parts.append(f'<text x="{x0+col0+24}" y="{ry+29}" font-size="16" font-family="Georgia">{safe_label(left_val)}</text>')
        if right_val:
            parts.append(f'<text x="{x0+col0+col1+24}" y="{ry+29}" font-size="16" font-family="Georgia">{safe_label(right_val)}</text>')

    cloud_top = y0 + header_h + row_h * len(rows) + 24
    cloud_h = 250
    parts.append(f'<rect x="{x0+col0+18}" y="{cloud_top}" width="{col1-36}" height="{cloud_h}" rx="16" fill="#fff7ec" stroke="#e7d3b9"/>')
    parts.append(f'<rect x="{x0+col0+col1+18}" y="{cloud_top}" width="{col2-36}" height="{cloud_h}" rx="16" fill="#f2fbf7" stroke="#cfe5dd"/>')

    def draw_word_cluster(items: list[tuple[str, int]], panel_x: int, panel_w: int, palette: list[str]) -> None:
        positions = [
            (0.50, 0.22), (0.30, 0.40), (0.70, 0.38), (0.52, 0.56),
            (0.25, 0.64), (0.74, 0.63), (0.36, 0.80), (0.64, 0.79),
        ]
        max_count = max((count for _, count in items), default=1)
        cx = panel_x + panel_w / 2
        cy = cloud_top + cloud_h / 2
        for radius, opacity in [(105, 0.18), (72, 0.28), (42, 0.42)]:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="#ffffff" fill-opacity="{opacity}" stroke="none"/>')
        for idx, (tool, count) in enumerate(items):
            px = panel_x + positions[idx][0] * panel_w
            py = cloud_top + positions[idx][1] * cloud_h
            size = 15 + int(16 * count / max_count)
            rotate = -8 if idx % 3 == 0 else (8 if idx % 4 == 0 else 0)
            color = palette[idx % len(palette)]
            parts.append(
                f'<text x="{px:.1f}" y="{py:.1f}" text-anchor="middle" font-size="{size}" font-family="Arial" font-weight="600" fill="{color}" transform="rotate({rotate} {px:.1f} {py:.1f})">{safe_label(tool)}</text>'
            )

    draw_word_cluster(left_tools, x0 + col0 + 18, col1 - 36, ["#8f5324", "#c7782d", "#7a4a1c", "#d99f5c"])
    draw_word_cluster(right_tools, x0 + col0 + col1 + 18, col2 - 36, ["#1f6f63", "#2a9d8f", "#335c67", "#5aa9a1"])
    save_svg(path, parts)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_data()
    adopted = df[df["is_adopted"]].copy()
    correlations = correlation_table(adopted)
    low_full, high_not_full = contradiction_tables(df)
    maturity_summary = maturity_integration_summary(df)
    perf_summary = performance_summary(df)
    tool_summary = tool_level_summary(df)


if __name__ == "__main__":
    main()
