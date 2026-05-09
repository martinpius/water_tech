from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "original_data.xlsx"
OUTPUT_DIR = ROOT / "final_folder"


def parse_assigned_numeric(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value,
                  (int, 
                   float, 
                   np.integer, 
                   np.floating)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"Assigned\s*([0-9.]+)", text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except ValueError:
        return None


def load_data() -> pd.DataFrame:
    raw = pd.read_excel(SOURCE_FILE, header=None)
    frame = raw.iloc[3:].copy().reset_index(drop=True)
    frame.columns = raw.iloc[2]
    frame = frame.loc[:, [column for column in frame.columns if not pd.isna(column)]]
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].ffill()

    for column in [
        "Awareness Score",
        "Number of Tools Adopted",
        "Integration Score",
        "Performance Score",
        "Maturity Score",
    ]:
        frame[column] = frame[column].map(parse_assigned_numeric)

    frame["budget_group"] = np.where(frame[
        "Budget Category"].astype(str).eq("Small"), 
                                     "Small", "Medium+")
    frame["size_group"] = np.where(frame[
        "Institution Size"].astype(str).eq("Small"), 
                                   "Small", "Large+")
    frame["area_group"] = frame["Area Category"].astype(str)
    return frame


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,
                                        np.ndarray, np.ndarray, 
                                        float, np.ndarray]:
    x1 = np.column_stack([np.ones(len(x)), x])
    xtx_inv = np.linalg.pinv(x1.T @ x1)
    beta = xtx_inv @ x1.T @ y
    fitted = x1 @ beta
    residuals = y - fitted
    mse = float(np.sum(residuals**2) / max(1, len(y) - x1.shape[1]))
    return beta, fitted, residuals, mse, xtx_inv


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    tss = float(np.sum((y - y.mean()) ** 2))
    rss = float(np.sum((y - pred) ** 2))
    return 1.0 - rss / tss


def t_pdf(x: float, df: int) -> float:
    coeff = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return coeff * (1 + (x * x) / df) ** (-(df + 1) / 2)


def t_cdf(x: float,
          df: int, 
          grid_size: int = 40000) -> float:
    if x == 0:
        return 0.5
    sign = 1 if x > 0 else -1
    grid = np.linspace(0, abs(x), grid_size + 1)
    values = np.array([t_pdf(v, df) for v in grid])
    area = np.trapezoid(values, grid)
    return 0.5 + sign * area


def two_sided_t_pvalue(t_stat: float,
                       df: int) -> float:
    return 2 * (1 - t_cdf(abs(float(t_stat)),
                          df))


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Performance Score",
        "Awareness Score",
        "Number of Tools Adopted",
        "Integration Score",
        "Maturity Score",
        "budget_group",
        "size_group",
        "area_group",
    ]
    model_df = df[columns].dropna().copy()
    for column in [
        "Awareness Score",
        "Number of Tools Adopted",
        "Integration Score",
        "Maturity Score",
    ]:
        model_df[f"{column}_c"] = model_df[column] - model_df[column].mean()
    model_df["tools_x_integration"] = (
        model_df["Number of Tools Adopted_c"] * model_df["Integration Score_c"]
    )
    model_df["integration_x_maturity"] = (
        model_df["Integration Score_c"] * model_df["Maturity Score_c"]
    )
    model_df["budget_small"] = (model_df["budget_group"] == "Small").astype(float)
    model_df["size_small"] = (model_df["size_group"] == "Small").astype(float)
    model_df["rural"] = (model_df["area_group"] == "Rural").astype(float)
    return model_df


def build_design_matrix(model_df: pd.DataFrame,
                        include_size: bool, 
                        include_area: bool) -> pd.DataFrame:
    predictors = [
        "Awareness Score_c",
        "Number of Tools Adopted_c",
        "Integration Score_c",
        "Maturity Score_c",
        "budget_small",
        "tools_x_integration",
        "integration_x_maturity",
    ]
    if include_size:
        predictors.append("size_small")
    if include_area:
        predictors.append("rural")
    return model_df[predictors].copy()

# Get the summary for the fitted model

def model_summary(name: str, 
                  x: pd.DataFrame, 
                  y: np.ndarray) -> tuple[dict[str, 
                float | str], pd.DataFrame, 
                np.ndarray, np.ndarray]:
    beta, fitted, residuals, mse, xtx_inv = fit_ols(x.to_numpy(float), y)
    n = len(y)
    p = x.shape[1] + 1
    df_resid = n - p
    rss = float(np.sum(residuals**2))
    r2 = r2_score(y, fitted)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    aic = n * math.log(rss / n) + 2 * p
    bic = n * math.log(rss / n) + math.log(n) * p

    se = np.sqrt(np.diag(mse * xtx_inv))
    t_stats = beta / se
    p_values = np.array([two_sided_t_pvalue(value, df_resid) for value in t_stats])
    coef_table = pd.DataFrame(
        {
            "term": ["Intercept"] + list(x.columns),
            "coefficient": np.round(beta, 4),
            "std_error": np.round(se, 4),
            "t_stat": np.round(t_stats, 4),
            "p_value": np.round(p_values, 5),
        }
    )
    rng = np.random.default_rng(1234)
    boot = []
    x_np = x.to_numpy(float)
    for _ in range(2500):
        idx = rng.integers(0, n, size=n)
        b, *_ = fit_ols(x_np[idx], y[idx])
        boot.append(b)
    boot = np.asarray(boot)
    coef_table["bootstrap_ci_low"] = np.round(np.quantile(boot, 
                            0.025, axis=0), 4)
    coef_table["bootstrap_ci_high"] = np.round(np.quantile(boot,
                    0.975, axis=0), 4)

    overview = {
        "model": name,
        "predictors": ", ".join(x.columns),
        "r": round(math.sqrt(max(r2, 0.0)), 4),
        "r2": round(r2, 4),
        "adj_r2": round(adj_r2, 4),
        "aic": round(aic, 4),
        "bic": round(bic, 4),
    }
    return overview, coef_table, fitted, residuals

# Set the critical chi-squares

def critical_chi_square_95(df: int) -> float:
    table = {
        1: 3.841,
        2: 5.991,
        3: 7.815,
        4: 9.488,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
        10: 18.307,
    }
    return table.get(df, float("nan"))


# Computing modelmfits stats

def diagnostics_table(x: pd.DataFrame, 
                      y: np.ndarray) -> pd.DataFrame:
    x_np = x.to_numpy(float)
    x1 = np.column_stack([np.ones(len(x_np)), x_np])
    _, fitted, residuals, _, _ = fit_ols(x_np, y)
    n = len(y)
    p = x1.shape[1]
    r2 = r2_score(y, fitted)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)

    z = (residuals - residuals.mean()) / max(residuals.std(ddof=0), 1e-9)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    jb = float(n / 6.0 * (skew**2 + (kurt - 3.0) ** 2 / 4.0))

    _, bp_fit, _, _, _ = fit_ols(x_np, residuals**2)
    sse = float(np.sum(((residuals**2) - bp_fit) ** 2))
    sst = float(np.sum(((residuals**2) - np.mean(residuals**2)) ** 2))
    bp_r2 = 1.0 - sse / sst if sst else 0.0
    bp_lm = n * bp_r2

    zmat = np.column_stack([x1, fitted**2, fitted**3])
    beta_reset = np.linalg.pinv(zmat.T @ zmat) @ zmat.T @ y
    fitted_reset = zmat @ beta_reset
    rss_restricted = float(np.sum(residuals**2))
    rss_unrestricted = float(np.sum((y - fitted_reset) ** 2))
    reset_f = ((rss_restricted - rss_unrestricted) / 2.0) / (
        rss_unrestricted / max(1, n - zmat.shape[1])
    )

    rows = [
        ("R", round(math.sqrt(max(r2, 0.0)), 4)),
        ("R-Squared", round(r2, 4)),
        ("Adjusted R-Squared", round(adj_r2, 4)),
        ("Jarque-Bera", round(jb, 4)),
        ("Breusch-Pagan LM", round(bp_lm, 4)),
        ("Breusch-Pagan 95% Critical", round(critical_chi_square_95(x_np.shape[1]), 4)),
        ("RESET F", round(reset_f, 4)),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"])

# Assessing normality assumption

def draw_residual_plot(fitted: np.ndarray,
                       residuals: np.ndarray, 
                       path: Path) -> None:
    width, height = 860, 560
    margin = 80
    image = Image.new("RGB", (width, height), "#fcfbf7")
    draw = ImageDraw.Draw(image)

    x_lo, x_hi = float(fitted.min()), float(fitted.max())
    y_lo, y_hi = float(residuals.min()), float(residuals.max())
    pad_y = 0.05 * max(1.0, y_hi - y_lo)
    y_lo -= pad_y
    y_hi += pad_y

    def x_map(v: float) -> float:
        return margin + (v - x_lo) * (width - 2 * margin) / max(1e-9, x_hi - x_lo)

    def y_map(v: float) -> float:
        return height - margin - (v - y_lo) * (height - 2 * margin) / max(1e-9, y_hi - y_lo)

    draw.line((margin, height - margin, width - margin, 
               height - margin),  width=2)
    draw.line((margin, margin, margin, height - margin),
              fill="#222222", width=2)

    zero_y = y_map(0.0)
    draw.line((margin, zero_y, width - margin, zero_y),
               width=2)

    for value in np.linspace(x_lo, x_hi, 5):
        x_pos = x_map(float(value))
        draw.line((x_pos, margin, x_pos, height - margin),
                width=1)
        draw.text((x_pos - 10, height - margin + 10),
                  f"{value:.2f}")
    for value in np.linspace(y_lo, y_hi, 5):
        y_pos = y_map(float(value))
        draw.line((margin, y_pos, width - margin, y_pos), width=1)
        draw.text((margin - 48, y_pos - 6), f"{value:.2f}")

    for x_val, y_val in zip(fitted, residuals):
        cx = x_map(float(x_val))
        cy = y_map(float(y_val))
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), width=1)

    draw.text((width / 2 - 45, height - 35), "Fitted values")
    y_label = Image.new("RGBA", (120, 24), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Residuals")
    rotated = y_label.rotate(90, expand=True)
    image.paste(rotated, (12, int(height / 2 - 50)), rotated)
    image.save(path)

# Plotting diagnostics for our fitted model

def draw_qq_plot(residuals: np.ndarray, path: Path) -> None:
    width, height = 860, 560
    margin = 80
    image = Image.new("RGB", (width, height), "#fcfbf7")
    draw = ImageDraw.Draw(image)


    ordered = np.sort(residuals)
    n = len(ordered)
    theoretical = np.array([NormalDist().inv_cdf((i - 0.5) / n) for i in range(1, n + 1)])
    sample_std = ordered.std(ddof=1) if n > 1 else 1.0
    standardized = (ordered - ordered.mean()) / max(sample_std, 1e-9)

    x_lo, x_hi = float(theoretical.min()), float(theoretical.max())
    y_lo, y_hi = float(standardized.min()), float(standardized.max())
    pad = 0.15
    x_lo -= pad
    x_hi += pad
    y_lo -= pad
    y_hi += pad

    def x_map(v: float) -> float:
        return margin + (v - x_lo) * (width - 2 * margin) / max(1e-9, x_hi - x_lo)

    def y_map(v: float) -> float:
        return height - margin - (v - y_lo) * (height - 2 * margin) / max(1e-9, y_hi - y_lo)

    draw.line((margin, height - margin, width - margin, height - margin),  width=2)
    draw.line((margin, margin, margin, height - margin),  width=2)

    for value in np.linspace(x_lo, x_hi, 5):
        x_pos = x_map(float(value))
        draw.line((x_pos, margin, x_pos, height - margin), width=1)
        draw.text((x_pos - 12, height - margin + 10), f"{value:.1f}")
    for value in np.linspace(y_lo, y_hi, 5):
        y_pos = y_map(float(value))
        draw.line((margin, y_pos, width - margin, y_pos), width=1)
        draw.text((margin - 36, y_pos - 6), f"{value:.1f}")

    slope, intercept = np.polyfit(theoretical, standardized, 1)
    draw.line(
        (
            x_map(x_lo),
            y_map(intercept + slope * x_lo),
            x_map(x_hi),
            y_map(intercept + slope * x_hi),
        ),
        width=2,
    )

    for x_val, y_val in zip(theoretical, standardized):
        cx = x_map(float(x_val))
        cy = y_map(float(y_val))
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), width=1)

    draw.text((width / 2 - 82, height - 35), "Theoretical quantiles")
    y_label = Image.new("RGBA", (120, 24), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Sample quantiles")
    rotated = y_label.rotate(90, expand=True)
    image.paste(rotated, (12, int(height / 2 - 56)), rotated)
    image.save(path)



def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    model_df = prepare_model_data(df)
    y = model_df["Performance Score"].to_numpy(float)
    area_counts = model_df["area_group"].value_counts()

    models = []
    stored = {}
    for name, include_size, include_area in [
        ("Base", False, False),
        ("Base + Institution Size", True, False),
        ("Base + Institution Size + Area", True, True),
    ]:
        x = build_design_matrix(model_df, include_size=include_size, include_area=include_area)
        overview, coef_table, fitted, residuals = model_summary(name, x, y)
        models.append(overview)
        stored[name] = {"x": x, "coef": coef_table, "fitted": fitted, "residuals": residuals}

    comparison_df = pd.DataFrame(models).sort_values("adj_r2", ascending=False)
    chosen_model = "Base + Institution Size + Area"
    coef_table = stored[chosen_model]["coef"]
    fitted = stored[chosen_model]["fitted"]
    residuals = stored[chosen_model]["residuals"]
    x_chosen = stored[chosen_model]["x"]
    diagnostics_df = diagnostics_table(x_chosen, y)

    model_df.to_csv(OUTPUT_DIR / "analysis_data.csv", index=False)
    area_counts.rename_axis("area_group").reset_index(
        name="count").to_csv(OUTPUT_DIR / "location_count_check.csv", 
                             index=False)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", 
                         index=False)
    coef_table.to_csv(OUTPUT_DIR / "final_model_coefficients.csv", 
                      index=False)
    diagnostics_df.to_csv(OUTPUT_DIR / "final_model_fit_statistics.csv", 
                          index=False)

    draw_residual_plot(fitted, residuals, OUTPUT_DIR / "residuals_vs_fitted.png")
    draw_qq_plot(residuals, OUTPUT_DIR / "qq_plot.png")


if __name__ == "__main__":
    main()
