from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd



ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "original_data.xlsx"
OUTPUT_FILE = ROOT / "The final results.docx"


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

    frame["budget_group"] = frame["Budget Category"].replace(
        {
            "Moderate": "Medium+",
            "Moderate-Large": "Medium+",
            "Large": "Medium+",
            "Huge": "Medium+",
        }
    )
    excluded = {
        "Institutional Group",
        "Institution",
        "Location of Respondent",
        "Area Category",
        "Digital Tools Adopted",
        "Awareness Status",
        "System Integration Level",
        "Technological Maturity",
        "Performance Category",
    }
    kept = [column for column in frame.columns if column not in excluded]
    frame = frame[kept].copy()
    return frame


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    x1 = add_intercept(x)
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
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return c * (1 + (x * x) / df) ** (-(df + 1) / 2)


def t_cdf(x: float, df: int, grid_size: int = 80000) -> float:
    if x == 0:
        return 0.5
    sign = 1 if x > 0 else -1
    grid = np.linspace(0, abs(x), grid_size + 1)
    values = np.array([t_pdf(v, df) for v in grid])
    area = np.trapezoid(values, grid)
    return 0.5 + sign * area


def two_sided_t_pvalue(t_stat: float, df: int) -> float:
    return 2 * (1 - t_cdf(abs(float(t_stat)), df))


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


def model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for column in ["Awareness Score", "Number of Tools Adopted", "Integration Score", "Maturity Score"]:
        df[f"{column}_c"] = df[column] - df[column].mean()

    x = pd.concat(
        [
            df[["Awareness Score_c", "Number of Tools Adopted_c", "Integration Score_c", "Maturity Score_c"]],
            pd.get_dummies(df["budget_group"], prefix="budget", drop_first=True, dtype=float),
            pd.DataFrame(
                {
                    "tools_x_integration": df["Number of Tools Adopted_c"] * df["Integration Score_c"],
                    "integration_x_maturity": df["Integration Score_c"] * df["Maturity Score_c"],
                }
            ),
        ],
        axis=1,
    )
    y = df["Performance Score"].to_numpy(float)
    beta, fitted, residuals, mse, xtx_inv = fit_ols(x.to_numpy(float), y)
    n = len(y)
    p = x.shape[1] + 1
    df_resid = n - p
    r2 = r2_score(y, fitted)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    r_value = math.sqrt(r2)

    se = np.sqrt(np.diag(mse * xtx_inv))
    t_stats = beta / se
    p_values = np.array([two_sided_t_pvalue(value, df_resid) for value in t_stats])

    x1 = add_intercept(x.to_numpy(float))
    hat = x1 @ xtx_inv @ x1.T
    leverage = np.diag(hat)
    omega = np.diag((residuals / (1.0 - leverage)) ** 2)
    hc3 = xtx_inv @ x1.T @ omega @ x1 @ xtx_inv
    hc3_se = np.sqrt(np.diag(hc3))

    coef_table = pd.DataFrame(
        {
            "term": ["Intercept"] + list(x.columns),
            "coefficient": np.round(beta, 4),
            "std_error": np.round(se, 4),
            "hc3_std_error": np.round(hc3_se, 4),
            "t_stat": np.round(t_stats, 4),
            "p_value": np.round(p_values, 5),
        }
    )

    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2500):
        idx = rng.integers(0, n, size=n)
        xb = x.to_numpy(float)[idx]
        yb = y[idx]
        b, *_ = fit_ols(xb, yb)
        boot.append(b)
    boot = np.asarray(boot)
    coef_table["bootstrap_ci_low"] = np.round(np.quantile(boot, 0.025, axis=0), 4)
    coef_table["bootstrap_ci_high"] = np.round(np.quantile(boot, 0.975, axis=0), 4)

    z = (residuals - residuals.mean()) / residuals.std(ddof=0)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    jb = float(n / 6.0 * (skew**2 + (kurt - 3.0) ** 2 / 4.0))
    _, bp_fit, _, _, _ = fit_ols(x.to_numpy(float), residuals**2)
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
    dw = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))
    cooks_d = (residuals**2 / (p * mse)) * (leverage / (1.0 - leverage) ** 2)

    diagnostics = pd.DataFrame(
        [
            ["R", round(r_value, 4)],
            ["R-squared", round(r2, 4)],
            ["Adjusted R-squared", round(adj_r2, 4)],
            ["Jarque-Bera", round(jb, 4)],
            ["Breusch-Pagan LM", round(bp_lm, 4)],
            ["BP 95% critical value", round(critical_chi_square_95(x.shape[1]), 4)],
            ["RESET F", round(reset_f, 4)],
            ["Durbin-Watson", round(dw, 4)],
            ["Max leverage", round(float(leverage.max()), 4)],
            ["Max Cook's D", round(float(cooks_d.max()), 4)],
        ],
        columns=["Statistic", "Value"],
    )

    desc = df[
        ["Awareness Score", "Number of Tools Adopted", "Integration Score", "Maturity Score", "Performance Score"]
    ].describe().round(3)
    cat_counts = pd.DataFrame({"Budget Group": df["budget_group"].value_counts()})
    return x, y, coef_table, diagnostics, desc, cat_counts

if __name__ == "__main__":
    pass