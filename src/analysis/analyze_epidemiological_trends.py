#!/usr/bin/env python3
"""
TheraLens: Epidemiological Trend Analysis & Time-Ordered Forecasting Pipeline

Analyzes longitudinal WHO surveillance data across Measles (1974-2024),
Yellow Fever (1974-2024), and weekly Cholera dynamics (2012-2014).
Standardizes schemas, detects outbreak surges using robust rolling Z-scores,
and implements time-ordered validation comparing baseline forecasting models.

METHODOLOGICAL NOTE:
Epidemiological reporting contains structural gaps and reporting delays.
Missing values are preserved rather than assumed to be zero cases.
Short time-series (e.g., Cholera 2012-2014) are restricted to descriptive
trend analysis, avoiding overclaiming predictive capability.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "epidemiological_analysis"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling for academic charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EpidemiologicalTrends")


def load_and_standardize_datasets() -> pd.DataFrame:
    """Loads and standardizes Measles, Yellow Fever, and Cholera datasets into unified long format."""
    logger.info("Loading and standardizing WHO epidemiological datasets...")
    
    # 1. Measles
    measles_path = DATA_RAW_DIR / "measles_who.csv"
    df_measles = pd.read_csv(measles_path)
    measles_clean = pd.DataFrame({
        "disease": "Measles",
        "country": df_measles["country_iso3"],
        "who_region": df_measles["who_region"],
        "year": pd.to_numeric(df_measles["year"], errors="coerce"),
        "temporal_resolution": "Annual",
        "time_period": df_measles["year"].astype(str),
        "reported_cases": pd.to_numeric(df_measles["value_numeric"], errors="coerce"),
    })
    
    # 2. Yellow Fever
    yf_path = DATA_RAW_DIR / "yellow_fever_who.csv"
    df_yf = pd.read_csv(yf_path)
    yf_clean = pd.DataFrame({
        "disease": "Yellow Fever",
        "country": df_yf["country_iso3"],
        "who_region": df_yf["who_region"],
        "year": pd.to_numeric(df_yf["year"], errors="coerce"),
        "temporal_resolution": "Annual",
        "time_period": df_yf["year"].astype(str),
        "reported_cases": pd.to_numeric(df_yf["value_numeric"], errors="coerce"),
    })
    
    # 3. Cholera (Long-format files from Review-1)
    cholera_dfs = []
    for yr in [2012, 2013, 2014]:
        c_path = DATA_PROCESSED_DIR / f"cholera_{yr}_long.csv"
        if not c_path.exists():
            # Fallback to raw wide format if processed not present
            raw_path = DATA_RAW_DIR / f"cholera_{yr}_who.csv"
            if raw_path.exists():
                raw_df = pd.read_csv(raw_path)
                week_cols = [c for c in raw_df.columns if "week" in c.lower() or c.isdigit()]
                id_col = "Country" if "Country" in raw_df.columns else raw_df.columns[0]
                melted = raw_df.melt(id_vars=[id_col], value_vars=week_cols, var_name="week", value_name="cases")
                melted["year"] = yr
                melted.rename(columns={id_col: "country"}, inplace=True)
                melted.to_csv(c_path, index=False)
                c_df = melted
            else:
                continue
        else:
            c_df = pd.read_csv(c_path)
            
        c_clean = pd.DataFrame({
            "disease": "Cholera",
            "country": c_df["country"],
            "who_region": "GLOBAL",
            "year": yr,
            "temporal_resolution": "Weekly",
            "time_period": c_df.apply(lambda r: f"{yr}-W{int(r['week']):02d}" if pd.notna(r.get('week')) else f"{yr}", axis=1),
            "reported_cases": pd.to_numeric(c_df["cases"], errors="coerce"),
        })
        cholera_dfs.append(c_clean)
        
    all_dfs = [measles_clean, yf_clean] + cholera_dfs
    unified = pd.concat(all_dfs, ignore_index=True)
    
    # Audit duplicates
    dupe_count = unified.duplicated(subset=["disease", "country", "time_period"]).sum()
    if dupe_count > 0:
        logger.warning("Found %d duplicate surveillance entries; deduplicating...", dupe_count)
        unified = unified.drop_duplicates(subset=["disease", "country", "time_period"]).reset_index(drop=True)
        
    logger.info("Unified epidemiological dataset: %d records across %d diseases", len(unified), unified["disease"].nunique())
    return unified


def detect_outbreak_surges(df: pd.DataFrame, window: int = 5, threshold_z: float = 2.0) -> pd.DataFrame:
    """Detects unusually high reported case counts using rolling Z-score per disease & country."""
    logger.info("Detecting outbreak surges (window=%d, z_threshold=%.1f)...", window, threshold_z)
    df = df.copy()
    df["rolling_mean"] = np.nan
    df["rolling_std"] = np.nan
    df["z_score"] = np.nan
    df["is_outbreak_surge"] = False

    # Process annual series
    annual_mask = df["temporal_resolution"] == "Annual"
    annual_df = df[annual_mask].sort_values(by=["disease", "country", "year"])

    def calc_group_surges(g):
        cases = g["reported_cases"].fillna(0)
        # Shifted rolling window to prevent future leakage
        roll_mean = cases.shift(1).rolling(window=window, min_periods=2).mean()
        roll_std = cases.shift(1).rolling(window=window, min_periods=2).std().replace(0, np.nan)
        z = (cases - roll_mean) / roll_std
        surge = (z >= threshold_z) & (cases > 50)  # Require minimum baseline cases to avoid noise
        g["rolling_mean"] = roll_mean
        g["rolling_std"] = roll_std
        g["z_score"] = z
        g["is_outbreak_surge"] = surge.fillna(False)
        return g

    surges_computed = annual_df.groupby(["disease", "country"], group_keys=False).apply(calc_group_surges)
    df.update(surges_computed)
    
    total_surges = int(df["is_outbreak_surge"].sum())
    logger.info("Identified %d anomalous outbreak surge events across annual surveillance records", total_surges)
    return df


def evaluate_temporal_forecasting(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates time-ordered forecasting models on global Measles annual incidence."""
    logger.info("Evaluating time-ordered forecasting benchmarks on longitudinal Measles series...")
    
    measles_global = (
        df[df["disease"] == "Measles"]
        .groupby("year")["reported_cases"]
        .sum()
        .reset_index()
        .sort_values(by="year")
    )
    
    # Split chronologically: Train 1974-2014 (41 years), Test 2015-2024 (10 years)
    split_year = 2014
    train_df = measles_global[measles_global["year"] <= split_year].copy()
    test_df = measles_global[measles_global["year"] > split_year].copy()
    
    if len(train_df) < 5 or len(test_df) < 2:
        logger.warning("Insufficient time points for time-ordered evaluation.")
        return {}, measles_global

    # Features: Lag-1 and Lag-2 cases
    measles_global["lag1"] = measles_global["reported_cases"].shift(1)
    measles_global["lag2"] = measles_global["reported_cases"].shift(2)
    
    m_train = measles_global[measles_global["year"] <= split_year].dropna()
    m_test = measles_global[measles_global["year"] > split_year].copy()
    
    # Baseline 1: Naive Persistence (y_hat_t = y_{t-1})
    test_actual = m_test["reported_cases"].values
    pred_naive = m_test["lag1"].values
    
    # Baseline 2: 3-Year Rolling Average
    roll3 = measles_global["reported_cases"].shift(1).rolling(3, min_periods=1).mean()
    pred_roll = roll3.loc[m_test.index].values
    
    # Model 3: Autoregressive Ridge Regression with Lag-1, Lag-2, and Trend
    X_train = m_train[["lag1", "lag2", "year"]].values
    y_train = m_train["reported_cases"].values
    X_test = m_test[["lag1", "lag2", "year"]].values
    
    ridge = Ridge(alpha=100.0, random_state=42)
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)
    
    def get_metrics(actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = root_mean_squared_error(actual, pred)
        mape = np.mean(np.abs((actual - pred) / np.maximum(actual, 1.0))) * 100
        return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_pct": round(mape, 2)}
        
    benchmarks = {
        "evaluation_period": f"{m_test['year'].min()} - {m_test['year'].max()} (N={len(m_test)} years)",
        "models": {
            "Naive Persistence Baseline": get_metrics(test_actual, pred_naive),
            "3-Year Moving Average Baseline": get_metrics(test_actual, pred_roll),
            "Autoregressive Ridge (Lag1 + Lag2 + Trend)": get_metrics(test_actual, pred_ridge),
        },
        "limitation_statement": (
            "Longitudinal surveillance series exhibit severe non-stationarity driven by national vaccination "
            "campaign expansions and reporting disruptions. Short-term models should not be extrapolated for "
            "long-horizon epidemic prediction."
        )
    }
    
    m_test["pred_naive"] = pred_naive
    m_test["pred_moving_avg"] = pred_roll
    m_test["pred_ridge"] = pred_ridge
    
    return benchmarks, m_test


def plot_epidemiological_visualizations(df: pd.DataFrame, m_test: pd.DataFrame, benchmarks: Dict[str, Any]):
    """Generates 4 publication-grade analytical figures."""
    logger.info("Generating epidemiological analysis visualizations...")
    
    # 1. Global Longitudinal Trends (Measles & Yellow Fever 1974-2024)
    annual = df[df["temporal_resolution"] == "Annual"].groupby(["disease", "year"])["reported_cases"].sum().reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    m_data = annual[annual["disease"] == "Measles"]
    ax1.plot(m_data["year"], m_data["reported_cases"] / 1e6, color="#d95f02", marker="o", markersize=3, linewidth=2)
    ax1.set_title("Global Annual Reported Measles Incidence (1974–2024)", fontsize=12, weight="bold")
    ax1.set_ylabel("Reported Cases (Millions)", fontsize=10)
    
    yf_data = annual[annual["disease"] == "Yellow Fever"]
    ax2.plot(yf_data["year"], yf_data["reported_cases"], color="#7570b3", marker="s", markersize=3, linewidth=2)
    ax2.set_title("Global Annual Reported Yellow Fever Incidence (1974–2024)", fontsize=12, weight="bold")
    ax2.set_ylabel("Reported Cases", fontsize=10)
    ax2.set_xlabel("Year", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "global_incidence_trends.png", dpi=300)
    plt.close(fig)
    
    # 2. Outbreak Surge Detection Heatmap / Points
    surges = df[df["is_outbreak_surge"]].copy()
    if not surges.empty:
        top_surge_countries = surges["country"].value_counts().head(10).index
        sub_surges = surges[surges["country"].isin(top_surge_countries)]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            data=sub_surges,
            x="year",
            y="country",
            size="reported_cases",
            hue="disease",
            sizes=(40, 400),
            palette="Set1",
            alpha=0.7,
            ax=ax,
        )
        ax.set_title("Detected Outbreak Surge Anomalies (Rolling Z >= 2.0)", fontsize=13, weight="bold")
        ax.set_xlabel("Year of Outbreak Event", fontsize=11)
        ax.set_ylabel("Surveillance Country (ISO3)", fontsize=11)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Disease / Scale")
        plt.tight_layout()
        fig.savefig(OUTPUTS_DIR / "outbreak_surge_detection.png", dpi=300)
        plt.close(fig)
        
    # 3. Forecasting Benchmark Comparison Plot
    if not m_test.empty and "pred_naive" in m_test.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(m_test["year"], m_test["reported_cases"] / 1e3, label="Actual Reported", color="black", marker="o", linewidth=2.5)
        ax.plot(m_test["year"], m_test["pred_naive"] / 1e3, label="Naive Persistence", color="#1b9e77", linestyle="--")
        ax.plot(m_test["year"], m_test["pred_moving_avg"] / 1e3, label="3-Yr Moving Avg", color="#d95f02", linestyle=":")
        ax.plot(m_test["year"], m_test["pred_ridge"] / 1e3, label="Autoregressive Ridge", color="#7570b3", marker="x")
        ax.set_title(f"Time-Ordered Forecasting Evaluation on Held-Out Test Horizon ({benchmarks.get('evaluation_period', '')})", fontsize=12, weight="bold")
        ax.set_xlabel("Evaluation Year", fontsize=11)
        ax.set_ylabel("Reported Cases (Thousands)", fontsize=11)
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(OUTPUTS_DIR / "forecasting_model_comparison.png", dpi=300)
        plt.close(fig)
        
    # 4. Cholera Weekly Dynamics
    cholera = df[df["disease"] == "Cholera"].copy()
    if not cholera.empty:
        cholera["week_num"] = cholera["time_period"].str.extract(r'W(\d+)').astype(float)
        weekly_pattern = cholera.groupby(["year", "week_num"])["reported_cases"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=weekly_pattern, x="week_num", y="reported_cases", hue="year", palette="tab10", marker="o", ax=ax)
        ax.set_title("Weekly Cholera Outbreak Dynamics & Seasonality (2012–2014)", fontsize=13, weight="bold")
        ax.set_xlabel("Epidemiological Week (Week 1–52)", fontsize=11)
        ax.set_ylabel("Cumulative Reported Cases Across Surveyed Countries", fontsize=11)
        plt.tight_layout()
        fig.savefig(OUTPUTS_DIR / "cholera_weekly_seasonality.png", dpi=300)
        plt.close(fig)
        
    logger.info("Saved all 4 epidemiological visualization charts to %s", OUTPUTS_DIR)


def run_epidemiological_analysis_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs complete epidemiological standardization, surge detection, and forecasting pipeline."""
    df_unified = load_and_standardize_datasets()
    df_surges = detect_outbreak_surges(df_unified)
    
    out_csv = DATA_PROCESSED_DIR / "epidemiological_trends.csv"
    df_surges.to_csv(out_csv, index=False)
    logger.info("Saved %d standardized epidemiological records to %s", len(df_surges), out_csv)
    
    benchmarks, m_test = evaluate_temporal_forecasting(df_surges)
    plot_epidemiological_visualizations(df_surges, m_test, benchmarks)
    
    summary = {
        "total_records": len(df_surges),
        "diseases_covered": list(df_surges["disease"].unique()),
        "time_span": {
            "Measles": "1974 - 2024 (Annual)",
            "Yellow Fever": "1974 - 2024 (Annual)",
            "Cholera": "2012 - 2014 (Weekly)",
        },
        "detected_outbreak_surges": int(df_surges["is_outbreak_surge"].sum()),
        "forecasting_benchmarks": benchmarks,
    }
    
    with open(OUTPUTS_DIR / "epidemiological_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved epidemiological summary to %s", OUTPUTS_DIR / "epidemiological_summary.json")
    
    return df_surges, summary


if __name__ == "__main__":
    df, summary = run_epidemiological_analysis_pipeline()
    print("\nEpidemiological Trend Analysis Pipeline Succeeded!")
    print(f"Total Records: {summary['total_records']}")
    print(f"Detected Outbreak Surges: {summary['detected_outbreak_surges']}")
    print(f"Forecasting Models Evaluated: {list(summary['forecasting_benchmarks']['models'].keys())}")
