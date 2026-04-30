# src/project.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch, Rectangle

from config import (
    TARGET_COL,
    ID_COLS,
    TRAIN_FILE,
    TEST_FILE,
    STORE_FILE,
    CALENDAR_FILE,
    PROMO_FILE,
    EXTERNAL_FILE,
    VALID_DAYS,
    FEATURE_COLS_BASE,
    FEATURE_COLS_TS,
    CATEGORICAL_COLS,
)
from utils import (
    ensure_dir,
    save_csv,
    save_json,
    safe_divide,
    coefficient_of_variation,
    mean_absolute_percentage_error_safe,
)


"""
Starter code notes
------------------
1. Use only the CSV files provided in ./data. Do not regenerate data inside this script.
2. Do not introduce extra randomness, shuffling, sampling, or manual re-ordering unless a task explicitly asks for it.
3. Keep output file names, column names, and return types unchanged.
4. Unless a task explicitly asks you to fill / sort / round values, keep pandas default behavior.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--figure_dir", type=str, default="figures")
    return parser.parse_args()


############################################################################
######################### SECTION 1 LOAD DATA ##############################
############################################################################

def load_raw_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / TRAIN_FILE, parse_dates=["date"])

    # TASK 1 -----------------------------------------------------------------
    test = pd.read_csv(data_dir / TEST_FILE, parse_dates=["date"])
    stores = pd.read_csv(data_dir / STORE_FILE)
    calendar = pd.read_csv(data_dir / CALENDAR_FILE, parse_dates=["date"])
    promo = pd.read_csv(data_dir / PROMO_FILE, parse_dates=["date"])
    external = pd.read_csv(data_dir / EXTERNAL_FILE, parse_dates=["date"])
    # ------------------------------------------------------------------------

    return {
        "train": train,
        "test": test,
        "stores": stores,
        "calendar": calendar,
        "promo": promo,
        "external": external,
    }


def build_schema_summary(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in raw.items():
        # TASK 2 -----------------------------------------------------------------
        date_min = df["date"].min() if "date" in df.columns else pd.NaT
        date_max = df["date"].max() if "date" in df.columns else pd.NaT
        rows.append({
            "table_name": name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_missing_total": int(df.isna().sum().sum()),
            "date_min": date_min,
            "date_max": date_max,
        })
        # ------------------------------------------------------------------------
    return pd.DataFrame(rows)


############################################################################
######################### SECTION 2 MERGE & CLEAN ##########################
############################################################################

def merge_master(order_df, stores, calendar, promo, external):
    # TASK 3 -----------------------------------------------------------------
    df = order_df.merge(stores, on="store_id", how="left")
    df = df.merge(calendar, on="date", how="left")
    df = df.merge(promo, on=["store_id", "date", "category"], how="left")
    df = df.merge(external, on=["city", "date"], how="left")
    # ------------------------------------------------------------------------
    return df


def clean_master(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaning_log = []

    before_rows = len(df)
    df = df.drop_duplicates().copy()
    cleaning_log.append({
        "step": "drop_duplicates",
        "rows_before": before_rows,
        "rows_after": len(df),
    })

    if "promo_flag" not in df.columns:
        if "promo_depth" in df.columns:
            df["promo_flag"] = df["promo_depth"].fillna(0).gt(0).astype(int)
            cleaning_log.append({
                "step": "derive_promo_flag_from_promo_depth",
                "n_derived": int(len(df)),
            })
        else:
            df["promo_flag"] = 0
            cleaning_log.append({
                "step": "set_default_promo_flag",
                "n_derived": int(len(df)),
            })

    if "discount_rate" not in df.columns:
        if "promo_depth" in df.columns:
            df["discount_rate"] = df["promo_depth"].fillna(0)
            cleaning_log.append({
                "step": "derive_discount_rate_from_promo_depth",
                "n_derived": int(len(df)),
            })
        else:
            df["discount_rate"] = 0.0
            cleaning_log.append({
                "step": "set_default_discount_rate",
                "n_derived": int(len(df)),
            })

    fill_zero_cols = [
        "promo_flag", "coupon_flag", "promo_depth",
        "discount_rate", "stockout_minutes"
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            # TASK 4 -------------------------------------------------------------
            n_filled = int(df[col].isna().sum())
            df[col] = df[col].fillna(0)
            cleaning_log.append({
                "step": f"fillna_{col}",
                "n_filled": n_filled,
            })
            # --------------------------------------------------------------------

    if "discount_rate" in df.columns:
        # TASK 5 -----------------------------------------------------------------
        n_affected = int(((df["discount_rate"] < 0) | (df["discount_rate"] > 1)).sum())
        df["discount_rate"] = df["discount_rate"].clip(0, 1)
        cleaning_log.append({
            "step": "clip_discount_rate",
            "n_affected": n_affected,
        })
        # ------------------------------------------------------------------------

    if "stockout_minutes" in df.columns:
        # TASK 6 -----------------------------------------------------------------
        n_affected = int((df["stockout_minutes"] < 0).sum())
        df["stockout_minutes"] = df["stockout_minutes"].clip(lower=0)
        cleaning_log.append({
            "step": "clip_stockout_minutes",
            "n_affected": n_affected,
        })
        # ------------------------------------------------------------------------

    if "avg_delivery_time" in df.columns:
        # TASK 7 -----------------------------------------------------------------
        n_affected = int((df["avg_delivery_time"] <= 0).sum())
        df.loc[df["avg_delivery_time"] <= 0, "avg_delivery_time"] = np.nan
        cleaning_log.append({
            "step": "set_invalid_delivery_time_nan",
            "n_affected": n_affected,
        })
        # ------------------------------------------------------------------------

    df = df.sort_values(["store_id", "category", "date"]).reset_index(drop=True)
    log_df = pd.DataFrame(cleaning_log)
    return df, log_df


############################################################################
####################### SECTION 3 BUSINESS ANALYSIS ########################
############################################################################

def promo_lift_func(group: pd.DataFrame) -> float:
    promo_mean = group.loc[group["promo_flag"] == 1, TARGET_COL].mean()
    nonpromo_mean = group.loc[group["promo_flag"] == 0, TARGET_COL].mean()
    if pd.isna(nonpromo_mean) or nonpromo_mean == 0:
        return np.nan
    return promo_mean / nonpromo_mean - 1


def kpi_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    city_kpi = (
        df.groupby("city")
          .agg(
              avg_daily_orders=(TARGET_COL, "mean"),
              avg_gmv=("gmv", "mean"),
              avg_refund_rate=("refund_rate", "mean"),
              avg_delivery_time=("avg_delivery_time", "mean"),
          )
          .reset_index()
          .sort_values("avg_daily_orders", ascending=False)
    )

    # TASK 8 -----------------------------------------------------------------
    category_stats = (
        df.groupby("category")[TARGET_COL]
        .agg(["mean", "std"])
        .rename(columns={"mean": "orders_mean", "std": "orders_std"})
        .reset_index()
    )
    category_stats["orders_cv"] = category_stats["orders_std"] / category_stats["orders_mean"]
    # ------------------------------------------------------------------------

    # TASK 9 -----------------------------------------------------------------
    promo_lift = (
        df.groupby(["store_id", "category"])
        .apply(promo_lift_func)
        .reset_index(name="promo_lift")
    )
    # ------------------------------------------------------------------------

    # TASK 10 ----------------------------------------------------------------
    city_category_pivot = pd.pivot_table(
        df,
        values=TARGET_COL,
        index="city",
        columns="category",
        aggfunc="mean",
    ).reset_index()
    # ------------------------------------------------------------------------

    # TASK 11 ----------------------------------------------------------------
    store_alerts = (
        df.groupby("store_id")
        .agg(
            avg_orders=(TARGET_COL, "mean"),
            avg_stockout_minutes=("stockout_minutes", "mean"),
            avg_refund_rate=("refund_rate", "mean"),
            avg_gmv=("gmv", "mean"),
        )
        .reset_index()
    )
    p75_orders = store_alerts["avg_orders"].quantile(0.75)
    p75_stockout = store_alerts["avg_stockout_minutes"].quantile(0.75)
    store_alerts["high_orders_high_stockout_flag"] = (
        (store_alerts["avg_orders"] > p75_orders) &
        (store_alerts["avg_stockout_minutes"] > p75_stockout)
    ).astype(int)
    # ------------------------------------------------------------------------

    return {
        "city_kpi": city_kpi,
        "category_stats": category_stats,
        "promo_lift": promo_lift,
        "city_category_pivot": city_category_pivot,
        "store_alerts": store_alerts,
    }


############################################################################
##################### SECTION 4 TIME-SERIES FEATURES #######################
############################################################################

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["store_id", "category", "date"]).reset_index(drop=True)

    grp = df.groupby(["store_id", "category"])
    df["lag_1"] = grp[TARGET_COL].shift(1)
    df["lag_7"] = grp[TARGET_COL].shift(7)
    df["lag_14"] = grp[TARGET_COL].shift(14)

    # TASK 12 ----------------------------------------------------------------
    df["rolling_mean_7"] = grp[TARGET_COL].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df["rolling_std_7"] = grp[TARGET_COL].transform(
        lambda x: x.shift(1).rolling(7).std()
    )
    df["rolling_mean_28"] = grp[TARGET_COL].transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
    df["promo_last_7d_count"] = grp["promo_flag"].transform(
        lambda x: x.shift(1).rolling(7).sum()
    )
    df["stockout_last_7d_mean"] = grp["stockout_minutes"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    # ------------------------------------------------------------------------

    prev_promo_date = df["date"].where(grp["promo_flag"].shift(1).fillna(0).gt(0))
    prev_promo_date = prev_promo_date.groupby([df["store_id"], df["category"]]).ffill()
    days_since_last_promo = (df["date"] - prev_promo_date).dt.days
    cold_start_days = grp.cumcount() + 1
    df["days_since_last_promo"] = days_since_last_promo.fillna(cold_start_days).astype(float)

    return df


def build_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "feature_name": col,
            "dtype": str(df[col].dtype),
            "missing_ratio": float(df[col].isna().mean()),
        })
    return pd.DataFrame(rows)


############################################################################
########################### SECTION 5 MODELING #############################
############################################################################

def split_train_valid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["date"].max()
    valid_start = max_date - pd.Timedelta(days=VALID_DAYS - 1)

    train_df = df[df["date"] < valid_start].copy()
    valid_df = df[df["date"] >= valid_start].copy()
    return train_df, valid_df


def prepare_model_matrix(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str]):
    # TASK 13 ----------------------------------------------------------------
    all_cols = feature_cols + CATEGORICAL_COLS

    train_raw = train_df[all_cols].copy()
    valid_raw = valid_df[all_cols].copy()

    train_encoded = pd.get_dummies(train_raw, columns=CATEGORICAL_COLS, drop_first=True)
    valid_encoded = pd.get_dummies(valid_raw, columns=CATEGORICAL_COLS, drop_first=True)

    X_train = train_encoded
    X_valid = valid_encoded.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[TARGET_COL]
    y_valid = valid_df[TARGET_COL]
    # ------------------------------------------------------------------------

    return X_train, X_valid, y_train, y_valid


def run_baselines(valid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    baseline_1 = valid_df["lag_7"].fillna(valid_df["rolling_mean_7"])
    mae_1 = mean_absolute_error(valid_df[TARGET_COL], baseline_1)
    rmse_1 = mean_squared_error(valid_df[TARGET_COL], baseline_1) ** 0.5

    rows.append({
        "model_name": "baseline_lag7",
        "mae": mae_1,
        "rmse": rmse_1,
        "mape": mean_absolute_percentage_error_safe(valid_df[TARGET_COL], baseline_1),
    })

    baseline_2 = valid_df["rolling_mean_7"].fillna(valid_df["lag_7"])
    mae_2 = mean_absolute_error(valid_df[TARGET_COL], baseline_2)
    rmse_2 = mean_squared_error(valid_df[TARGET_COL], baseline_2) ** 0.5

    rows.append({
        "model_name": "baseline_roll7",
        "mae": mae_2,
        "rmse": rmse_2,
        "mape": mean_absolute_percentage_error_safe(valid_df[TARGET_COL], baseline_2),
    })

    return pd.DataFrame(rows)


def train_formal_model(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    feature_cols = FEATURE_COLS_BASE + FEATURE_COLS_TS

    # TASK 14 ----------------------------------------------------------------
    required_cols = feature_cols + [TARGET_COL]
    model_train = train_df.dropna(subset=required_cols).copy()
    model_valid = valid_df.dropna(subset=required_cols).copy()

    X_train, X_valid, y_train, y_valid = prepare_model_matrix(model_train, model_valid, feature_cols)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_valid)
    pred = np.clip(pred, 0, None)

    mae = mean_absolute_error(y_valid, pred)
    rmse = mean_squared_error(y_valid, pred) ** 0.5
    mape = mean_absolute_percentage_error_safe(y_valid, pred)

    metrics = pd.DataFrame([{
        "model_name": "linear_regression",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n_valid_rows": len(model_valid),
    }])
    # ------------------------------------------------------------------------

    pred_df = model_valid[ID_COLS + [TARGET_COL]].copy()
    pred_df["prediction"] = pred
    pred_df["abs_error"] = np.abs(pred_df[TARGET_COL] - pred_df["prediction"])

    return model, metrics, pred_df


def error_analysis(pred_df: pd.DataFrame, full_valid_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid_scored = pred_df.merge(
        full_valid_df.drop(columns=[TARGET_COL]),
        on=ID_COLS,
        how="left"
    )

    error_by_city = (
        valid_scored.groupby("city")["abs_error"]
        .mean()
        .reset_index(name="mae_by_city")
        .sort_values("mae_by_city", ascending=False)
    )

    error_by_category = (
        valid_scored.groupby("category")["abs_error"]
        .mean()
        .reset_index(name="mae_by_category")
        .sort_values("mae_by_category", ascending=False)
    )

    error_by_condition = (
        valid_scored.groupby(["promo_flag", "is_weekend"])["abs_error"]
        .mean()
        .reset_index(name="mean_abs_error")
    )

    top10_errors = (
        valid_scored.sort_values("abs_error", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return {
        "error_by_city": error_by_city,
        "error_by_category": error_by_category,
        "error_by_condition": error_by_condition,
        "top10_errors": top10_errors,
    }


############################################################################
######################### SECTION 6 VISUALIZATION ##########################
############################################################################

def plot_daily_orders_overview(daily_df: pd.DataFrame):
    # TASK 15 ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))
    fig_bg_color = "#ffffff"
    plot_bg_color = "#f3f3f3"
    fig.patch.set_facecolor(fig_bg_color)
    ax.set_facecolor(plot_bg_color)

    daily_df = daily_df.sort_values("date").copy()

    # Main lines
    ax.plot(
        daily_df["date"], daily_df["total_orders"],
        color="#4472C4", linewidth=1, alpha=0.85, label="Total Daily Orders", zorder=2
    )
    ax.plot(
        daily_df["date"], daily_df["rolling_mean_7"],
        color="#ED7D31", linewidth=2.5, label="7-Day Rolling Mean", zorder=3
    )

    # High-promo highlights: days where promo_share exceeds median
    promo_threshold = daily_df["promo_share"].median()
    y_min_data = daily_df["total_orders"].min()
    y_max_data = daily_df["total_orders"].max()
    ax.fill_between(
        daily_df["date"],
        y_min_data * 0.95,
        y_max_data * 1.05,
        where=daily_df["promo_share"] > promo_threshold,
        color="#70AD47", alpha=0.18, label="High Promo Period", zorder=1
    )

    # Axis limits with headroom for inset
    ax.set_ylim(y_min_data * 0.93, y_max_data * 1.18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Annotation 1: peak
    peak_idx = daily_df["total_orders"].idxmax()
    peak_row = daily_df.loc[peak_idx]
    ax.annotate(
        f"Peak: {int(peak_row['total_orders'])}",
        xy=(peak_row["date"], peak_row["total_orders"]),
        xytext=(30, 15), textcoords="offset points",
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
        zorder=5,
    )

    # Annotation 2: trough
    trough_idx = daily_df["total_orders"].idxmin()
    trough_row = daily_df.loc[trough_idx]
    ax.annotate(
        f"Trough: {int(trough_row['total_orders'])}",
        xy=(trough_row["date"], trough_row["total_orders"]),
        xytext=(30, -25), textcoords="offset points",
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
        zorder=5,
    )

    # Inset: last 28 days (top-right)
    ax_inset = ax.inset_axes([0.64, 0.52, 0.34, 0.43])
    ax_inset.set_facecolor(plot_bg_color)
    last_28 = daily_df.tail(28)
    ax_inset.plot(last_28["date"], last_28["total_orders"], color="#4472C4", linewidth=1)
    ax_inset.plot(last_28["date"], last_28["rolling_mean_7"], color="#ED7D31", linewidth=1.5)
    ax_inset.fill_between(
        last_28["date"],
        last_28["total_orders"].min() * 0.95,
        last_28["total_orders"].max() * 1.05,
        where=last_28["promo_share"] > promo_threshold,
        color="#70AD47", alpha=0.18,
    )
    ax_inset.set_title("Last 28 Days", fontsize=9)
    ax_inset.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_inset.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_inset.tick_params(labelsize=7)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.8)

    # Connector lines from main to inset
    ax.indicate_inset_zoom(ax_inset, edgecolor="gray")

    ax.set_title("Executive Overview of Daily Orders")
    ax.set_xlabel("Date")
    ax.set_ylabel("Orders")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    # ------------------------------------------------------------------------
    return fig


def plot_validation_diagnostics(valid_df: pd.DataFrame):
    # TASK 16 ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_bg_color = "#f3f3f3"
    ax.set_facecolor(plot_bg_color)

    if "abs_error" not in valid_df.columns:
        valid_df = valid_df.copy()
        valid_df["abs_error"] = np.abs(valid_df[TARGET_COL] - valid_df["prediction"])

    # Scatter: actual vs predicted
    ax.scatter(
        valid_df[TARGET_COL], valid_df["prediction"],
        alpha=0.4, s=8, color="#4472C4", label="Validation Points", zorder=2
    )

    # 45-degree reference line
    lim_max = max(valid_df[TARGET_COL].max(), valid_df["prediction"].max()) * 1.02
    ax.plot(
        [0, lim_max], [0, lim_max],
        color="red", linestyle="--", linewidth=1.5, label="45-Degree Reference", zorder=3
    )
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    # Top 3 largest-error annotations
    top3 = valid_df.nlargest(3, "abs_error").reset_index(drop=True)
    for i, row in top3.iterrows():
        date_str = pd.Timestamp(row["date"]).strftime("%m-%d")
        label = f"Top{i + 1}: {date_str} {row['store_id']}-{row['category']}"
        ax.annotate(
            label,
            xy=(row[TARGET_COL], row["prediction"]),
            xytext=(lim_max * 0.44, lim_max * 0.96 - i * lim_max * 0.07),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            ha="left",
            zorder=5,
        )

    # Residual inset (bottom-right)
    ax_res = ax.inset_axes([0.53, 0.03, 0.44, 0.34])
    ax_res.set_facecolor(plot_bg_color)
    residuals = valid_df["prediction"] - valid_df[TARGET_COL]
    ax_res.hist(residuals, bins=50, color="#70AD47", edgecolor="white", linewidth=0.3)
    ax_res.axvline(0, color="red", linestyle="--", linewidth=1)
    ax_res.set_title("Residual Distribution", fontsize=8)
    ax_res.set_xlabel("Prediction - Actual", fontsize=7)
    ax_res.tick_params(labelsize=7)
    for spine in ax_res.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.6)

    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Validation Diagnostics")
    ax.set_xlabel("Actual Orders")
    ax.set_ylabel("Predicted Orders")
    fig.tight_layout()
    # ------------------------------------------------------------------------
    return fig


############################################################################
############################### MAIN LOOP ##################################
############################################################################

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(args.figure_dir)

    raw = load_raw_data(data_dir)
    schema_summary = build_schema_summary(raw)
    save_csv(schema_summary, output_dir / "01_schema_summary.csv")

    train_master = merge_master(
        raw["train"], raw["stores"], raw["calendar"], raw["promo"], raw["external"]
    )
    train_master, cleaning_log = clean_master(train_master)

    save_csv(cleaning_log, output_dir / "02_cleaning_log.csv")
    save_csv(train_master.head(300), output_dir / "02_master_sample.csv")

    kpi_outputs = kpi_analysis(train_master)
    save_csv(kpi_outputs["city_kpi"], output_dir / "03_city_kpi.csv")
    save_csv(kpi_outputs["category_stats"], output_dir / "03_category_stats.csv")
    save_csv(kpi_outputs["promo_lift"], output_dir / "03_promo_lift.csv")
    save_csv(kpi_outputs["city_category_pivot"], output_dir / "03_city_category_pivot.csv")
    save_csv(kpi_outputs["store_alerts"], output_dir / "03_store_alerts.csv")

    feat_df = add_time_features(train_master)
    feat_df = add_lag_rolling_features(feat_df)

    feature_dict = build_feature_dictionary(feat_df)
    save_csv(feature_dict, output_dir / "04_feature_dictionary.csv")
    save_csv(feat_df.head(300), output_dir / "04_feature_preview.csv")

    train_df, valid_df = split_train_valid(feat_df)

    split_summary = pd.DataFrame([{
        "train_start": train_df["date"].min(),
        "train_end": train_df["date"].max(),
        "valid_start": valid_df["date"].min(),
        "valid_end": valid_df["date"].max(),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
    }])
    save_csv(split_summary, output_dir / "04_train_valid_split_summary.csv")

    baseline_metrics = run_baselines(valid_df)
    model, formal_metrics, pred_df = train_formal_model(train_df, valid_df)

    model_metrics = pd.concat([baseline_metrics, formal_metrics], ignore_index=True)
    save_csv(model_metrics, output_dir / "05_model_metrics.csv")

    ablation = pd.DataFrame([
        {"feature_set": "calendar_only", "todo": "students implement"},
        {"feature_set": "calendar_plus_ops", "todo": "students implement"},
        {"feature_set": "full_features", "todo": "students implement"},
    ])
    save_csv(ablation, output_dir / "05_ablation.csv")

    error_outputs = error_analysis(pred_df, valid_df)
    save_csv(error_outputs["error_by_city"], output_dir / "06_error_by_city.csv")
    save_csv(error_outputs["error_by_category"], output_dir / "06_error_by_category.csv")
    save_csv(error_outputs["error_by_condition"], output_dir / "06_error_by_condition.csv")
    save_csv(error_outputs["top10_errors"], output_dir / "06_top10_errors.csv")

    daily_df = (
        train_master.groupby("date", as_index=False)
        .agg(
            total_orders=(TARGET_COL, "sum"),
            promo_share=("promo_flag", "mean"),
        )
        .sort_values("date")
    )
    daily_df["rolling_mean_7"] = daily_df["total_orders"].rolling(7).mean()

    fig_a = plot_daily_orders_overview(daily_df)
    fig_a.savefig(
        figure_dir / "fig_a.png",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig_a.get_facecolor(),
    )

    fig_b = plot_validation_diagnostics(pred_df)
    fig_b.savefig(figure_dir / "fig_b.png", dpi=200, bbox_inches="tight")

    print("Project pipeline completed.")


if __name__ == "__main__":
    main()
