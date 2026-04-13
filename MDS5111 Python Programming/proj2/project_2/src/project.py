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
    # Read the remaining five input tables from data_dir and store them in the
    # variables below.
    #
    # Requirements:
    # - Use the exact filenames defined in config.py.
    # - Parse the "date" column for any table that contains a date column.
    # - Keep other columns as read by default.
    # - Return a dict with the exact keys used below: train, test, stores,
    #   calendar, promo, external.
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 1 in load_raw_data().")

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
        # Append one summary dict per table to rows.
        #
        # Required fields in each dict:
        # - table_name: table key name
        # - n_rows: number of rows
        # - n_cols: number of columns
        # - n_missing_total: total number of missing cells in the whole table
        # - date_min: minimum date if a "date" column exists, otherwise pd.NaT
        # - date_max: maximum date if a "date" column exists, otherwise pd.NaT
        #
        # Keep the row order consistent with raw.items().
        # ------------------------------------------------------------------------
        raise NotImplementedError("TODO: complete Task 2 in build_schema_summary().")
    return pd.DataFrame(rows)


############################################################################
######################### SECTION 2 MERGE & CLEAN ##########################
############################################################################

def merge_master(order_df, stores, calendar, promo, external):
    # TASK 3 -----------------------------------------------------------------
    # Merge the input tables into one master table.
    #
    # Requirements:
    # - Use left joins throughout.
    # - Merge order_df with stores on "store_id".
    # - Then merge with calendar on "date".
    # - Then merge with promo on ["store_id", "date", "category"].
    # - Then merge with external on ["city", "date"].
    # - Do not drop rows, deduplicate, or sort in this function.
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 3 in merge_master().")
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
            # For each column in fill_zero_cols that exists in df:
            # - count how many missing values it has BEFORE filling
            # - fill missing values with 0
            # - append one log record with:
            #     step = f"fillna_{col}"
            #     n_filled = <number of values filled>
            #
            # Keep the existing loop and log format unchanged.
            # ---------------------------------------------------------------
            raise NotImplementedError("TODO: complete Task 4 in clean_master().")

    if "discount_rate" in df.columns:
        # TASK 5 -----------------------------------------------------------------
        # Record and repair invalid discount values.
        #
        # Requirements:
        # - Count values outside [0, 1] before fixing.
        # - Clip the full column into [0, 1].
        # - Append a log record with:
        #     step = "clip_discount_rate"
        #     n_affected = <count of invalid rows>
        # ------------------------------------------------------------------------
        raise NotImplementedError("TODO: complete Task 5 in clean_master().")

    if "stockout_minutes" in df.columns:
        # TASK 6 -----------------------------------------------------------------
        # Record and repair invalid stockout values.
        #
        # Requirements:
        # - Count rows where stockout_minutes < 0.
        # - Clip the column from below at 0.
        # - Append a log record with:
        #     step = "clip_stockout_minutes"
        #     n_affected = <count of invalid rows>
        # ------------------------------------------------------------------------
        raise NotImplementedError("TODO: complete Task 6 in clean_master().")

    if "avg_delivery_time" in df.columns:
        # TASK 7 -----------------------------------------------------------------
        # Record and repair invalid delivery-time values.
        #
        # Requirements:
        # - Treat values <= 0 as invalid.
        # - Count invalid rows before fixing.
        # - Replace invalid values with np.nan.
        # - Append a log record with:
        #     step = "set_invalid_delivery_time_nan"
        #     n_affected = <count of invalid rows>
        # ------------------------------------------------------------------------
        raise NotImplementedError("TODO: complete Task 7 in clean_master().")

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
    # Build category-level volatility summary.
    #
    # Requirements:
    # - Group by "category".
    # - Aggregate mean and std of TARGET_COL only.
    # - Rename the aggregated columns to "orders_mean" and "orders_std".
    # - Add a new column "orders_cv" = orders_std / orders_mean.
    # - Keep pandas default std behavior.
    # - Output columns should be:
    #     ["category", "orders_mean", "orders_std", "orders_cv"]
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 8 in kpi_analysis().")

    # TASK 9 -----------------------------------------------------------------
    # Compute promotion lift for each (store_id, category) pair.
    #
    # Requirements:
    # - Group by ["store_id", "category"].
    # - Use promo_lift_func to compute one value per group.
    # - Keep rows even if promo_lift is NaN.
    # - Output columns should be:
    #     ["store_id", "category", "promo_lift"]
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 9 in kpi_analysis().")

    # TASK 10 ----------------------------------------------------------------
    # Create a city-by-category summary table of average orders.
    #
    # Requirements:
    # - Use a pivot table.
    # - index = "city"
    # - columns = "category"
    # - values = TARGET_COL
    # - aggfunc = "mean"
    # - reset the index at the end
    # - Do not manually fill missing values.
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 10 in kpi_analysis().")

    # TASK 11 ----------------------------------------------------------------
    # Build a store-level alert table.
    #
    # Requirements:
    # - Group by "store_id" and compute the mean of:
    #     TARGET_COL -> avg_orders
    #     stockout_minutes -> avg_stockout_minutes
    #     refund_rate -> avg_refund_rate
    #     gmv -> avg_gmv
    # - On the aggregated table, compute the 75th percentile of avg_orders and
    #   avg_stockout_minutes.
    # - Add column "high_orders_high_stockout_flag" equal to 1 only when both
    #   metrics are strictly above their respective 75th percentiles; otherwise 0.
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 11 in kpi_analysis().")

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
    # Add grouped rolling-window features.
    #
    # Requirements:
    # - Work within each (store_id, category) group.
    # - Apply shift(1) BEFORE every rolling calculation to avoid leakage.
    # - Create the following columns:
    #     rolling_mean_7
    #     rolling_std_7
    #     rolling_mean_28
    #     promo_last_7d_count
    #     stockout_last_7d_mean
    # - Keep the natural NaN values for the cold-start rows.
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 12 in add_lag_rolling_features().")

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
    # Prepare the model matrices for training and validation.
    #
    # Requirements:
    # - Select feature_cols + CATEGORICAL_COLS from both train_df and valid_df.
    # - One-hot encode only CATEGORICAL_COLS using pd.get_dummies(..., drop_first=True).
    # - Align validation columns to training columns using left alignment on axis=1
    #   and fill missing columns with 0.
    # - Return X_train, X_valid, y_train, y_valid.
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 13 in prepare_model_matrix().")

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
    # Train and evaluate one formal model.
    #
    # Requirements:
    # - Use feature_cols = FEATURE_COLS_BASE + FEATURE_COLS_TS.
    # - Drop rows with missing values in feature_cols or TARGET_COL separately
    #   for the training and validation subsets.
    # - Call prepare_model_matrix(...) to build X/y matrices.
    # - Fit LinearRegression() with default parameters.
    # - Predict on the validation set.
    # - Clip predictions at 0 from below.
    # - Build a one-row metrics DataFrame with columns:
    #     model_name, mae, rmse, mape, n_valid_rows
    # - Do not rename the model; use "linear_regression".
    # ------------------------------------------------------------------------
    raise NotImplementedError("TODO: complete Task 14 in train_formal_model().")

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
    # Create an executive time-series overview figure.
    #
    # Required elements:
    # - one line for total daily orders
    # - one line for a 7-day rolling mean
    # - highlighted high-promotion periods using promo_share
    # - one inset showing the final 28 days
    # - at least two annotations for notable points
    # - one legend
    #
    # Presentation requirements:
    # - Sort by date before plotting.
    # - Keep the figure readable: legend, inset, and annotation text should not
    #   cover the main patterns or overlap badly with each other.
    # - Use a reproducible rule for the high-promotion periods.
    # - Return the matplotlib Figure object.
    # - Your final result should be similar to `./figures/references/fig_a.png`.
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))
    fig_bg_color = "#ffffff"
    plot_bg_color = "#f3f3f3"
    fig.patch.set_facecolor(fig_bg_color)
    ax.set_facecolor(plot_bg_color)

    raise NotImplementedError("TODO: complete Task 15 in plot_daily_orders_overview().")

    ax.set_title("Executive Overview of Daily Orders")
    ax.set_xlabel("Date")
    ax.set_ylabel("Orders")
    fig.tight_layout()
    return fig



def plot_validation_diagnostics(valid_df: pd.DataFrame):
    # TASK 16 ----------------------------------------------------------------
    # Create a validation diagnostics figure.
    #
    # Required elements:
    # - a scatter plot of actual vs predicted values
    # - a 45-degree reference line
    # - a residual inset (or another small axes inside the same figure)
    # - annotations for the top 3 largest errors
    #
    # Presentation requirements:
    # - If abs_error is missing, create it first.
    # - Keep labels and annotations readable; avoid obvious overlap with points,
    #   the inset, and the legend.
    # - Return the matplotlib Figure object.
    # - Your final result should be similar to `./figures/references/fig_b.png`.
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_bg_color = "#f3f3f3"
    ax.set_facecolor(plot_bg_color)

    raise NotImplementedError("TODO: complete Task 16 in plot_validation_diagnostics().")

    ax.set_title("Validation Diagnostics")
    ax.set_xlabel("Actual Orders")
    ax.set_ylabel("Predicted Orders")
    fig.tight_layout()
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
