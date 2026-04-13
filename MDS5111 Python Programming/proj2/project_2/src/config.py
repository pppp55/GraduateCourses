# src/config.py
# Constants of the project.
# YOU SHOULD NOT MODIFY THIS FILE.

TARGET_COL = "orders"

ID_COLS = ["date", "store_id", "category"]

TRAIN_FILE = "orders_train.csv"
TEST_FILE = "orders_test.csv"
STORE_FILE = "store_info.csv"
CALENDAR_FILE = "calendar.csv"
PROMO_FILE = "promotion.csv"
EXTERNAL_FILE = "city_external.csv"

VALID_DAYS = 28

FEATURE_COLS_BASE = [
    "promo_flag",
    "promo_depth",
    "coupon_flag",
    "discount_rate",
    "stockout_minutes",
    "avg_delivery_time",
    "refund_rate",
    "rain_level",
    "traffic_index",
    "is_weekend",
    "is_holiday",
    "is_payday_window",
    "month",
    "day_of_week",
]

FEATURE_COLS_TS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "promo_last_7d_count",
    "stockout_last_7d_mean",
    "days_since_last_promo",
]

CATEGORICAL_COLS = [
    "city",
    "city_tier",
    "business_district_type",
    "store_size",
    "category",
    "promo_type",
    "temperature_bin",
]