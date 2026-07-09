import math
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import lightgbm as lgbm


def fill_missing_values(df):
    df = df.copy()

    df["revenue"] = (
        df.groupby(["sku", "channel"])["revenue"]
          .transform(lambda x: x.ffill().bfill())
    )

    df["COGS"] = (
        df.groupby(["sku", "channel"])["COGS"]
          .transform(lambda x: x.ffill().bfill())
    )

    return df

def correct_outliers(df, factor=3):
    df = df.copy()

    z_scores = (
        (df["qty"] - df["qty"].mean())
        / df["qty"].std()
    )

    outlier_mask = np.abs(z_scores) > factor

    replacement = int(round(df["qty"].mean()))

    df.loc[outlier_mask, "qty"] = replacement

    return df

def add_template(data, min_date, max_date, skus):
    data = data.copy()

    data["shipped_date"] = pd.to_datetime(data["shipped_date"])

    # Create date template
    date_template = pd.DataFrame({
        "shipped_date": pd.date_range(start=min_date, end=max_date)
    })
    date_template["key"] = 1

    # Create SKU template
    sku_template = pd.DataFrame({
        "sku": skus
    })
    sku_template["key"] = 1

    # Cross join Date × SKU
    template = (
        pd.merge(date_template, sku_template, on="key")
          .drop("key", axis=1)
    )

    # Merge with original data
    data_merge = pd.merge(
        template,
        data,
        on=["shipped_date", "sku"],
        how="left"
    )

    # Fill missing sales
    data_merge["qty"] = data_merge["qty"].fillna(0)
    data_merge["revenue"] = data_merge["revenue"].fillna(0)
    data_merge["COGS"] = data_merge["COGS"].fillna(0)

    return data_merge

def engineer_temporal_features(data):
    """
    Create temporal features from shipped_date.
    """

    data = data.copy()

    data["shipped_date"] = pd.to_datetime(data["shipped_date"])

    data["month"] = data["shipped_date"].dt.month
    data["day"] = data["shipped_date"].dt.day
    data["dayofweek"] = data["shipped_date"].dt.dayofweek
    data["weekofyear"] = (
        data["shipped_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["quarter"] = data["shipped_date"].dt.quarter

    data["is_weekend"] = (
        data["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    data["is_even_day"] = (
        data["day"] % 2 == 0
    ).astype(int)

    return data

def engineer_sku_features(data):

    data = data.copy()

    data["shipped_date"] = pd.to_datetime(data["shipped_date"])

    data = data.sort_values(
        ["sku", "shipped_date"]
    )

    data["dow"] = data["shipped_date"].dt.weekday

    data["mean_qty_sku_dow"] = (
        data.groupby(["sku", "dow"])["qty"]
        .transform(
            lambda x:
            x.shift(1).expanding().mean()
        )
    )

    data["month"] = data["shipped_date"].dt.month

    data["mean_qty_sku_month"] = (
        data.groupby(["sku", "month"])["qty"]
        .transform(
            lambda x:
            x.shift(1).expanding().mean()
        )
    )

    sku_mean = (
        data.groupby("sku")["qty"]
        .transform(
            lambda x:
            x.shift(1).expanding().mean()
        )
    )

    data["mean_qty_sku_dow"] = (
        data["mean_qty_sku_dow"]
        .fillna(sku_mean)
    )

    data["mean_qty_sku_month"] = (
        data["mean_qty_sku_month"]
        .fillna(sku_mean)
    )

    data.fillna(0, inplace=True)

    return data

def cal_ma_lag_features(data):

    data = data.copy()

    data["lag1"] = (
        data.groupby("sku")["qty"]
        .shift(1)
    )

    data["lag7"] = (
        data.groupby("sku")["qty"]
        .shift(7)
    )

    data["lag30"] = (
        data.groupby("sku")["qty"]
        .shift(30)
    )

    data["rolling_mean_7"] = (
        data.groupby("sku")["qty"]
        .transform(
            lambda x:
            x.rolling(7, min_periods=1).mean()
        )
    )

    data["rolling_mean_30"] = (
        data.groupby("sku")["qty"]
        .transform(
            lambda x:
            x.rolling(30, min_periods=1).mean()
        )
    )

    data["rolling_std_7"] = (
        data.groupby("sku")["qty"]
        .transform(
            lambda x:
            x.rolling(7, min_periods=1).std()
        )
    )

    data["rolling_std_30"] = (
        data.groupby("sku")["qty"]
        .transform(
            lambda x:
            x.rolling(30, min_periods=1).std()
        )
    )

    return data

def weighted_absolute_percentage_error(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(y_true - y_pred))
        / denominator
    ) * 100

def load_model(file_path):
    """
    Load a trained model from disk.
    Supports both pickle models and LightGBM Booster.
    """

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            print(f"Model loaded from {file_path}")

    except (pickle.UnpicklingError, FileNotFoundError):

        model = lgbm.Booster(model_file=file_path)

        print(f"LightGBM Booster loaded from {file_path}")

    return model

def plot_sku_forecast(
    data,
    sku_list,
    date_col="shipped_date",
    actual_col="qty",
    pred_col="prediction",
    sku_col="sku",
    n_cols=3,
    figsize=(18, 10),
):
    """
    Plot Actual vs Forecast for selected SKUs.
    """

    plot_df = data[data[sku_col].isin(sku_list)].copy()

    plot_df[date_col] = pd.to_datetime(plot_df[date_col])

    n_skus = len(sku_list)
    n_rows = math.ceil(n_skus / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.flatten()

    for i, sku in enumerate(sku_list):

        sku_data = (
            plot_df[plot_df[sku_col] == sku]
            .sort_values(date_col)
        )

        ax = axes[i]

        ax.plot(
            sku_data[date_col],
            sku_data[actual_col],
            label="Actual",
            linewidth=2,
        )

        ax.plot(
            sku_data[date_col],
            sku_data[pred_col],
            linestyle="--",
            marker="o",
            markersize=3,
            label="Forecast",
        )

        ax.set_title(f"SKU: {sku}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Quantity")
        ax.legend()
        ax.grid(True)

        ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()