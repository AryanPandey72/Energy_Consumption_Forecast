# Energy Consumption Forecasting using LSTM 

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import layers, models

DATA_TXT_PATH = "/content/household_power_consumption.txt"  # set to your uploaded file path

# Columns and target
DATE_COL = "Date"
TIME_COL_ONLY = "Time"
TIME_COL = "timestamp"               # new combined datetime column
TARGET_COL = "Global_active_power"   # target variable in kW

# Frequency (lowercase to avoid pandas deprecation)
RESAMPLE_FREQ = "h"                  # hourly

# Windowing
INPUT_WINDOW = 48
FORECAST_HORIZON = 24
SHIFT = 1

# Training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
LSTM_UNITS = 64
LSTM_LAYERS = 1
DROPOUT = 0.2
PATIENCE = 5
SCALER_TYPE = "Standard"  # "Standard" or "MinMax"
SEED = 42

# Splits by time
VAL_SIZE_RATIO = 0.1
TEST_SIZE_RATIO = 0.1

# Output
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports", exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

def save_and_show(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.show()
    plt.close()

def make_scalers():
    if SCALER_TYPE.lower() == "minmax":
        return MinMaxScaler(), MinMaxScaler()
    return StandardScaler(), StandardScaler()

def train_val_test_split_by_time(X, y, val_ratio, test_ratio):
    n = len(X)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_end = n - test_n - val_n
    val_end = n - test_n
    return (X.iloc[:train_end], y.iloc[:train_end],
            X.iloc[train_end:val_end], y.iloc[train_end:val_end],
            X.iloc[val_end:], y.iloc[val_end:])

def make_windows(X_arr, y_arr, input_window, horizon, shift):
    X_list, y_list = [], []
    for i in range(len(X_arr) - input_window - shift - horizon + 1):
        X_list.append(X_arr[i: i + input_window])
        start = i + input_window + shift - 1
        y_list.append(y_arr[start + 1: start + 1 + horizon].ravel())
    return np.array(X_list), np.array(y_list)

def coerce_numeric(df, exclude_cols):
    for c in df.columns:
        if c in exclude_cols:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Data loading and EDA
def load_and_prepare_household_power(txt_path):
    print(f"Reading dataset from: {txt_path}")
    df = pd.read_csv(
        txt_path,
        sep=";",
        na_values=["?"],
        dtype=str,         # read as strings first to avoid mixed-type warnings
        low_memory=False
    )

    if DATE_COL not in df.columns or TIME_COL_ONLY not in df.columns:
        raise ValueError(f"Expected '{DATE_COL}' and '{TIME_COL_ONLY}' columns, found: {df.columns.tolist()}")
    dt_str = df[DATE_COL].str.strip() + " " + df[TIME_COL_ONLY].str.strip()
    df[TIME_COL] = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S", errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL]).drop(columns=[DATE_COL, TIME_COL_ONLY])

    df = coerce_numeric(df, exclude_cols=[TIME_COL])
    df = df.sort_values(TIME_COL).set_index(TIME_COL)

    df = df.resample(RESAMPLE_FREQ).mean()
    df = df.interpolate(method="time").ffill().bfill()
    df = df.dropna(how="all")

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not found; available: {df.columns.tolist()}")
    if df[TARGET_COL].isna().all():
        raise ValueError("After cleaning, target column is all NaN. Check parsing or column selection.")

    print("Data shape after load:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def add_time_features(df):
    df = df.copy()
    idx = df.index
    hour = idx.hour.values
    dow = idx.dayofweek.values
    df["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    df["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    df["dow_cos"] = np.cos(2*np.pi*dow/7.0)
    df["target_rm_24"] = df[TARGET_COL].rolling(24, min_periods=1).mean()
    df["target_rs_24"] = df[TARGET_COL].rolling(24, min_periods=1).std().fillna(0.0)
    return df

def plot_eda(df):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df[TARGET_COL], lw=1)
    plt.title("Global_active_power time series")
    plt.xlabel("Time")
    plt.ylabel(TARGET_COL)
    save_and_show("reports/figures/eda_target.png")

    tmp = df.copy()
    tmp["hour"] = tmp.index.hour
    hourly = tmp.groupby("hour")[TARGET_COL].mean()
    plt.figure(figsize=(8,4))
    hourly.plot()
    plt.title("Average by hour of day")
    plt.xlabel("Hour")
    plt.ylabel(TARGET_COL)
    save_and_show("reports/figures/eda_hourly_mean.png")

    tmp = df.copy()
    tmp["dow"] = tmp.index.dayofweek
    weekly = tmp.groupby("dow")[TARGET_COL].mean()
    plt.figure(figsize=(8,4))
    weekly.plot()
    plt.title("Average by day of week")
    plt.xlabel("DayOfWeek (0=Mon)")
    plt.ylabel(TARGET_COL)
    save_and_show("reports/figures/eda_weekly_mean.png")

# =========================
# Model
# =========================
def build_lstm_model(n_features):
    inputs = layers.Input(shape=(None, n_features))
    x = inputs
    for _ in range(max(0, LSTM_LAYERS - 1)):
        x = layers.LSTM(LSTM_UNITS, return_sequences=True)(x)
        x = layers.Dropout(DROPOUT)(x)
    x = layers.LSTM(LSTM_UNITS, return_sequences=False)(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(FORECAST_HORIZON)(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

# Training, evaluation, plots

def safe_array(a):
    a = np.array(a, dtype=np.float32)
    a[~np.isfinite(a)] = np.nan
    if a.ndim == 2:
        col_means = np.nanmean(a, axis=0)
        inds = np.where(np.isnan(a))
        a[inds] = np.take(col_means, inds[1])
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a

def train_and_evaluate(df):
    df_feat = add_time_features(df)
    feature_cols = [c for c in df_feat.columns if c != TARGET_COL]
    X = df_feat[feature_cols]
    y = df_feat[TARGET_COL]

    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.dropna()
    X = X.loc[y.index]
    X = X.fillna(X.median())

    X_tr, y_tr, X_va, y_va, X_te, y_te = train_val_test_split_by_time(X, y, VAL_SIZE_RATIO, TEST_SIZE_RATIO)

    X_scaler, y_scaler = make_scalers()
    X_tr_s = X_scaler.fit_transform(X_tr)
    X_va_s = X_scaler.transform(X_va)
    X_te_s = X_scaler.transform(X_te)
    y_tr_s = y_scaler.fit_transform(y_tr.values.reshape(-1,1))
    y_va_s = y_scaler.transform(y_va.values.reshape(-1,1))
    y_te_s = y_scaler.transform(y_te.values.reshape(-1,1))

    X_tr_s = safe_array(X_tr_s)
    X_va_s = safe_array(X_va_s)
    X_te_s = safe_array(X_te_s)
    y_tr_s = safe_array(y_tr_s)
    y_va_s = safe_array(y_va_s)
    y_te_s = safe_array(y_te_s)

    Xtr3, ytr2 = make_windows(X_tr_s, y_tr_s, INPUT_WINDOW, FORECAST_HORIZON, SHIFT)
    Xva3, yva2 = make_windows(X_va_s, y_va_s, INPUT_WINDOW, FORECAST_HORIZON, SHIFT)
    Xte3, yte2 = make_windows(X_te_s, y_te_s, INPUT_WINDOW, FORECAST_HORIZON, SHIFT)

    print("Train windows:", Xtr3.shape, ytr2.shape)
    print("Val windows:", Xva3.shape, yva2.shape)
    print("Test windows:", Xte3.shape, yte2.shape)

    n_features = Xtr3.shape[-1]
    model = build_lstm_model(n_features)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    ]

    history = model.fit(
        Xtr3, ytr2,
        validation_data=(Xva3, yva2),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Learning curves
    hist = history.history
    plt.figure(figsize=(10,4))
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.title("Learning curves (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_and_show("reports/figures/learning_curves.png")

    # Predict and inverse-scale
    y_pred_s = model.predict(Xte3)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = y_scaler.inverse_transform(yte2)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred)**2)))
    r2 = float(r2_score(y_true.ravel(), y_pred.ravel()))
    print(f"Test MAE: {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R2: {r2:.3f}")

    # Horizon-wise errors
    mae_per_h = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_per_h = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))

    plt.figure(figsize=(10,4))
    plt.plot(mae_per_h, label="MAE per step")
    plt.plot(rmse_per_h, label="RMSE per step")
    plt.title("Horizon-wise errors")
    plt.xlabel("Forecast step (1..H)")
    plt.ylabel("Error")
    plt.legend()
    save_and_show("reports/figures/horizon_errors.png")

    # Residuals histogram (t+1)
    residuals_h1 = y_true[:,0] - y_pred[:,0]
    plt.figure(figsize=(10,4))
    sns.histplot(residuals_h1, bins=40, kde=True)
    plt.title("Residuals distribution (t+1)")
    plt.xlabel("Residual")
    save_and_show("reports/figures/residuals_hist.png")

    # Sample forecast plot (last test window)
    idx = -1
    plt.figure(figsize=(10,4))
    plt.plot(y_true[idx], label="True")
    plt.plot(y_pred[idx], label="Pred")
    plt.title("Sample next-H forecast (last test sample)")
    plt.xlabel("Step")
    plt.ylabel(TARGET_COL)
    plt.legend()
    save_and_show("reports/figures/sample_forecast.png")

    # Overlay history and next-H forecast
    last_time = y_te.index[-1]
    freq = pd.tseries.frequencies.to_offset(RESAMPLE_FREQ)
    future_index = pd.date_range(last_time + freq, periods=FORECAST_HORIZON, freq=freq)
    forecast_series = pd.Series(y_pred[idx], index=future_index)

    hist_points = min(len(df), INPUT_WINDOW*10)
    df_tail = df.iloc[-hist_points:][TARGET_COL]
    plt.figure(figsize=(12,4))
    plt.plot(df_tail.index, df_tail.values, label="History", lw=1)
    plt.plot(forecast_series.index, forecast_series.values, label="Forecast", lw=2)
    plt.title("History and next-H forecast")
    plt.xlabel("Time")
    plt.ylabel(TARGET_COL)
    plt.legend()
    save_and_show("reports/figures/history_forecast_overlay.png")

    # Save artifacts
    model.save("reports/lstm_energy_forecast.h5")
    np.save("reports/y_true_test.npy", y_true)
    np.save("reports/y_pred_test.npy", y_pred)
    with open("reports/metrics.json", "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2}, f, indent=2)

    # Latest forecast from end of full data
    X_full = df_feat[feature_cols]
    y_full = df_feat[TARGET_COL]
    X_scaler_f, y_scaler_f = make_scalers()
    Xs = X_scaler_f.fit_transform(X_full)
    ys = y_scaler_f.fit_transform(y_full.values.reshape(-1,1))
    X_last = Xs[-INPUT_WINDOW:]
    X_last = np.expand_dims(X_last, axis=0)
    y_last_pred_s = model.predict(X_last)
    y_last_pred = y_scaler_f.inverse_transform(y_last_pred_s)[0]
    last_time_full = df_feat.index[-1]
    future_index_full = pd.date_range(last_time_full + freq, periods=FORECAST_HORIZON, freq=freq)
    forecast_df = pd.DataFrame({TARGET_COL: y_last_pred}, index=future_index_full)
    forecast_df.to_csv("reports/forecast_latest.csv")

    print("Artifacts saved in 'reports/'.")
    return {"metrics": {"mae": mae, "rmse": rmse, "r2": r2}, "forecast_df": forecast_df}

def main():
    df = load_and_prepare_household_power(DATA_TXT_PATH)
    plot_eda(df)
    _ = train_and_evaluate(df)
    print("Done.")

if __name__ == "__main__":
    main()
