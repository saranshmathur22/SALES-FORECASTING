"""
Sales Forecasting: ARIMA/SARIMA vs Prophet vs LSTM
====================================================
Dataset: Rossmann Store Sales (Kaggle) — daily sales for a single
European drugstore (Store #2), Jan 2013 - Jul 2015. This gives a
realistic single-series retail signal with:
  - long-term trend
  - weekly seasonality (store closed every Sunday)
  - yearly seasonality
  - promotional effects (Promo flag)
  - holiday effects (state + school holidays, which also close the store)

Outputs:
  - metrics table (RMSE / MAE / MAPE) for each model on a held-out test window
  - forecast-vs-actual chart per model
  - combined comparison chart
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STORE_ID = 2

# ---------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------
df = pd.read_csv("train.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
store_df = df[df["Store"] == STORE_ID].sort_values("Date").reset_index(drop=True)

daily = store_df[["Date", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].copy()
daily["IsStateHoliday"] = (daily["StateHoliday"] != "0").astype(int)
daily = daily.set_index("Date").asfreq("D")
# fill the asfreq-induced gaps (shouldn't be many) — treat as closed/no-promo
daily["Sales"] = daily["Sales"].fillna(0)
daily["Open"] = daily["Open"].fillna(0)
daily["Promo"] = daily["Promo"].fillna(0)
daily["IsStateHoliday"] = daily["IsStateHoliday"].fillna(0)
daily["SchoolHoliday"] = daily["SchoolHoliday"].fillna(0)

print(f"Store {STORE_ID} series length: {len(daily)} days, {daily.index.min().date()} to {daily.index.max().date()}")
print(f"Closed days: {(daily['Sales']==0).sum()}")

# ---------------------------------------------------------------
# 2. Train / test split — hold out the last 6 weeks (42 days)
# ---------------------------------------------------------------
TEST_DAYS = 42
train = daily.iloc[:-TEST_DAYS].copy()
test = daily.iloc[-TEST_DAYS:].copy()
print(f"Train: {len(train)} days | Test: {len(test)} days ({test.index.min().date()} -> {test.index.max().date()})")


def metrics(actual, pred, name):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))
    # MAPE only over days the store was actually open (avoid div-by-zero on closed days)
    open_mask = actual > 0
    mape = np.mean(np.abs((actual[open_mask] - pred[open_mask]) / actual[open_mask])) * 100
    print(f"{name:10s} | RMSE: {rmse:,.0f}  MAE: {mae:,.0f}  MAPE (open days): {mape:.2f}%")
    return {"model": name, "rmse": rmse, "mae": mae, "mape": mape}


results = []
forecasts = {}

# ---------------------------------------------------------------
# 3. SARIMA (statsmodels) — weekly seasonal ARIMA with exogenous promo
# ---------------------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX

exog_train = train[["Promo", "IsStateHoliday"]]
exog_test = test[["Promo", "IsStateHoliday"]]

sarima = SARIMAX(
    train["Sales"],
    exog=exog_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_fit = sarima.fit(disp=False)
sarima_fc = sarima_fit.get_forecast(steps=TEST_DAYS, exog=exog_test).predicted_mean
sarima_fc.index = test.index
forecasts["SARIMA"] = sarima_fc
results.append(metrics(test["Sales"], sarima_fc, "SARIMA"))

# ---------------------------------------------------------------
# 4. Prophet — trend + weekly/yearly seasonality + promo regressor + holidays
# ---------------------------------------------------------------
from prophet import Prophet

prophet_train = train.reset_index()[["Date", "Sales", "Promo"]].rename(
    columns={"Date": "ds", "Sales": "y"}
)

holidays_df = daily.reset_index()
holidays_df = holidays_df[holidays_df["IsStateHoliday"] == 1][["Date"]].rename(columns={"Date": "ds"})
holidays_df["holiday"] = "state_holiday"

m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays_df,
    changepoint_prior_scale=0.1,
)
m.add_regressor("Promo")
m.fit(prophet_train)

future = test.reset_index()[["Date", "Promo"]].rename(columns={"Date": "ds"})
prophet_fc = m.predict(future)
prophet_pred = pd.Series(prophet_fc["yhat"].values, index=test.index)
forecasts["Prophet"] = prophet_pred
results.append(metrics(test["Sales"], prophet_pred, "Prophet"))

# ---------------------------------------------------------------
# 5. LSTM (PyTorch) — sequence model on lagged sales + promo feature
# ---------------------------------------------------------------
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 14
features = daily[["Sales", "Promo"]].copy()

scaler_sales = MinMaxScaler()
scaler_promo = MinMaxScaler()
scaled = features.copy()

# fit scalers on train only, transform full series
scaler_sales.fit(train[["Sales"]])
scaler_promo.fit(train[["Promo"]])
scaled["Sales"] = scaler_sales.transform(features[["Sales"]])[:, 0]
scaled["Promo"] = scaler_promo.transform(features[["Promo"]])[:, 0]

def make_sequences(arr, lookback):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback : i])
        y.append(arr[i, 0])  # predict Sales
    return np.array(X), np.array(y)

full_arr = scaled[["Sales", "Promo"]].values
train_len = len(train)

X_all, y_all = make_sequences(full_arr, LOOKBACK)
# index i in X_all/y_all corresponds to original row (LOOKBACK + i)
split_idx = train_len - LOOKBACK
X_train, y_train = X_all[:split_idx], y_all[:split_idx]
X_test, y_test = X_all[split_idx:], y_all[split_idx:]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

class LSTMForecaster(nn.Module):
    def __init__(self, n_features=2, hidden=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

torch.manual_seed(42)
model = LSTMForecaster()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

EPOCHS = 150
for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    opt.step()
    if (epoch + 1) % 30 == 0:
        print(f"LSTM epoch {epoch+1}/{EPOCHS} - loss: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    lstm_pred_scaled = model(X_test_t).numpy().flatten()
lstm_pred = scaler_sales.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
lstm_pred = pd.Series(lstm_pred, index=test.index)
forecasts["LSTM"] = lstm_pred
results.append(metrics(test["Sales"], lstm_pred, "LSTM"))

# ---------------------------------------------------------------
# 6. Results table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("rmse")
results_df.to_csv("model_metrics.csv", index=False)
print("\n=== Model comparison (lower is better) ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------
# 7. Charts
# ---------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
colors = {"SARIMA": "#e07a5f", "Prophet": "#3d5a80", "LSTM": "#81b29a"}

# 7a. Full history with trend context
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(daily.index, daily["Sales"], color="#333333", linewidth=0.8, label=f"Actual daily sales — Store {STORE_ID}")
ax.axvspan(test.index.min(), test.index.max(), color="gray", alpha=0.12, label="Test window (held out)")
ax.set_title(f"Rossmann Store {STORE_ID} Daily Sales — Jan 2013 to Jul 2015", fontsize=13, fontweight="bold")
ax.set_ylabel("Sales (€)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("chart_full_history.png", dpi=140)
plt.close(fig)

# 7b. Per-model forecast vs actual (zoomed to test window)
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, (name, fc) in zip(axes, forecasts.items()):
    ax.plot(test.index, test["Sales"], color="#333333", linewidth=1.8, label="Actual")
    ax.plot(test.index, fc, color=colors[name], linewidth=1.8, linestyle="--", label=f"{name} forecast")
    ax.fill_between(test.index, test["Sales"], fc, color=colors[name], alpha=0.08)
    row = results_df[results_df["model"] == name].iloc[0]
    ax.set_title(f"{name}  —  RMSE {row['rmse']:,.0f}   MAE {row['mae']:,.0f}   MAPE {row['mape']:.2f}%", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.suptitle("Forecast vs Actual — Held-out 6-Week Test Window", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("chart_per_model.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# 7c. Combined overlay
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(test.index, test["Sales"], color="black", linewidth=2.2, label="Actual", zorder=5)
for name, fc in forecasts.items():
    ax.plot(test.index, fc, linewidth=1.6, linestyle="--", color=colors[name], label=name)
ax.set_title(f"All Models — Forecast vs Actual (Store {STORE_ID}, Test Window)", fontsize=13, fontweight="bold")
ax.set_ylabel("Sales (€)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("chart_combined.png", dpi=140)
plt.close(fig)

# 7d. Metrics bar chart
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(results_df))
ax.bar(x, results_df["mape"], color=[colors[m] for m in results_df["model"]])
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"])
ax.set_ylabel("MAPE (%)")
ax.set_title("Forecast Error Comparison (lower = better)", fontsize=12, fontweight="bold")
for i, v in enumerate(results_df["mape"]):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=10)
fig.tight_layout()
fig.savefig("chart_error_comparison.png", dpi=140)
plt.close(fig)

print("\nSaved charts: chart_full_history.png, chart_per_model.png, chart_combined.png, chart_error_comparison.png")
print("Saved metrics: model_metrics.csv")
