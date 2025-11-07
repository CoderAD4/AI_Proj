# app.py
"""
AI Stock Trend Predictor (Final Robust Version)

Features:
- LSTM regression (predicts next-day CLOSE PERCENT CHANGE for stationarity).
- SVM classification (predicts UP/DOWN: next Close > next Open).
- SHAP/LIME explainability with FINAL stability fixes.
- Recursive forecasting implemented.

Usage:
    streamlit run app.py

Requirements:
pip install streamlit yfinance pandas numpy scikit-learn matplotlib tensorflow shap lime tqdm joblib
"""

import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Tuple, Dict, Any
import tempfile 

# ML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Explainability
from lime import lime_tabular # SHAP is imported locally later

st.set_page_config(layout="wide", page_title="AI Stock Trend Predictor (Final)")

# Ensure reproducibility and silence warnings
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------
# Helper functions & features
# ---------------------------

@st.cache_data(show_spinner=False)
def download_data(ticker: str, period: str = "5y", interval: str = "1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required_cols = ["Close", "Open", "High", "Low", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in downloaded data: {col}")

    df = df.dropna().copy()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and the target feature (Close_Change) to DataFrame."""
    df = df.copy()
    df['Close_Change'] = df['Close'].pct_change() * 100 # Target feature: % change (return)
    
    df['MA7'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['MA21'] = df['Close'].rolling(window=21, min_periods=1).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14, min_periods=1).mean()
    ma_down = down.rolling(14, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility'] = df['Close'].pct_change().rolling(window=21, min_periods=1).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(7)
    df = df.fillna(method='bfill').fillna(0)
    return df

def create_windows(df: pd.DataFrame, feature_cols: List[str], target_col: str, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates windows for LSTM.
    Returns X, y (Close_Change), and Close prices needed to reverse the prediction.
    """
    values = df[feature_cols].values
    targets = df[target_col].values
    closes = df['Close'].values 
    
    X, y, previous_closes = [], [], []
    for i in range(window_size, len(df)):
        X.append(values[i - window_size:i])
        y.append(targets[i])
        previous_closes.append(closes[i - 1]) 
        
    X = np.array(X)
    y = np.array(y)
    previous_closes = np.array(previous_closes)
    
    min_len = min(len(X), len(y), len(previous_closes))
    return X[:min_len], y[:min_len], previous_closes[:min_len]

# ---------------------------
# Scaling utilities
# ---------------------------

class Scalers:
    def __init__(self):
        self.feature_scaler = None  # MinMaxScaler for features (LSTM)
        self.target_scaler = None    # MinMaxScaler for target (LSTM)

    def fit(self, X_windows: np.ndarray, y: np.ndarray):
        ns, w, f = X_windows.shape
        flat = X_windows.reshape(ns * w, f)
        
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler.fit(flat)
        
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler.fit(y.reshape(-1,1))

    def transform_X(self, X_windows: np.ndarray) -> np.ndarray:
        ns, w, f = X_windows.shape
        flat = X_windows.reshape(ns * w, f)
        flat_s = self.feature_scaler.transform(flat)
        return flat_s.reshape(ns, w, f)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1,1)).flatten()

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(y.reshape(-1,1)).flatten()

# ---------------------------
# LSTM model utilities
# ---------------------------

def build_lstm_model(input_shape: Tuple[int,int], lr: float = 1e-3) -> tf.keras.models.Model:
    """Build a robust LSTM model for predicting percentage change."""
    inp = Input(shape=input_shape)
    x = LSTM(96, return_sequences=True, activation='tanh')(inp)
    x = Dropout(0.3)(x)
    x = LSTM(48, activation='tanh')(x)
    x = Dense(24, activation='relu')(x)
    out = Dense(1, name='out')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

@st.cache_resource(show_spinner=False)
def load_lstm_model(input_shape, model_path):
    """Loads LSTM model weights from the saved path (required for SHAP/TF fix)."""
    model = build_lstm_model(input_shape)
    model.load_weights(model_path)
    return model


# ---------------------------
# Recursive forecasting 
# ---------------------------

def recursive_forecast_with_recalc(model: tf.keras.models.Model,
                                   scalers: Scalers,
                                   df_history: pd.DataFrame,
                                   feature_cols: List[str],
                                   window_size: int,
                                   days: int) -> List[float]:
    """Generates multi-day forecasts by predicting % change and recalculating indicators."""
    df = df_history.copy().reset_index(drop=True)
    preds = []
    
    history_close_series = df['Close']
    history_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0.0

    for _ in range(days):
        window_raw = df[feature_cols].iloc[-window_size:].values
        
        window_scaled = scalers.feature_scaler.transform(window_raw.reshape(window_size, -1)).reshape(1, window_size, len(feature_cols))
        pred_change_scaled = model.predict(window_scaled, verbose=0).flatten()[0]
        pred_change_unscaled = scalers.inverse_transform_y(np.array([pred_change_scaled]))[0]

        last_close = df['Close'].iloc[-1]
        new_close = last_close * (1 + (pred_change_unscaled / 100))
        preds.append(float(new_close))

        temp_close_series = pd.concat([history_close_series, pd.Series(preds)], ignore_index=True)
        
        # Compute new features
        ma7 = temp_close_series.rolling(window=7, min_periods=1).mean().iloc[-1]
        ma21 = temp_close_series.rolling(window=21, min_periods=1).mean().iloc[-1]
        ema12 = temp_close_series.ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = temp_close_series.ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema12 - ema26
        delta = temp_close_series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(14, min_periods=1).mean().iloc[-1]
        ma_down = down.rolling(14, min_periods=1).mean().iloc[-1]
        rs = ma_up / (ma_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        volatility = temp_close_series.pct_change().rolling(window=21, min_periods=1).std().iloc[-1]
        momentum = temp_close_series.iloc[-1] - temp_close_series.shift(7).iloc[-1] if len(temp_close_series) >= 8 else 0.0
        new_close_change = pred_change_unscaled 

        new_row = {}
        for c in feature_cols:
            if c == 'Close': new_row['Close'] = new_close
            elif c == 'Close_Change': new_row['Close_Change'] = new_close_change 
            elif c == 'MA7': new_row['MA7'] = ma7
            elif c == 'MA21': new_row['MA21'] = ma21
            elif c == 'EMA12': new_row['EMA12'] = ema12
            elif c == 'EMA26': new_row['EMA26'] = ema26
            elif c == 'MACD': new_row['MACD'] = macd
            elif c == 'RSI': new_row['RSI'] = rsi
            elif c == 'Volatility': new_row['Volatility'] = volatility
            elif c == 'Momentum': new_row['Momentum'] = momentum
            elif c == 'Volume': new_row['Volume'] = history_volume 
            else: new_row[c] = 0.0

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return preds

# ---------------------------
# SHAP & LIME wrappers
# ---------------------------

def compute_shap_for_lstm(model: tf.keras.models.Model,
                          X_train_scaled: np.ndarray,
                          X_all_scaled: np.ndarray,
                          scalers: Scalers,
                          feature_cols: List[str],
                          background_size: int = 30) -> Dict[str, Any]:
    """Compute shap values using KernelExplainer exclusively to resolve gradient registry errors."""
    
    # CRITICAL FIX: Localize SHAP import to avoid global TF gradient registry conflict
    try:
        import shap 
    except ImportError:
        return {'feature_names': feature_cols, 'shap_values': np.zeros(len(feature_cols))}
        
    if X_train_scaled.shape[0] == 0:
        return {'feature_names': feature_cols, 'shap_values': np.zeros(len(feature_cols))}
    
    # 1. Pick background and sample
    bsize = min(background_size, X_train_scaled.shape[0])
    bg_idx = np.random.choice(X_train_scaled.shape[0], bsize, replace=False)
    background = X_train_scaled[bg_idx]  
    X_sample = X_all_scaled[-1:]         
    
    # 2. Kernel Explainer Setup 
    
    background_flat = background.reshape(background.shape[0], -1)
    sample_flat = X_sample.reshape(1, -1)
    
    def pred_fn(x):
        n = x.shape[0]
        x_reshaped = x.reshape((n, X_sample.shape[1], X_sample.shape[2]))
        preds = model.predict(x_reshaped, verbose=0) 
        return preds.flatten()
        
    explainer = shap.KernelExplainer(pred_fn, background_flat)
    
    # Use increased nsamples for better Kernel Explainer accuracy
    shap_vals = explainer.shap_values(sample_flat, nsamples=150) 
    
    # Reshape and average across time steps
    arr = np.array(shap_vals).reshape(X_sample.shape[1], X_sample.shape[2])
    shap_per_feature = arr.mean(axis=0)

    return {'feature_names': feature_cols, 'shap_values': shap_per_feature}

def compute_lime_for_svm(svm_model: SVC, svm_scaler: StandardScaler, feature_names: List[str],
                         training_data_raw: np.ndarray, instance_raw: np.ndarray, class_names: List[str]) -> Any:
    """Reworked LIME: Uses raw training data as context for better local model fit."""
    def predict_proba_fn(x_raw):
        x_scaled = svm_scaler.transform(x_raw)
        return svm_model.predict_proba(x_scaled)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_data_raw,  
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )
    exp = explainer.explain_instance(
        instance_raw, 
        predict_proba_fn, 
        num_features=min(10, len(feature_names)),
        num_samples=1000 
    )
    return exp


# ---------------------------
# Training Function (Final, Stable Version)
# ---------------------------

@st.cache_resource(show_spinner=False)
def prepare_and_train(X_windows, y_targets, previous_closes_for_pred, df_full, epochs_local, force_retrain, ticker_name, window_size_val):
    
    # CRITICAL FIX: Clear the session at the start to remove SHAP's gradient registrations
    tf.keras.backend.clear_session()
    
    scalers = Scalers()
    scalers.fit(X_windows, y_targets)
    Xs_scaled = scalers.transform_X(X_windows)
    ys_scaled = scalers.transform_y(y_targets)

    # Time splits
    n = Xs_scaled.shape[0]
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    X_train = Xs_scaled[:train_end]
    X_val = Xs_scaled[train_end:val_end]
    X_test = Xs_scaled[val_end:]
    y_train = ys_scaled[:train_end]
    y_val = ys_scaled[train_end:val_end]
    y_test = ys_scaled[val_end:]
    
    prev_closes_test = previous_closes_for_pred[val_end:]

    # --- LSTM model ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    cb = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True) 
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_local, batch_size=32, callbacks=[cb], verbose=0)
    
    # Save model weights
    model_hash = hash((ticker_name, window_size_val, epochs_local, tuple(feature_cols)))
    model_path = os.path.join(tempfile.gettempdir(), f'lstm_weights_{model_hash}.weights.h5')
    model.save_weights(model_path)
    
    # LSTM Test Evaluation 
    model_for_rmse = build_lstm_model(input_shape)
    model_for_rmse.load_weights(model_path) 
    
    true_closes = df_full['Close'].values 
    true_next_day_closes = true_closes[val_end + 1:] 
    
    y_pred_scaled = model_for_rmse.predict(X_test, verbose=0)
    
    # Clear the session again after using the model for RMSE to protect the environment
    tf.keras.backend.clear_session()
    
    y_pred_change_unscaled = scalers.inverse_transform_y(y_pred_scaled.flatten())
    y_pred_unscaled_abs = prev_closes_test * (1 + (y_pred_change_unscaled / 100))

    min_len = min(len(y_pred_unscaled_abs), len(true_next_day_closes))
    lstm_rmse = np.sqrt(mean_squared_error(true_next_day_closes[:min_len], y_pred_unscaled_abs[:min_len]))
    
    # --- SVM training ---
    X_svm_raw_full = X_windows[:, -1, :] 
    closes = df_full['Close'].values
    opens = df_full['Open'].values
    start_idx = window_size_val
    
    future_close = closes[start_idx:]        
    next_open = opens[start_idx:]           
    
    minlen = min(len(future_close), len(next_open), X_svm_raw_full.shape[0])
    X_svm_raw = X_svm_raw_full[:minlen]
    labels = (future_close[:minlen] > next_open[:minlen]).astype(int) 

    split_idx = int(len(labels) * 0.8)
    
    X_svm_train_raw = X_svm_raw[:split_idx] 
    X_svm_raw_latest = X_svm_raw[-1:] 

    svm_scaler = StandardScaler()
    X_svm_scaled = svm_scaler.fit_transform(X_svm_raw)
    X_svm_train = X_svm_scaled[:split_idx]
    y_svm_train = labels[:split_idx]

    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_svm_train, y_svm_train)
    
    svm_train_acc = accuracy_score(y_svm_train, svm.predict(X_svm_train))
    
    X_svm_test = X_svm_scaled[split_idx:]
    y_svm_test = labels[split_idx:]
    svm_test_acc = accuracy_score(y_svm_test, svm.predict(X_svm_test))

    return {
        'model_path': model_path,
        'scalers': scalers,
        'Xs_scaled': Xs_scaled,
        'X_train_scaled': X_train,
        'svm_scaler': svm_scaler,
        'svm': svm,
        'svm_train_acc': svm_train_acc,
        'svm_test_acc': svm_test_acc,
        'lstm_rmse': lstm_rmse,
        'feature_cols': feature_cols,
        'X_svm_train_raw': X_svm_train_raw,
        'X_svm_raw_latest': X_svm_raw_latest
    }


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🧠 AI Stock Trend Predictor — Final Robust Version")
st.markdown("Predicts **daily percentage change** for stability. SVM predicts **intraday direction**.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    period = st.selectbox("Historical period", options=["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Interval", options=["1d", "1wk"], index=0)
    window_size = st.slider("Window size (days)", 10, 90, 60, 5) 
    forecast_days = st.slider("Forecast horizon (days)", 1, 30, 7)
    epochs = st.slider("LSTM epochs", 5, 120, 50, 5)
    st.markdown("---")
    st.subheader("Quick Info")
    st.checkbox("Force retrain models (disable cache)", value=False, key="retrain_check") 

# Fetch data and prepare features
st.subheader(f"📈 Fetching data for {ticker} ({period})")
try:
    df_raw = download_data(ticker, period=period, interval=interval)
except ValueError as e:
    st.error(f"Data error: {e}")
    st.stop()
    
if df_raw.empty:
    st.error("No data returned for ticker. Try another ticker or longer period.")
    st.stop()

# Add indicators and target feature
df = add_indicators(df_raw)
df_full = df.copy() 
st.write("Data preview (last 6 rows):")
st.dataframe(df_raw.tail(6))

# Historical Stock Value Chart for the full period
st.subheader(f"📅 Historical Price Chart ({period})")
st.line_chart(df_raw['Close'])

# Prepare features
feature_cols = ['Close', 'Close_Change', 'Volume', 'MA7', 'MA21', 'EMA12', 'EMA26', 'MACD', 'RSI', 'Volatility', 'Momentum']
target_col = 'Close_Change'
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0.0

X_windows, y_targets, previous_closes_for_pred = create_windows(df, feature_cols, target_col, window_size)

if X_windows.shape[0] < 10:
    st.warning("Not enough data to create windows — try a longer period or smaller window.")
    st.stop()

st.write(f"Prepared {X_windows.shape[0]} windows (window_size={window_size})")


# Train/Load Models
with st.spinner("Training (LSTM) and fitting SVM — please wait..."):
    force_retrain_flag = st.session_state.get('retrain_check', False)
    meta = prepare_and_train(X_windows, y_targets, previous_closes_for_pred, df_full, epochs, force_retrain_flag, ticker, window_size)
    
# --- RE-LOAD LSTM MODEL FOR INFERENCE AND SHAP ---
input_shape = (X_windows.shape[1], X_windows.shape[2])
model = load_lstm_model(input_shape, meta['model_path'])

# Unpack other meta data
scalers = meta['scalers']
Xs_scaled_all = meta['Xs_scaled']
X_train_scaled = meta['X_train_scaled']
svm_scaler = meta['svm_scaler']
svm_model = meta['svm']
feature_cols = meta['feature_cols']
X_svm_train_raw = meta['X_svm_train_raw']
latest_row_raw = meta['X_svm_raw_latest'].flatten()

st.success("Models ready ✅")

# Quick stats
last_close = float(df_full['Close'].iloc[-1])
last_date = df_full.index[-1]
st.sidebar.markdown(f"**Last close:** {last_close:.2f}")
st.sidebar.markdown(f"**Last date:** {last_date.date()}")
st.sidebar.markdown(f"**LSTM Target:** % Change")


# --- Model Performance Display ---
st.subheader("Model Performance Snapshot (Test Set)")
col_lstm, col_svm1, col_svm2 = st.columns(3)

with col_lstm:
    st.metric("LSTM Test RMSE", f"${meta['lstm_rmse']:.2f}") 
    st.info("RMSE is the average dollar error of the absolute price prediction.")
    
with col_svm1:
    st.metric("SVM Train Accuracy", f"{meta['svm_train_acc']*100:.2f}%")
with col_svm2:
    st.metric("SVM Test Accuracy", f"{meta['svm_test_acc']*100:.2f}%")


# Inference: LSTM next-day prediction (Percent Change)
last_window_raw = X_windows[-1]  
last_window_scaled = scalers.feature_scaler.transform(last_window_raw.reshape(window_size, -1)).reshape(1, window_size, len(feature_cols))
pred_change_scaled = model.predict(last_window_scaled, verbose=0).flatten()[0]
pred_change_unscaled = scalers.inverse_transform_y(np.array([pred_change_scaled]))[0]

# Convert predicted change back to absolute price
pred_next = last_close * (1 + (pred_change_unscaled / 100))
pred_delta = pred_next - last_close


# SVM inference: use last-row raw 
last_row_for_svm = last_window_raw[-1].reshape(1, -1)
last_row_svm_scaled = svm_scaler.transform(last_row_for_svm)
svm_pred_label = int(svm_model.predict(last_row_svm_scaled)[0])
svm_pred_proba = float(svm_model.predict_proba(last_row_svm_scaled)[0][1])  

# Display predictions
st.subheader("🔮 Predictions for Next Day")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric("Last Close", f"{last_close:.2f}")
with col2:
    st.metric("LSTM Predicted Close", f"{pred_next:.2f}", delta=f"{pred_delta:.2f}")
    st.markdown(f"**Predicted Change:** **{pred_change_unscaled:.2f}%**")
with col3:
    trend_text = "UP (Close > Open)" if svm_pred_label == 1 else "DOWN/HOLD (Close ≤ Open)"
    st.metric("SVM Trend (Intraday)", trend_text, f"{svm_pred_proba*100:.2f}% confidence")
    st.markdown("SVM predicts **Next Day Close vs Open**.")


# SHAP explanation for LSTM (last window)
with st.spinner("Computing SHAP for LSTM (mean contribution over window)..."):
    shap_res = compute_shap_for_lstm(model, X_train_scaled, Xs_scaled_all, scalers, feature_cols, background_size=30)

shap_df = pd.DataFrame({
    'feature': shap_res['feature_names'],
    'shap': shap_res['shap_values']
})
shap_df['abs'] = np.abs(shap_df['shap'])
shap_df = shap_df.sort_values('abs', ascending=False)

# FIX: Set the threshold to a very small epsilon (1e-8) to only hide absolute zero values.
MIN_SHAP_THRESHOLD = 1e-8 
shap_df_filtered = shap_df[shap_df['abs'] > MIN_SHAP_THRESHOLD]


st.subheader("🧠 SHAP (LSTM) — Feature Contribution to **Percent Change** Prediction (Filtered)")
if shap_df_filtered.empty:
    st.warning("All features had negligible contribution to the latest prediction. Check model performance (RMSE).")
else:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(shap_df_filtered['feature'][::-1], shap_df_filtered['shap'][::-1])
    ax.set_xlabel("SHAP value (impact on predicted % change)")
    ax.set_title("SHAP Feature Contributions (LSTM)")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------
# LIME explanation for SVM 
# ---------------------------------------------
st.subheader("💡 LIME (SVM) — Local Explanation for Latest Day's Features")
st.markdown("LIME explains the prediction of **UP (Close > Open)** or **DOWN (Close ≤ Open)** using today's market features.")

with st.spinner("Computing LIME explanation for SVM (using full training set context)..."):
    try:
        class_names_lime = ['DOWN / HOLD', 'UP (Close > Open)']
        lime_exp = compute_lime_for_svm(
            svm_model, svm_scaler, feature_cols, X_svm_train_raw, latest_row_raw, class_names_lime
        )
        
        col_data, col_chart = st.columns([1, 2])
        
        with col_data:
            # Display Feature Values (Visible and contrasted data table)
            st.markdown("### Feature Values for Prediction")
            
            data_df = pd.DataFrame({
                'Feature': feature_cols,
                'Value': latest_row_raw.round(4) 
            })
            data_df['Value'] = data_df['Value'].apply(lambda x: f"{x:,.2f}" if abs(x) > 1000 else f"{x:.4f}")

            st.dataframe(data_df.set_index('Feature'), use_container_width=True)
            
            # Display Prediction Probability
            st.markdown("### Prediction Probability")
            st.metric(
                "Predicted Trend", 
                f"{'UP' if svm_pred_label == 1 else 'DOWN'}", 
                f"{svm_pred_proba*100:.2f}% Confidence (UP)"
            )

        with col_chart:
            # Explicitly plot the LIME explanation figure
            st.markdown("### Feature Contributions to Trend")
            
            lime_fig = lime_exp.as_pyplot_figure()
            plt.tight_layout()
            st.pyplot(lime_fig)
            plt.close(lime_fig)

    except Exception as e:
        st.warning(f"LIME explanation failed: {e}")
        st.error(f"Error details: {e}")


# Recursive forecast (recompute indicators)
with st.spinner(f"Running recursive multi-day forecast for {forecast_days} days..."):
    preds_recursive = recursive_forecast_with_recalc(model, scalers, df_full.reset_index(drop=True), feature_cols, window_size, forecast_days)

# Ensure forecast dates are strictly future days (business days not enforced — daily frequency)
start_date = df_full.index[-1]
future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

st.subheader(f"📅 Recursive Forecast: Next {forecast_days} Days")
fc_df = pd.DataFrame({'date': future_dates, 'predicted_close': preds_recursive})
st.dataframe(fc_df)

# Plot history + forecast
st.subheader("Historical vs Forecast Plot (Last 60 Days + Forecast)")
fig2, ax2 = plt.subplots(figsize=(10,5))

# FIX: Set hist_window explicitly to 60 days
hist_window = 60

hist_series = df_full['Close'].iloc[-hist_window:].reset_index(drop=True)
ax2.plot(range(len(hist_series)), hist_series.values, label='Historical Close')
hist_len = len(hist_series)
ax2.plot(range(hist_len, hist_len + len(preds_recursive)), preds_recursive, linestyle='--', marker='o', label='Forecast')
ax2.legend()
ax2.set_xlabel("Days (relative index)")
ax2.set_ylabel("Price")
ax2.set_title(f"{ticker} Historical vs Forecast (last {hist_window} days)")
st.pyplot(fig2)
plt.close(fig2)

# Portfolio simulation
st.subheader("📈 Portfolio Simulation (Based on Forecasted Close)")
shares = st.number_input("Shares to simulate", min_value=1, value=10, step=1)
entry_price = last_close
predicted_values = np.array(preds_recursive) * shares
entry_value = entry_price * shares
profit = predicted_values - entry_value
sim_df = pd.DataFrame({'date': future_dates, 'predicted_value': predicted_values, 'profit': profit})
st.dataframe(sim_df)
if len(profit) > 0:
    st.info(f"Projected total portfolio value change after {forecast_days} days: **{profit[-1]:.2f}**")

st.markdown("---")
st.caption("Prototype: tune hyperparams and test thoroughly before any real use. This app runs on CPU and is not investment advice.")