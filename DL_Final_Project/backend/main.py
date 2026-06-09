from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Optional
import logging
import numpy as np
import onnxruntime as ort
import os
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ONNX Model Loading ───────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_FILES = {
    "LSTM":        ("LSTM_stock_model.onnx",        19),
    "GRU":         ("GRU_stock_model.onnx",         19),
    "BiLSTM":      ("BiLSTM_stock_model.onnx",      19),
    "Transformer": ("Transformer_stock_model.onnx", 19),
}

MODELS: dict[str, tuple[ort.InferenceSession, int]] = {}

def load_models():
    for key, (filename, seq_len) in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                sess = ort.InferenceSession(path)
                MODELS[key] = (sess, seq_len)
                logger.info(f"Loaded model: {key} (seq_len={seq_len})")
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
        else:
            logger.warning(f"Model file not found: {path}")

load_models()

MAX_SEQ_LEN = max((v[1] for v in MODEL_FILES.values()), default=19)

# ─── Yahoo Finance 資料抓取 ───────────────────────────────────────────────────

def fetch_yahoo(stock_no: str, months: int) -> list[dict]:
    """
    使用 yfinance 抓取台股資料
    台股代號格式：2308 → 2308.TW
    抓取範圍：months + 1 個月（補足 window 用）
    """
    symbol = f"{stock_no}.TW"

    # 多抓 1 個月補足 window
    end = datetime.now()
    start = end - timedelta(days=(months + 1) * 31)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime('%Y-%m-%d'),
                            end=end.strftime('%Y-%m-%d'),
                            interval='1d')
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Yahoo Finance 資料抓取失敗：{e}")

    if df.empty:
        raise HTTPException(status_code=404,
            detail=f"查無股票代號 {stock_no}（{symbol}）的資料，請確認代號是否正確")

    rows = []
    for ts, row in df.iterrows():
        date_str = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]
        open_  = round(float(row['Open']),   2) if pd.notna(row['Open'])   else None
        high   = round(float(row['High']),   2) if pd.notna(row['High'])   else None
        low    = round(float(row['Low']),    2) if pd.notna(row['Low'])    else None
        close  = round(float(row['Close']),  2) if pd.notna(row['Close'])  else None
        # yfinance volume 單位為「股」，換算為「張」（1張=1000股）
        volume = int(row['Volume'] / 1000)       if pd.notna(row['Volume']) else None

        rows.append({
            "date":         date_str,
            "symbol":       stock_no,
            "open":         open_,
            "high":         high,
            "low":          low,
            "close":        close,
            "volume":       volume,
            "transactions": None,   # Yahoo Finance 不提供成交筆數
        })

    rows.sort(key=lambda x: x["date"])
    return rows

# ─── Model Inference ──────────────────────────────────────────────────────────

def normalize_window(window: list[dict]) -> np.ndarray:
    arr = np.array([
        [r["open"], r["high"], r["low"], r["close"], r["volume"]]
        for r in window
    ], dtype=np.float32)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    return (arr - mins) / denom

def denormalize_price(pred_norm: float, window: list[dict]) -> float:
    closes = [r["close"] for r in window]
    c_min, c_max = min(closes), max(closes)
    if c_max == c_min:
        return c_min
    return float(pred_norm * (c_max - c_min) + c_min)

def run_predictions(all_rows: list[dict], target_dates: set) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {k: [] for k in MODELS}

    for i in range(MAX_SEQ_LEN, len(all_rows)):
        target_date = all_rows[i]["date"]
        if target_date not in target_dates:
            continue

        for model_name, (sess, seq_len) in MODELS.items():
            if i < seq_len:
                continue

            window = all_rows[i - seq_len:i]

            if any(r["open"] is None or r["close"] is None or r["volume"] is None
                   for r in window):
                continue

            x = normalize_window(window)
            x_batch = x[np.newaxis, :, :].astype(np.float32)

            try:
                input_name = sess.get_inputs()[0].name
                pred_norm = sess.run(None, {input_name: x_batch})[0][0][0]
                pred_price = denormalize_price(float(pred_norm), window)
                results[model_name].append({
                    "date": target_date,
                    "predicted_close": round(pred_price, 2),
                })
            except Exception as e:
                logger.warning(f"Inference error [{model_name}] at {target_date}: {e}")

    return results

# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_daily_change(rows: list[dict]) -> list[dict]:
    result = []
    for i, row in enumerate(rows):
        prev_close = rows[i - 1]["close"] if i > 0 else None
        close = row["close"]
        if prev_close and close and prev_close != 0:
            pct = round((close - prev_close) / prev_close * 100, 2)
        else:
            pct = 0.0
        result.append({**row, "daily_change_pct": pct})
    return result

def dedup_sort(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for r in rows:
        if r["date"] not in seen:
            seen.add(r["date"])
            unique.append(r)
    unique.sort(key=lambda x: x["date"])
    return unique

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/stock/{stock_no}")
async def get_stock_data(
    stock_no: str,
    months: int = Query(default=3, ge=1, le=12, description="幾個月的資料"),
):
    """
    取得個股每日交易資料 + 四模型預測收盤價
    stock_no: 台股代號（如 2308、2330）
    months: 1~12 個月
    """
    # 抓資料（含額外 1 個月 window）
    all_rows = fetch_yahoo(stock_no, months)
    all_rows = dedup_sort(all_rows)

    # 切出使用者要顯示的日期範圍
    cutoff = (datetime.now() - timedelta(days=months * 31)).strftime('%Y-%m-%d')
    user_rows = [r for r in all_rows if r["date"] >= cutoff]
    target_dates = {r["date"] for r in user_rows}

    if not user_rows:
        raise HTTPException(status_code=404,
            detail=f"查無股票代號 {stock_no} 的資料")

    result = compute_daily_change(user_rows)

    predictions = {}
    if MODELS:
        try:
            predictions = run_predictions(all_rows, target_dates)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")

    return {
        "stock_no": stock_no,
        "count": len(result),
        "data": result,
        "predictions": predictions,
        "models_loaded": list(MODELS.keys()),
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "models_loaded": list(MODELS.keys()),
    }