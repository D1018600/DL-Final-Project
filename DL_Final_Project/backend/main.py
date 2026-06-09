from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime, timedelta
from typing import Optional
import logging
import numpy as np
import onnxruntime as ort
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TWSE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

# ─── ONNX Model Loading ───────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# 每個模型對應的 window 大小
MODEL_FILES = {
    "LSTM":        ("LSTM_stock_model.onnx",        19),
    "GRU":         ("GRU_stock_model.onnx",         19),
    "BiLSTM":      ("BiLSTM_stock_model.onnx",      19),
    "Transformer": ("Transformer_stock_model.onnx", 19),
}

MODELS: dict[str, tuple[ort.InferenceSession, int]] = {}  # name -> (session, seq_len)

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

MAX_SEQ_LEN = max((v[1] for v in MODEL_FILES.values()), default=20)

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
    """
    all_rows: 含補足 window 的完整資料
    target_dates: 使用者實際要顯示的日期集合
    各模型依自己的 seq_len 取 window，只回傳 target_dates 內的預測
    """
    results: dict[str, list[dict]] = {k: [] for k in MODELS}

    for i in range(MAX_SEQ_LEN, len(all_rows)):
        target_date = all_rows[i]["date"]
        if target_date not in target_dates:
            continue

        for model_name, (sess, seq_len) in MODELS.items():
            if i < seq_len:
                continue

            window = all_rows[i - seq_len:i]

            if any(r["open"] is None or r["close"] is None or r["volume"] is None for r in window):
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

def parse_number(s: str) -> Optional[float]:
    if not s or s.strip() == "--":
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None

def twse_date_to_iso(s: str) -> str:
    parts = s.strip().split("/")
    if len(parts) == 3:
        year = int(parts[0]) + 1911
        return f"{year}-{parts[1]}-{parts[2]}"
    return s

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

async def fetch_twse(stock_no: str, year: int, month: int) -> list[dict]:
    params = {
        "response": "json",
        "date": f"{year}{str(month).zfill(2)}01",
        "stockNo": stock_no,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(TWSE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("stat") != "OK" or not data.get("data"):
        return []

    rows = []
    for row in data["data"]:
        rows.append({
            "date": twse_date_to_iso(row[0]),
            "symbol": stock_no,
            "volume": int(parse_number(row[1]) / 1000) if parse_number(row[1]) else None,
            "open": parse_number(row[3]),
            "high": parse_number(row[4]),
            "low": parse_number(row[5]),
            "close": parse_number(row[6]),
            "transactions": int(parse_number(row[8])) if parse_number(row[8]) else None,
        })
    return rows

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
    months: int = Query(default=1, ge=1, le=12, description="幾個月的資料"),
):
    now = datetime.now()

    # 1. 抓使用者要求的月份
    user_rows = []
    for i in range(months - 1, -1, -1):
        target = now - timedelta(days=i * 30)
        try:
            rows = await fetch_twse(stock_no, target.year, target.month)
            user_rows.extend(rows)
        except Exception as e:
            logger.warning(f"Failed to fetch {stock_no} {target.year}/{target.month}: {e}")

    if not user_rows:
        raise HTTPException(status_code=404, detail=f"查無股票代號 {stock_no} 的資料，請確認代號或稍後再試")

    user_rows = dedup_sort(user_rows)
    target_dates = {r["date"] for r in user_rows}

    # 2. 額外往前抓 1 個月補足最大 window（MAX_SEQ_LEN = 20）
    extra_rows = []
    if MODELS:
        extra_target = now - timedelta(days=months * 30)
        try:
            extra_rows = await fetch_twse(stock_no, extra_target.year, extra_target.month)
        except Exception as e:
            logger.warning(f"Failed to fetch extra window data: {e}")

    # 3. 合併排序
    all_rows = dedup_sort(user_rows + extra_rows)

    # 4. 計算漲跌幅
    result = compute_daily_change(user_rows)

    # 5. 模型推論
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