"""거래소 API로부터 일봉 OHLCV 데이터를 수집해 DB 또는 CSV에 저장."""

from datetime import datetime, timedelta
import os
import time

import pandas as pd
import requests

from crypto_portfolio.config.settings import (
    EXCHANGE,
    SYMBOLS_BINANCE,
    SYMBOLS_UPBIT,
    ASSETS,
    DATA_MODE,
    CSV_DIR,
)


# ────────────────────────────────────────────
# Binance
# ────────────────────────────────────────────

def fetch_binance_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Binance에서 일봉 데이터 조회.

    Parameters
    ----------
    symbol : 거래쌍 (예: BTCUSDT)
    start_date / end_date : "YYYY-MM-DD"
    """
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    all_rows = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for row in data:
            all_rows.append({
                "date": datetime.utcfromtimestamp(row[0] / 1000).date(),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })

        current = data[-1][0] + 86_400_000
        time.sleep(0.2)

    return pd.DataFrame(all_rows)


# ────────────────────────────────────────────
# Upbit
# ────────────────────────────────────────────

def fetch_upbit_daily(market: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Upbit에서 일봉 데이터 조회.

    Parameters
    ----------
    market : 마켓 코드 (예: KRW-BTC)
    start_date / end_date : "YYYY-MM-DD"
    """
    url = "https://api.upbit.com/v1/candles/days"
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    all_rows = []
    cursor = end_dt + timedelta(days=1)

    while cursor > start_dt:
        params = {
            "market": market,
            "to": cursor.strftime("%Y-%m-%dT00:00:00"),
            "count": 200,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        for row in data:
            d = datetime.strptime(row["candle_date_time_kst"][:10], "%Y-%m-%d").date()
            if d < start_dt.date():
                continue
            all_rows.append({
                "date": d,
                "open": float(row["opening_price"]),
                "high": float(row["high_price"]),
                "low": float(row["low_price"]),
                "close": float(row["trade_price"]),
                "volume": float(row["candle_acc_trade_volume"]),
            })

        oldest = datetime.strptime(data[-1]["candle_date_time_kst"][:10], "%Y-%m-%d")
        cursor = oldest
        time.sleep(0.2)

    return pd.DataFrame(all_rows)


# ────────────────────────────────────────────
# 저장
# ────────────────────────────────────────────

def save_to_csv(df: pd.DataFrame, symbol: str) -> int:
    """DataFrame을 CSV 파일로 저장 (기존 데이터와 병합, 중복 제거)."""
    os.makedirs(CSV_DIR, exist_ok=True)
    path = os.path.join(CSV_DIR, f"{symbol}.csv")

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["date"])
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = df.sort_values("date").reset_index(drop=True)

    new_count = len(merged) - (len(pd.read_csv(path)) if os.path.exists(path) else 0)
    merged.to_csv(path, index=False)
    return max(new_count, 0)


def save_to_db(df: pd.DataFrame, symbol: str) -> int:
    """DataFrame을 DB에 upsert (중복 무시)."""
    if df.empty:
        return 0

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from crypto_portfolio.config.database import SessionLocal
    from crypto_portfolio.data.models import OHLCVDaily

    session = SessionLocal()
    try:
        records = df.to_dict("records")
        for rec in records:
            rec["symbol"] = symbol

        stmt = pg_insert(OHLCVDaily).values(records)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_symbol_date")
        result = session.execute(stmt)
        session.commit()
        return result.rowcount
    finally:
        session.close()


# ────────────────────────────────────────────
# 메인 수집 함수
# ────────────────────────────────────────────

def collect_all(start_date: str, end_date: str):
    """
    모든 자산에 대해 데이터를 수집하고 저장.

    DATA_MODE 설정에 따라 CSV 또는 DB에 저장.
    """
    if DATA_MODE == "db":
        from crypto_portfolio.config.database import init_db
        init_db()

    symbols_map = SYMBOLS_BINANCE if EXCHANGE == "binance" else SYMBOLS_UPBIT
    fetch_fn = fetch_binance_daily if EXCHANGE == "binance" else fetch_upbit_daily
    save_fn = save_to_db if DATA_MODE == "db" else save_to_csv

    for asset in ASSETS:
        symbol = symbols_map[asset]
        print(f"[수집] {asset} ({symbol}) …")
        df = fetch_fn(symbol, start_date, end_date)
        n = save_fn(df, asset)
        print(f"  → {len(df)}행 조회, {n}행 신규 저장 ({DATA_MODE} 모드)")


if __name__ == "__main__":
    collect_all("2023-01-01", "2026-03-14")
