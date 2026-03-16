"""DB 또는 CSV에서 OHLCV 데이터를 DataFrame으로 로드."""

import os

import pandas as pd

from crypto_portfolio.config.settings import ASSETS, DATA_MODE, CSV_DIR


def load_ohlcv(start_date: str = None, end_date: str = None) -> dict[str, pd.DataFrame]:
    """
    자산별 OHLCV DataFrame dict 반환.

    Returns
    -------
    {"BTC": DataFrame, "ETH": DataFrame}
    각 DataFrame columns: open, high, low, close, volume  /  index: date
    """
    if DATA_MODE == "db":
        return _load_from_db(start_date, end_date)
    return _load_from_csv(start_date, end_date)


def _load_from_csv(start_date: str = None, end_date: str = None) -> dict[str, pd.DataFrame]:
    result = {}
    for asset in ASSETS:
        path = os.path.join(CSV_DIR, f"{asset}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} 파일이 없습니다. 먼저 데이터를 수집하세요: "
                "python run_pipeline.py --step collect"
            )
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        result[asset] = df
    return result


def _load_from_db(start_date: str = None, end_date: str = None) -> dict[str, pd.DataFrame]:
    from sqlalchemy import text
    from crypto_portfolio.config.database import engine

    result = {}
    for asset in ASSETS:
        query = "SELECT date, open, high, low, close, volume FROM ohlcv_daily WHERE symbol = :symbol"
        params = {"symbol": asset}

        if start_date:
            query += " AND date >= :start"
            params["start"] = start_date
        if end_date:
            query += " AND date <= :end"
            params["end"] = end_date

        query += " ORDER BY date"

        df = pd.read_sql(text(query), engine, params=params, parse_dates=["date"])
        df = df.set_index("date")
        result[asset] = df

    return result
