"""기술지표 기반 Feature Engineering.

생성되는 피처 (자산별 10개):
  - daily_return  : 일간 수익률
  - ma5_ratio     : 5일 이동평균 / 종가 비율
  - ma20_ratio    : 20일 이동평균 / 종가 비율
  - rsi           : 14일 RSI (0~1 스케일)
  - volatility    : 20일 수익률 표준편차
  - volume_change : 거래량 변화율
  - macd          : MACD (12/26 EMA 차이, 종가 대비 비율)
  - macd_signal   : MACD 시그널 (9일 EMA)
  - bb_position   : 볼린저 밴드 내 위치 (0~1)
  - momentum      : 10일 모멘텀 (가격 변화율)

교차 피처 (2개):
  - btc_eth_corr  : BTC-ETH 20일 수익률 상관계수
  - btc_eth_spread: BTC-ETH 수익률 스프레드
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame에 기술지표 컬럼 추가."""
    out = df.copy()

    out["daily_return"] = out["close"].pct_change()

    out["ma5_ratio"] = out["close"].rolling(5).mean() / out["close"] - 1
    out["ma20_ratio"] = out["close"].rolling(20).mean() / out["close"] - 1

    out["rsi"] = _compute_rsi(out["close"], period=14) / 100.0

    out["volatility"] = out["daily_return"].rolling(20).std()

    out["volume_change"] = out["volume"].pct_change()

    ema12 = out["close"].ewm(span=12).mean()
    ema26 = out["close"].ewm(span=26).mean()
    macd_line = ema12 - ema26
    out["macd"] = macd_line / out["close"]
    out["macd_signal"] = (macd_line.ewm(span=9).mean()) / out["close"]

    ma20 = out["close"].rolling(20).mean()
    std20 = out["close"].rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    out["bb_position"] = (out["close"] - bb_lower) / (bb_upper - bb_lower)

    out["momentum"] = out["close"].pct_change(10)

    out = out.dropna().copy()
    return out


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI 계산."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_feature_matrix(ohlcv_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    여러 자산의 OHLCV dict → 하나의 피처 행렬로 합침.

    Parameters
    ----------
    ohlcv_dict : {"BTC": DataFrame, "ETH": DataFrame}

    Returns
    -------
    정규화된 피처 행렬 (index=date)
    """
    feature_cols = [
        "daily_return", "ma5_ratio", "ma20_ratio", "rsi", "volatility",
        "volume_change", "macd", "macd_signal", "bb_position", "momentum",
    ]

    frames = []
    for asset, df in ohlcv_dict.items():
        feat = add_features(df)
        subset = feat[feature_cols].copy()
        subset.columns = [f"{asset}_{c}" for c in feature_cols]
        frames.append(subset)

    merged = pd.concat(frames, axis=1, join="inner")

    # 교차 피처: BTC-ETH 상관계수 & 수익률 스프레드
    btc_ret = merged["BTC_daily_return"]
    eth_ret = merged["ETH_daily_return"]
    merged["btc_eth_corr"] = btc_ret.rolling(20).corr(eth_ret)
    merged["btc_eth_spread"] = btc_ret - eth_ret

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    return normalize_features(merged)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score 정규화. 평균/표준편차를 속성으로 저장해 추론 시 재사용."""
    result = df.copy()
    result.attrs["mean"] = df.mean().to_dict()
    result.attrs["std"] = df.std().to_dict()
    for col in df.columns:
        std = df[col].std()
        if std > 0:
            result[col] = (df[col] - df[col].mean()) / std
    return result
