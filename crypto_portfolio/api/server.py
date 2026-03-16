"""FastAPI 추론 서버 – 현재 포트폴리오 추천 비중 반환.

실행
----
uvicorn crypto_portfolio.api.server:app --reload --host 0.0.0.0 --port 8000
"""

from datetime import date

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO

from crypto_portfolio.config.settings import MODEL_PATH, WINDOW_SIZE
from crypto_portfolio.data.loader import load_ohlcv
from crypto_portfolio.features.engineering import build_feature_matrix

app = FastAPI(
    title="Crypto Portfolio Optimizer",
    description="PPO 기반 가상자산 포트폴리오 비중 추천 API",
    version="1.0.0",
)


class PortfolioWeights(BaseModel):
    date: str
    btc: float
    eth: float
    cash: float


class HealthResponse(BaseModel):
    status: str


_model: PPO | None = None


def get_model() -> PPO:
    global _model
    if _model is None:
        _model = PPO.load(MODEL_PATH)
    return _model


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@app.get("/recommend", response_model=PortfolioWeights)
def recommend_weights(current_btc: float = 0.0, current_eth: float = 0.0, current_cash: float = 1.0):
    """
    현재 포트폴리오 비중을 입력받아 PPO가 추천하는 새 비중 반환.

    Query Params
    ------------
    current_btc  : 현재 BTC 비중 (0~1)
    current_eth  : 현재 ETH 비중 (0~1)
    current_cash : 현재 CASH 비중 (0~1)
    """
    model = get_model()

    ohlcv = load_ohlcv()
    feature_df = build_feature_matrix(ohlcv)
    features = feature_df.values

    recent = features[-WINDOW_SIZE:]
    if len(recent) < WINDOW_SIZE:
        recent = np.pad(recent, ((WINDOW_SIZE - len(recent), 0), (0, 0)), mode="edge")

    current_weights = np.array([current_btc, current_eth, current_cash], dtype=np.float32)
    obs = np.concatenate([recent.flatten(), current_weights]).astype(np.float32)

    action, _ = model.predict(obs, deterministic=True)

    e = np.exp(action - np.max(action))
    weights = e / e.sum()

    today = date.today().isoformat()
    return PortfolioWeights(
        date=today,
        btc=round(float(weights[0]), 4),
        eth=round(float(weights[1]), 4),
        cash=round(float(weights[2]), 4),
    )


@app.get("/recommend/json")
def recommend_weights_json(current_btc: float = 0.0, current_eth: float = 0.0, current_cash: float = 1.0):
    """n8n 등 외부 자동화 도구 연동용 (동일 로직, dict 반환)."""
    result = recommend_weights(current_btc, current_eth, current_cash)
    return result.model_dump()
