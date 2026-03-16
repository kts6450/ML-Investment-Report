"""Gymnasium 환경 – 포트폴리오 비중 최적화.

State  : [기술지표 벡터] + [현재 포트폴리오 비중 3개]
Action : Dirichlet-softmax → BTC, ETH, CASH 비중 (합=1)
Reward : 리스크 조정 수익률 (differential Sharpe ratio) - 거래비용
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crypto_portfolio.config.settings import (
    INITIAL_BALANCE,
    TRANSACTION_FEE,
    WINDOW_SIZE,
)


class PortfolioEnv(gym.Env):
    """
    Parameters
    ----------
    feature_matrix : np.ndarray, shape (T, n_features)
    price_data : np.ndarray, shape (T, 2)  – BTC, ETH 종가
    window_size : int
    tx_cost_penalty : float – 거래비용 배수 (클수록 리밸런싱 억제)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_matrix: np.ndarray,
        price_data: np.ndarray,
        window_size: int = WINDOW_SIZE,
        tx_cost_penalty: float = 3.0,
    ):
        super().__init__()

        self.features = feature_matrix.astype(np.float32)
        self.prices = price_data.astype(np.float64)
        self.window = window_size
        self.n_features = feature_matrix.shape[1]
        self.n_assets = 3  # BTC, ETH, CASH
        self.tx_cost_penalty = tx_cost_penalty

        obs_dim = self.window * self.n_features + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32,
        )

        self._max_steps = len(self.features) - self.window - 1

        # Differential Sharpe ratio 추적용
        self._ema_return = 0.0
        self._ema_return_sq = 0.0
        self._eta = 0.05  # EMA decay rate

    # ──────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.weights = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.portfolio_value = INITIAL_BALANCE
        self.history = []
        self._ema_return = 0.0
        self._ema_return_sq = 0.0
        self._returns_buffer = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        new_weights = self._softmax(action)

        t = self.window + self.step_idx
        price_today = self.prices[t]
        price_tomorrow = self.prices[t + 1]

        asset_returns = np.zeros(self.n_assets)
        asset_returns[0] = (price_tomorrow[0] - price_today[0]) / price_today[0]
        asset_returns[1] = (price_tomorrow[1] - price_today[1]) / price_today[1]
        asset_returns[2] = 0.0

        turnover = np.sum(np.abs(new_weights - self.weights))
        tx_cost = turnover * TRANSACTION_FEE

        port_return = np.dot(new_weights, asset_returns) - tx_cost

        self.portfolio_value *= (1 + port_return)
        self.weights = new_weights.copy()
        self.step_idx += 1

        # ── Reward 설계 ──
        # 1) Differential Sharpe Ratio (온라인 Sharpe 근사)
        self._ema_return = (1 - self._eta) * self._ema_return + self._eta * port_return
        self._ema_return_sq = (1 - self._eta) * self._ema_return_sq + self._eta * port_return ** 2
        variance = self._ema_return_sq - self._ema_return ** 2
        std = np.sqrt(max(variance, 1e-8))
        diff_sharpe = self._ema_return / std if std > 1e-6 else 0.0

        # 2) 거래비용 페널티 (강화)
        turnover_penalty = turnover * TRANSACTION_FEE * self.tx_cost_penalty

        # 3) 최종 reward: risk-adjusted return - turnover penalty
        reward = float(diff_sharpe * 0.5 + port_return * 10.0 - turnover_penalty)

        self.history.append({
            "weights": new_weights.copy(),
            "port_return": port_return,
            "portfolio_value": self.portfolio_value,
        })

        terminated = self.step_idx >= self._max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    # ──────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        t = self.window + self.step_idx
        window_features = self.features[t - self.window : t].flatten()
        obs = np.concatenate([window_features, self.weights.astype(np.float32)])
        return obs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
