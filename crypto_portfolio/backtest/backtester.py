"""백테스트 모듈 – PPO 포트폴리오 vs BTC Buy-and-Hold 비교.

산출 지표:
  - 누적 수익률
  - 연간화 Sharpe Ratio
  - Maximum Drawdown (MDD)
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from crypto_portfolio.config.settings import MODEL_PATH, INITIAL_BALANCE
from crypto_portfolio.envs.portfolio_env import PortfolioEnv


# ────────────────────────────────────────────
# 지표 계산
# ────────────────────────────────────────────

def cumulative_return(values: np.ndarray) -> float:
    return (values[-1] / values[0]) - 1


def sharpe_ratio(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = daily_returns - risk_free
    if excess.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    return float(dd.min())


# ────────────────────────────────────────────
# 백테스트 실행
# ────────────────────────────────────────────

def run_backtest(
    test_features: np.ndarray,
    test_prices: np.ndarray,
    model_path: str = MODEL_PATH,
) -> dict:
    """
    테스트 데이터로 백테스트 수행.

    Returns
    -------
    dict with keys: ppo_values, bh_values, ppo_metrics, bh_metrics, weights_history
    """
    model = PPO.load(model_path)
    env = PortfolioEnv(test_features, test_prices)
    obs, _ = env.reset()

    ppo_values = [INITIAL_BALANCE]
    weights_history = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ppo_values.append(env.portfolio_value)
        weights_history.append(env.weights.copy())

    ppo_values = np.array(ppo_values)

    # BTC Buy-and-Hold 기준선
    window = env.window
    btc_prices = test_prices[window:window + len(ppo_values), 0]
    bh_values = INITIAL_BALANCE * (btc_prices / btc_prices[0])

    min_len = min(len(ppo_values), len(bh_values))
    ppo_values = ppo_values[:min_len]
    bh_values = bh_values[:min_len]

    ppo_returns = np.diff(ppo_values) / ppo_values[:-1]
    bh_returns = np.diff(bh_values) / bh_values[:-1]

    result = {
        "ppo_values": ppo_values,
        "bh_values": bh_values,
        "weights_history": np.array(weights_history[:min_len]),
        "ppo_metrics": {
            "cumulative_return": cumulative_return(ppo_values),
            "sharpe_ratio": sharpe_ratio(ppo_returns),
            "max_drawdown": max_drawdown(ppo_values),
        },
        "bh_metrics": {
            "cumulative_return": cumulative_return(bh_values),
            "sharpe_ratio": sharpe_ratio(bh_returns),
            "max_drawdown": max_drawdown(bh_values),
        },
    }
    return result


# ────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────

def plot_backtest(result: dict, save_path: str = None):
    """누적 수익률 + 비중 변화 차트."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    # 상단: 누적 자산가치
    ax1 = axes[0]
    ax1.plot(result["ppo_values"], label="PPO Portfolio", linewidth=2)
    ax1.plot(result["bh_values"], label="BTC Buy-and-Hold", linewidth=2, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("Backtest: PPO Portfolio vs BTC Buy-and-Hold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 하단: 비중 변화
    ax2 = axes[1]
    w = result["weights_history"]
    if len(w) > 0:
        ax2.stackplot(
            range(len(w)),
            w[:, 0], w[:, 1], w[:, 2],
            labels=["BTC", "ETH", "CASH"],
            alpha=0.8,
        )
        ax2.set_ylabel("Weight")
        ax2.set_xlabel("Time Step")
        ax2.set_title("Portfolio Weight Allocation")
        ax2.legend(loc="upper right")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"차트 저장: {save_path}")

    plt.show()


def print_metrics(result: dict):
    """지표를 표로 출력."""
    print("\n" + "=" * 50)
    print("         백테스트 결과 비교")
    print("=" * 50)
    fmt = "{:<20} {:>12} {:>12}"
    print(fmt.format("지표", "PPO", "BTC B&H"))
    print("-" * 50)

    ppo = result["ppo_metrics"]
    bh = result["bh_metrics"]

    print(fmt.format("누적 수익률", f"{ppo['cumulative_return']:.2%}", f"{bh['cumulative_return']:.2%}"))
    print(fmt.format("Sharpe Ratio", f"{ppo['sharpe_ratio']:.4f}", f"{bh['sharpe_ratio']:.4f}"))
    print(fmt.format("Max Drawdown", f"{ppo['max_drawdown']:.2%}", f"{bh['max_drawdown']:.2%}"))
    print("=" * 50)
