"""PPO 학습 스크립트.

개선 사항:
  - Custom MLP 네트워크 (256-256-128)
  - 학습률 linear decay 스케줄링
  - 튜닝된 PPO 하이퍼파라미터
  - 500k timesteps 학습

사용법
------
python -m crypto_portfolio.training.train
"""

import os

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from crypto_portfolio.config.settings import (
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_PATH,
    TOTAL_TIMESTEPS,
    SEED,
)
from crypto_portfolio.data.loader import load_ohlcv
from crypto_portfolio.envs.portfolio_env import PortfolioEnv
from crypto_portfolio.features.engineering import build_feature_matrix


class RewardLoggerCallback(BaseCallback):
    """에피소드 보상을 주기적으로 출력."""

    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]

        dones = self.locals["dones"]
        if dones[0]:
            self.episode_rewards.append(self.current_reward)
            if len(self.episode_rewards) % self.log_interval == 0:
                recent = self.episode_rewards[-self.log_interval:]
                mean_r = np.mean(recent)
                print(
                    f"  [Episode {len(self.episode_rewards):>4d}] "
                    f"mean_reward(last {self.log_interval})={mean_r:.6f}"
                )
            self.current_reward = 0.0
        return True


def linear_schedule(initial_value: float):
    """학습률을 선형으로 감소시키는 스케줄러."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def prepare_data(train_ratio: float = 0.8):
    """DB에서 데이터를 로드하고 train/test 분할."""
    ohlcv = load_ohlcv()

    feature_df = build_feature_matrix(ohlcv)
    dates = feature_df.index

    btc_close = ohlcv["BTC"].loc[dates, "close"].values
    eth_close = ohlcv["ETH"].loc[dates, "close"].values
    prices = np.column_stack([btc_close, eth_close])

    features = feature_df.values

    split = int(len(features) * train_ratio)
    return (
        features[:split], prices[:split],
        features[split:], prices[split:],
    )


def make_env(features, prices, seed=None):
    def _init():
        env = PortfolioEnv(features, prices)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# Custom MLP 네트워크 구조
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256, 128],  # Actor: 3층
        vf=[256, 256, 128],  # Critic: 3층
    ),
    activation_fn=th.nn.Tanh,
)


def train():
    """PPO 모델 학습 및 저장."""
    np.random.seed(SEED)

    train_feat, train_price, test_feat, test_price = prepare_data()
    print(f"학습 데이터: {len(train_feat)}일, 테스트 데이터: {len(test_feat)}일")
    print(f"피처 수: {train_feat.shape[1]}개")

    train_env = DummyVecEnv([make_env(train_feat, train_price, seed=SEED)])
    eval_env = DummyVecEnv([make_env(test_feat, test_price, seed=SEED + 1)])

    os.makedirs(MODEL_DIR, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR,
        eval_freq=5000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    reward_logger = RewardLoggerCallback(log_interval=5)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(LEARNING_RATE),
        n_steps=512,
        batch_size=128,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=linear_schedule(0.2),
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    print(f"PPO 학습 시작 (총 {TOTAL_TIMESTEPS:,} timesteps) …")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, reward_logger],
    )

    model.save(MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")
    print(f"최적 모델 위치: {os.path.join(MODEL_DIR, 'best_model.zip')}")

    return model, test_feat, test_price


if __name__ == "__main__":
    train()
