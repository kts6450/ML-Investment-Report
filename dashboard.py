"""Streamlit 대시보드 – 포트폴리오 최적화 시스템 시각화.

실행: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from stable_baselines3 import PPO

from crypto_portfolio.config.settings import MODEL_PATH, WINDOW_SIZE, INITIAL_BALANCE
from crypto_portfolio.data.loader import load_ohlcv
from crypto_portfolio.features.engineering import build_feature_matrix
from crypto_portfolio.envs.portfolio_env import PortfolioEnv
from crypto_portfolio.backtest.backtester import (
    cumulative_return,
    sharpe_ratio,
    max_drawdown,
)
from crypto_portfolio.training.train import prepare_data

# ────────────────────────────────────────────
# 페이지 설정
# ────────────────────────────────────────────

st.set_page_config(
    page_title="Crypto Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3d3d5c;
    }
    .stMetric > div {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d3d5c;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────
# 데이터 로드 (캐싱)
# ────────────────────────────────────────────

@st.cache_data
def load_all_data():
    ohlcv = load_ohlcv()
    feature_df = build_feature_matrix(ohlcv)
    _, _, test_feat, test_price = prepare_data()
    return ohlcv, feature_df, test_feat, test_price


@st.cache_resource
def load_model():
    return PPO.load(MODEL_PATH)


@st.cache_data
def run_backtest_cached(_test_feat, _test_price):
    model = load_model()
    env = PortfolioEnv(_test_feat, _test_price)
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
    window = env.window
    btc_prices = _test_price[window:window + len(ppo_values), 0]
    bh_values = INITIAL_BALANCE * (btc_prices / btc_prices[0])

    min_len = min(len(ppo_values), len(bh_values))
    ppo_values = ppo_values[:min_len]
    bh_values = bh_values[:min_len]

    return ppo_values, bh_values, np.array(weights_history[:min_len])


def get_recommendation(current_weights):
    model = load_model()
    ohlcv = load_ohlcv()
    feature_df = build_feature_matrix(ohlcv)
    features = feature_df.values

    recent = features[-WINDOW_SIZE:]
    if len(recent) < WINDOW_SIZE:
        recent = np.pad(recent, ((WINDOW_SIZE - len(recent), 0), (0, 0)), mode="edge")

    cw = np.array(current_weights, dtype=np.float32)
    obs = np.concatenate([recent.flatten(), cw]).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)

    e = np.exp(action - np.max(action))
    return e / e.sum()


# ────────────────────────────────────────────
# 메인 UI
# ────────────────────────────────────────────

st.title("📊 Crypto Portfolio Optimizer")
st.caption("PPO 강화학습 기반 가상자산 포트폴리오 비중 최적화 시스템")

ohlcv, feature_df, test_feat, test_price = load_all_data()

# ── 탭 구성 ──
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 추천 비중", "📈 백테스트 결과", "📉 시장 데이터", "⚙️ 모델 정보"
])


# ════════════════════════════════════════════
# 탭 1: 추천 비중
# ════════════════════════════════════════════

with tab1:
    st.header("실시간 포트폴리오 추천")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("현재 비중 입력")
        c_btc = st.slider("BTC 비중", 0.0, 1.0, 0.0, 0.01)
        c_eth = st.slider("ETH 비중", 0.0, 1.0, 0.0, 0.01)
        c_cash = max(0.0, 1.0 - c_btc - c_eth)
        st.metric("CASH 비중 (자동)", f"{c_cash:.0%}")

        if c_btc + c_eth > 1.0:
            st.error("BTC + ETH 비중이 100%를 초과합니다.")

    with col_result:
        weights = get_recommendation([c_btc, c_eth, c_cash])

        st.subheader("PPO 추천 비중")

        m1, m2, m3 = st.columns(3)
        m1.metric("🟠 BTC", f"{weights[0]:.1%}", f"{weights[0] - c_btc:+.1%}")
        m2.metric("🔵 ETH", f"{weights[1]:.1%}", f"{weights[1] - c_eth:+.1%}")
        m3.metric("🟢 CASH", f"{weights[2]:.1%}", f"{weights[2] - c_cash:+.1%}")

        fig_pie = go.Figure(data=[go.Pie(
            labels=["BTC", "ETH", "CASH"],
            values=[weights[0], weights[1], weights[2]],
            marker=dict(colors=["#F7931A", "#627EEA", "#26A17B"]),
            hole=0.4,
            textinfo="label+percent",
            textfont_size=16,
        )])
        fig_pie.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════
# 탭 2: 백테스트 결과
# ════════════════════════════════════════════

with tab2:
    st.header("백테스트: PPO Portfolio vs BTC Buy-and-Hold")

    ppo_values, bh_values, weights_hist = run_backtest_cached(test_feat, test_price)

    ppo_ret = np.diff(ppo_values) / ppo_values[:-1]
    bh_ret = np.diff(bh_values) / bh_values[:-1]

    # 지표 카드
    col1, col2, col3 = st.columns(3)

    ppo_cum = cumulative_return(ppo_values)
    bh_cum = cumulative_return(bh_values)
    col1.metric(
        "누적 수익률",
        f"{ppo_cum:.2%}",
        f"B&H 대비 {ppo_cum - bh_cum:+.2%}",
        delta_color="normal" if ppo_cum > bh_cum else "inverse",
    )

    ppo_sr = sharpe_ratio(ppo_ret)
    bh_sr = sharpe_ratio(bh_ret)
    col2.metric(
        "Sharpe Ratio",
        f"{ppo_sr:.4f}",
        f"B&H 대비 {ppo_sr - bh_sr:+.4f}",
        delta_color="normal" if ppo_sr > bh_sr else "inverse",
    )

    ppo_mdd = max_drawdown(ppo_values)
    bh_mdd = max_drawdown(bh_values)
    col3.metric(
        "Max Drawdown",
        f"{ppo_mdd:.2%}",
        f"B&H 대비 {ppo_mdd - bh_mdd:+.2%}",
        delta_color="normal" if ppo_mdd > bh_mdd else "inverse",
    )

    # 누적 수익률 차트
    fig_bt = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=["포트폴리오 가치 ($)", "비중 배분"],
    )

    fig_bt.add_trace(
        go.Scatter(y=ppo_values, name="PPO Portfolio", line=dict(color="#00D4AA", width=2.5)),
        row=1, col=1,
    )
    fig_bt.add_trace(
        go.Scatter(y=bh_values, name="BTC Buy-and-Hold", line=dict(color="#F7931A", width=2, dash="dash")),
        row=1, col=1,
    )

    if len(weights_hist) > 0:
        x = list(range(len(weights_hist)))
        fig_bt.add_trace(
            go.Scatter(x=x, y=weights_hist[:, 0], name="BTC", stackgroup="one",
                       fillcolor="rgba(247,147,26,0.7)", line=dict(width=0)),
            row=2, col=1,
        )
        fig_bt.add_trace(
            go.Scatter(x=x, y=weights_hist[:, 1], name="ETH", stackgroup="one",
                       fillcolor="rgba(98,126,234,0.7)", line=dict(width=0)),
            row=2, col=1,
        )
        fig_bt.add_trace(
            go.Scatter(x=x, y=weights_hist[:, 2], name="CASH", stackgroup="one",
                       fillcolor="rgba(38,161,123,0.7)", line=dict(width=0)),
            row=2, col=1,
        )

    fig_bt.update_layout(
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    fig_bt.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig_bt.update_yaxes(title_text="Weight", range=[0, 1], row=2, col=1)
    fig_bt.update_xaxes(title_text="Time Step", row=2, col=1)

    st.plotly_chart(fig_bt, use_container_width=True)

    # 비교 테이블
    st.subheader("상세 지표 비교")
    comparison = pd.DataFrame({
        "지표": ["누적 수익률", "Sharpe Ratio", "Max Drawdown"],
        "PPO Portfolio": [f"{ppo_cum:.2%}", f"{ppo_sr:.4f}", f"{ppo_mdd:.2%}"],
        "BTC Buy-and-Hold": [f"{bh_cum:.2%}", f"{bh_sr:.4f}", f"{bh_mdd:.2%}"],
        "PPO 우위": [f"{ppo_cum - bh_cum:+.2%}", f"{ppo_sr - bh_sr:+.4f}", f"{ppo_mdd - bh_mdd:+.2%}"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# 탭 3: 시장 데이터
# ════════════════════════════════════════════

with tab3:
    st.header("시장 데이터 및 기술지표")

    asset = st.selectbox("자산 선택", ["BTC", "ETH"])
    df = ohlcv[asset].copy()

    # 캔들스틱 차트
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=asset,
    )])
    fig_candle.update_layout(
        title=f"{asset}/USDT 일봉 차트",
        height=450,
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # 피처 시각화
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        rsi_col = f"{asset}_rsi"
        if rsi_col in feature_df.columns:
            fig_rsi = px.line(feature_df, y=rsi_col, title=f"{asset} RSI (정규화)")
            fig_rsi.update_layout(height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)

    with col_f2:
        vol_col = f"{asset}_volatility"
        if vol_col in feature_df.columns:
            fig_vol = px.line(feature_df, y=vol_col, title=f"{asset} 변동성 (정규화)")
            fig_vol.update_layout(height=300)
            st.plotly_chart(fig_vol, use_container_width=True)

    # 최근 데이터
    st.subheader(f"{asset} 최근 10일 데이터")
    st.dataframe(df.tail(10).round(2), use_container_width=True)


# ════════════════════════════════════════════
# 탭 4: 모델 정보
# ════════════════════════════════════════════

with tab4:
    st.header("모델 및 학습 정보")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.subheader("강화학습 환경")
        st.markdown("""
        | 요소 | 설명 |
        |------|------|
        | **State** | 기술지표 22개 × 20일 + 현재비중 3개 = **443차원** |
        | **Action** | 3개 실수 → softmax → BTC/ETH/CASH 비중 |
        | **Reward** | Diff. Sharpe × 0.5 + Return × 10 - Turnover Penalty |
        | **Episode** | 전체 학습기간 1회 시뮬레이션 |
        """)

    with col_info2:
        st.subheader("PPO 하이퍼파라미터")
        st.markdown("""
        | 파라미터 | 값 |
        |----------|-----|
        | **Network** | MLP 256-256-128 (Tanh) |
        | **Timesteps** | 500,000 |
        | **Learning Rate** | 3e-4 → 0 (linear decay) |
        | **Batch Size** | 128 |
        | **Gamma** | 0.995 |
        | **Entropy Coef** | 0.005 |
        """)

    st.subheader("Feature 목록 (총 22개)")
    features_info = pd.DataFrame({
        "피처": [
            "daily_return", "ma5_ratio", "ma20_ratio", "rsi", "volatility",
            "volume_change", "macd", "macd_signal", "bb_position", "momentum",
            "btc_eth_corr", "btc_eth_spread",
        ],
        "설명": [
            "일간 종가 수익률", "5일 MA / 종가 - 1", "20일 MA / 종가 - 1",
            "14일 RSI (0~1)", "20일 수익률 표준편차", "거래량 변화율",
            "MACD (종가 대비)", "MACD 시그널 (종가 대비)",
            "볼린저밴드 내 위치 (0~1)", "10일 모멘텀",
            "BTC-ETH 20일 상관계수", "BTC-ETH 수익률 차이",
        ],
        "유형": [
            "자산별", "자산별", "자산별", "자산별", "자산별",
            "자산별", "자산별", "자산별", "자산별", "자산별",
            "교차", "교차",
        ],
    })
    st.dataframe(features_info, use_container_width=True, hide_index=True)

    st.subheader("시스템 아키텍처")
    st.code("""
[Binance API] → [데이터 수집] → [CSV/DB 저장]
                                      │
                                      ▼
                            [Feature Engineering]
                            (22개 기술지표 + Z-score)
                                      │
                                      ▼
                            [Gymnasium 환경]
                            (Diff. Sharpe Reward)
                                      │
                                      ▼
                             [PPO 학습 (SB3)]
                             (256-256-128, 500k)
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                        [백테스트]      [FastAPI 서버]
                                              │
                                              ▼
                                    [n8n → Discord 알림]
    """, language=None)
