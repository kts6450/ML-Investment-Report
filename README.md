# Crypto Portfolio Optimizer

> PPO 강화학습 기반 가상자산(BTC, ETH, CASH) 포트폴리오 비중 최적화 시스템

## 프로젝트 구조

```
ML/
├── .env.example                  # 환경변수 템플릿
├── requirements.txt              # Python 패키지
├── run_pipeline.py               # 전체 파이프라인 실행 스크립트
├── n8n_workflow_example.json     # n8n 워크플로 예제
│
├── crypto_portfolio/             # 메인 패키지
│   ├── config/
│   │   ├── settings.py           # 전역 설정 (DB, 거래소, 학습 파라미터)
│   │   └── database.py           # SQLAlchemy 엔진/세션
│   │
│   ├── data/
│   │   ├── models.py             # DB ORM 모델 (ohlcv_daily 테이블)
│   │   ├── collector.py          # Binance/Upbit API → DB/CSV 저장
│   │   └── loader.py             # DB/CSV → DataFrame 로드
│   │
│   ├── features/
│   │   └── engineering.py        # 기술지표 생성 (22개 피처)
│   │
│   ├── envs/
│   │   └── portfolio_env.py      # Gymnasium 환경 (포트폴리오 비중 최적화)
│   │
│   ├── training/
│   │   └── train.py              # PPO 학습 + 모델 저장
│   │
│   ├── backtest/
│   │   └── backtester.py         # 백테스트 + 시각화 + 지표 출력
│   │
│   └── api/
│       └── server.py             # FastAPI 추론 서버
│
├── models/                       # 학습된 PPO 모델 저장
├── data_csv/                     # CSV 데이터 저장 (DB 없이 사용 시)
└── notebooks/                    # 탐색/분석용 Jupyter 노트북
```

## 시스템 아키텍처

```
[Binance/Upbit API]
        │
        ▼
  [데이터 수집기] ──→ [PostgreSQL / CSV]
                           │
                           ▼
                  [Feature Engineering]
                  (22개 기술지표 + Z-score 정규화)
                           │
                           ▼
                  [Gymnasium 환경]
                  (Diff. Sharpe Reward)
                           │
                           ▼
                   [PPO 학습 (SB3)]
                   (256-256-128, 500k steps)
                           │
                   ┌───────┴───────┐
                   ▼               ▼
             [백테스트]      [FastAPI 서버]
          (Sharpe, MDD,         │
           B&H 비교)           ▼
                         [n8n 자동호출]
                               │
                               ▼
                          [Discord]
```

## 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. 데이터 수집

```bash
python run_pipeline.py --step collect
```

### 3. 모델 학습

```bash
python run_pipeline.py --step train
```

### 4. 백테스트

```bash
python run_pipeline.py --step backtest
```

### 5. API 서버 실행

```bash
python run_pipeline.py --step serve
# → http://localhost:8000/docs 에서 Swagger UI 확인
```

## 강화학습 환경 설계

### State (관측 공간)

| 구성 요소 | 차원 |
|-----------|------|
| 과거 20일 기술지표 (22개 × 20일) | 440 |
| 현재 포트폴리오 비중 (BTC, ETH, CASH) | 3 |
| **합계** | **443** |

### Action (행동 공간)

3개 실수 → softmax → BTC/ETH/CASH 비중 (합=1)

### Reward (보상 함수)

```
reward = diff_sharpe × 0.5 + port_return × 10.0 - turnover_penalty
```

- **Differential Sharpe Ratio**: 온라인 EMA 기반 리스크 조정 수익률
- **수익률 스케일업**: 일간 수익률 × 10 (학습 신호 강화)
- **Turnover Penalty**: 비중 변화 × 수수료 × 3배 (과도한 리밸런싱 억제)

## Feature 목록 (총 22개)

### 자산별 피처 (× 2자산 = 20개)

| Feature | 설명 |
|---------|------|
| `daily_return` | 일간 종가 수익률 |
| `ma5_ratio` | 5일 MA / 종가 - 1 |
| `ma20_ratio` | 20일 MA / 종가 - 1 |
| `rsi` | 14일 RSI (0~1 스케일) |
| `volatility` | 20일 수익률 표준편차 |
| `volume_change` | 거래량 변화율 |
| `macd` | MACD (종가 대비 비율) |
| `macd_signal` | MACD 시그널선 (종가 대비 비율) |
| `bb_position` | 볼린저 밴드 내 위치 (0~1) |
| `momentum` | 10일 모멘텀 |

### 교차 피처 (2개)

| Feature | 설명 |
|---------|------|
| `btc_eth_corr` | BTC-ETH 20일 수익률 상관계수 |
| `btc_eth_spread` | BTC-ETH 수익률 스프레드 |

## PPO 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| Network | MLP 256-256-128 (Tanh) |
| Total Timesteps | 500,000 |
| Learning Rate | 3e-4 → 0 (linear decay) |
| n_steps | 512 |
| batch_size | 128 |
| n_epochs | 15 |
| gamma | 0.995 |
| gae_lambda | 0.98 |
| clip_range | 0.2 → 0 (linear decay) |
| ent_coef | 0.005 |

## 백테스트 결과

| 지표 | PPO Portfolio | BTC Buy-and-Hold |
|------|:---:|:---:|
| 누적 수익률 | **-31.55%** | -37.93% |
| Sharpe Ratio | **-1.2029** | -1.2723 |
| Max Drawdown | **-44.67%** | -49.53% |

> 하락장에서 PPO가 CASH 비중을 높여 BTC 대비 약 6%p 손실 방어

## 기술 스택

- **데이터**: Binance/Upbit REST API, PostgreSQL / CSV
- **분석**: Pandas, NumPy
- **강화학습**: Gymnasium, Stable-Baselines3 (PPO), PyTorch
- **서버**: FastAPI, Uvicorn
- **자동화**: n8n, Discord Webhook
- **시각화**: Matplotlib
