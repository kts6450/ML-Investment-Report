import os
from dotenv import load_dotenv

load_dotenv()


# ── PostgreSQL ──
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "crypto_portfolio")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── 거래소 ──
EXCHANGE = os.getenv("EXCHANGE", "binance")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "")

# ── 데이터 저장 모드 ──
# "db" = PostgreSQL 사용, "csv" = CSV 파일 사용 (DB 없이 테스트 가능)
DATA_MODE = os.getenv("DATA_MODE", "csv")
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_csv")

# ── 자산 목록 ──
ASSETS = ["BTC", "ETH"]  # CASH는 별도 처리
SYMBOLS_BINANCE = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
SYMBOLS_UPBIT = {"BTC": "KRW-BTC", "ETH": "KRW-ETH"}

# ── 강화학습 ──
WINDOW_SIZE = 20        # 관측 윈도우 크기 (과거 N일)
INITIAL_BALANCE = 10_000  # 초기 자산 (USD 기준)
TRANSACTION_FEE = 0.001   # 거래 수수료 0.1%

# ── 학습 ──
TOTAL_TIMESTEPS = 500_000
LEARNING_RATE = 3e-4
SEED = 42

# ── 모델 저장 ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_portfolio")

# ── FastAPI ──
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
