from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from crypto_portfolio.config.settings import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI 의존성 주입용 DB 세션 제공."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """테이블이 없으면 생성."""
    Base.metadata.create_all(bind=engine)
