"""SQLAlchemy ORM 모델 – OHLCV 일봉 데이터 저장용."""

from sqlalchemy import Column, Integer, String, Float, Date, UniqueConstraint

from crypto_portfolio.config.database import Base


class OHLCVDaily(Base):
    """
    일봉 OHLCV 테이블.

    Columns
    -------
    symbol : str   – 자산 심볼 (BTC, ETH)
    date   : date  – 거래일
    open / high / low / close : float – 가격
    volume : float – 거래량
    """

    __tablename__ = "ohlcv_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_symbol_date"),
    )

    def __repr__(self):
        return f"<OHLCVDaily {self.symbol} {self.date} close={self.close}>"
