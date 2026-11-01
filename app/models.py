from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    challenge_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    challenge_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    num_qubits: Mapped[int] = mapped_column(Integer, nullable=False)
    gates: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    result: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
