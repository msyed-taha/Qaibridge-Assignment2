import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

load_dotenv()
os.environ.setdefault("QAI_MAX_QUBITS", "6")

from .challenges import CHALLENGES, CHALLENGES_BY_ID, Challenge, evaluate_challenge
from .database import Base, engine, get_db
from .models import SimulationRun
from .quantum_kernel import QuantumCircuit


log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
MAX_QUBITS = 6
MAX_GATES = 40
MAX_SHOTS = 2048
SINGLE_QUBIT_GATES = {"H", "X", "Y", "Z", "S", "T", "SDG", "TDG", "SX"}
ROTATION_GATES = {"RX", "RY", "RZ", "P"}
TWO_QUBIT_GATES = {"CNOT", "CZ", "SWAP"}
THREE_QUBIT_GATES = {"CCX"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Qaibridge Assignment Simulator",
    description="A small educational quantum simulator with saved PostgreSQL history.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GateOperation(BaseModel):
    gate: str = Field(min_length=1, max_length=8)
    qubit: int = Field(ge=0, le=MAX_QUBITS - 1)
    control_qubit: int | None = Field(default=None, ge=0, le=MAX_QUBITS - 1)
    control2: int | None = Field(default=None, ge=0, le=MAX_QUBITS - 1)
    angle: float | None = Field(default=None, allow_inf_nan=False)


class SimulationRequest(BaseModel):
    num_qubits: int = Field(ge=1, le=MAX_QUBITS)
    gates: list[GateOperation] = Field(default_factory=list, max_length=MAX_GATES)
    shots: int = Field(default=512, ge=0, le=MAX_SHOTS)
    challenge_id: str | None = Field(default=None, max_length=32)


def build_circuit(request: SimulationRequest) -> QuantumCircuit:
    circuit = QuantumCircuit(
        n_qubits=request.num_qubits,
        name="Assignment Circuit",
        max_gates=MAX_GATES,
    )
    for index, operation in enumerate(request.gates, start=1):
        gate = operation.gate.upper()
        target = operation.qubit
        where = f"Gate {index} ({gate})"
        if target >= request.num_qubits:
            raise HTTPException(422, f"{where}: target qubit is outside this circuit.")

        if gate in SINGLE_QUBIT_GATES:
            if operation.control_qubit is not None or operation.control2 is not None:
                raise HTTPException(422, f"{where}: single-qubit gates do not use controls.")
            if operation.angle is not None:
                raise HTTPException(422, f"{where}: this gate does not use an angle.")
            circuit._add(gate, [target])
        elif gate in ROTATION_GATES:
            if operation.control_qubit is not None or operation.control2 is not None:
                raise HTTPException(422, f"{where}: rotation gates do not use controls.")
            angle = operation.angle if operation.angle is not None else math.pi / 2
            circuit._add(gate, [target], [angle])
        elif gate in TWO_QUBIT_GATES:
            control = operation.control_qubit
            if control is None or control >= request.num_qubits or control == target:
                raise HTTPException(422, f"{where}: choose a different, valid control qubit.")
            if operation.control2 is not None or operation.angle is not None:
                raise HTTPException(422, f"{where}: unexpected extra gate parameter.")
            circuit._add(gate, [control, target])
        elif gate in THREE_QUBIT_GATES:
            control1, control2 = operation.control_qubit, operation.control2
            if (
                control1 is None
                or control2 is None
                or control1 >= request.num_qubits
                or control2 >= request.num_qubits
                or len({control1, control2, target}) != 3
            ):
                raise HTTPException(422, f"{where}: select two distinct controls and a target.")
            if operation.angle is not None:
                raise HTTPException(422, f"{where}: this gate does not use an angle.")
            circuit._add(gate, [control1, control2, target])
        else:
            raise HTTPException(422, f"Unknown gate: {operation.gate}")
    return circuit


def public_challenge(challenge: Challenge, completed: set[str]) -> dict[str, Any]:
    return {
        "id": challenge.id,
        "title": challenge.title,
        "concept": challenge.concept,
        "num_qubits": challenge.num_qubits,
        "goal": challenge.goal,
        "hint": challenge.hint,
        "completed": challenge.id in completed,
    }


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/challenges")
def list_challenges(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    completed = set(
        db.scalars(
            select(SimulationRun.challenge_id)
            .where(
                SimulationRun.challenge_passed.is_(True),
                SimulationRun.challenge_id.is_not(None),
            )
            .distinct()
        )
    )
    return [public_challenge(challenge, completed) for challenge in CHALLENGES]


@app.post("/api/simulations")
def simulate(request: SimulationRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    challenge: Challenge | None = None
    if request.challenge_id is not None:
        challenge = CHALLENGES_BY_ID.get(request.challenge_id)
        if challenge is None:
            raise HTTPException(422, "Unknown challenge.")
        if request.num_qubits != challenge.num_qubits:
            raise HTTPException(422, "The circuit must use the selected challenge's qubit count.")

    circuit = build_circuit(request)
    try:
        result = circuit.run(
            shots=request.shots,
            amplitude_states=2**request.num_qubits,
            per_qubit=True,
        ).to_dict()
    except (MemoryError, ValueError, IndexError, RuntimeError) as error:
        raise HTTPException(422, str(error)) from error

    challenge_passed = evaluate_challenge(challenge, result) if challenge else None
    run = SimulationRun(
        challenge_id=challenge.id if challenge else None,
        challenge_passed=challenge_passed,
        num_qubits=request.num_qubits,
        gates=[operation.model_dump(exclude_none=True) for operation in request.gates],
        result=result,
    )
    try:
        db.add(run)
        db.commit()
        db.refresh(run)
    except SQLAlchemyError as error:
        db.rollback()
        log.exception("Could not save simulation run.")
        raise HTTPException(503, "Simulation completed, but its history could not be saved.") from error

    return {
        "run_id": run.id,
        "challenge_id": run.challenge_id,
        "challenge_passed": run.challenge_passed,
        "result": result,
    }


@app.get("/api/history")
def list_history(
    limit: int = Query(default=20, ge=1, le=50),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    runs = db.scalars(
        select(SimulationRun).order_by(desc(SimulationRun.created_at), desc(SimulationRun.id)).limit(limit)
    )
    return [
        {
            "id": run.id,
            "challenge_id": run.challenge_id,
            "challenge_passed": run.challenge_passed,
            "num_qubits": run.num_qubits,
            "gate_count": len(run.gates),
            "gates": run.gates,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        for run in runs
    ]
