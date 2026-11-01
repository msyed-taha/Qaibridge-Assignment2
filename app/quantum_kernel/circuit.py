"""
Module 1 – Custom Simulation Kernel
circuit.py  –  High-level quantum circuit builder and execution engine

A QuantumCircuit is an ordered list of Operation objects.
Calling `.run()` creates a fresh QuantumStateVector and executes all
operations in sequence, returning a SimulationResult.

Besides ordinary gates a circuit may contain a few "black-box" operations
used by the SFOD algorithms (Module 2) and the Logic Transformer (Module 5):

    BARRIER  – no-op marker; `execute()` calls back here (per-iteration tracking)
    ORACLE   – phase oracle (−1)^{f(x)} over a set of marked basis states
    DIAGONAL – full diagonal unitary (e.g. a fused QAOA cost layer)
    CPERM    – controlled permutation of a register (Shor's modular multiplication)
"""

from __future__ import annotations

import cmath
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .gates import gate_arity, gate_param_count, MULTI_QUBIT_GATES
from .state_vector import QuantumStateVector
from .memory_manager import check_memory, MAX_QUBITS

SPECIAL_OPS = ("BARRIER", "ORACLE", "DIAGONAL", "CPERM")

# Result payload limits — a 20-qubit uniform superposition has a million
# non-zero amplitudes; shipping them all to the browser helps nobody.
MAX_REPORTED_STATES = 1024
MAX_REPORTED_PHASES = 256


def _sig(x: float, digits: int = 8) -> float:
    """Round to significant digits, so a 28-qubit chance of 3.7e-9 isn't reported as 0."""
    return float(f"{float(x):.{digits}g}")


class SimulationCancelled(Exception):
    """Raised between gates when the caller asked the run to stop (e.g. the user left the page)."""


# ──────────────────────────────────────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Operation:
    """Single quantum operation in the circuit."""
    gate:   str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    label:  str = ""
    meta:   Optional[Dict[str, Any]] = None   # payload of black-box operations

    def to_dict(self) -> Dict:
        d: Dict[str, Any] = {"gate": self.gate, "qubits": self.qubits}
        if self.params:
            d["params"] = [round(float(p), 10) for p in self.params]
        if self.label:
            d["label"] = self.label
        if self.gate == "ORACLE" and self.meta:
            d["marked_count"] = len(self.meta.get("marked", []))
        if self.gate == "CPERM" and self.meta:
            d["controls"] = self.meta.get("controls", [])
            d["targets"] = self.meta.get("targets", [])
        return d


@dataclass
class SimulationResult:
    """Full result returned after circuit execution."""
    n_qubits:       int
    gate_count:     int
    elapsed_ms:     float
    amplitudes:     List[Dict]       # top-N amplitude snapshots
    probabilities:  Dict[str, float] # most probable basis states (≤ MAX_REPORTED_STATES)
    counts:         Dict[str, int]   # sample counts (1 024 shots default)
    is_entangled:   bool
    summary:        Dict
    phases:         List[Dict]       # phase info for Bloch-sphere-like vis
    operations:     List[Dict]       # circuit operations (for circuit diagram)
    memory_report:  Dict
    depth:          int = 0
    bloch:          List[Dict] = field(default_factory=list)
    truncated:      bool = False     # True when more states exist than were reported
    qubit_ones:     List[float] = field(default_factory=list)   # chance each qubit reads 1 (when asked for)

    def to_dict(self) -> Dict:
        return {
            "n_qubits":      self.n_qubits,
            "gate_count":    self.gate_count,
            "depth":         self.depth,
            "elapsed_ms":    round(self.elapsed_ms, 3),
            "amplitudes":    self.amplitudes,
            "probabilities": self.probabilities,
            "counts":        self.counts,
            "is_entangled":  self.is_entangled,
            "summary":       self.summary,
            "phases":        self.phases,
            "operations":    self.operations[:2000],
            "memory_report": self.memory_report,
            "bloch":         self.bloch,
            "truncated":     self.truncated,
            "qubit_ones":    self.qubit_ones,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Circuit Builder
# ──────────────────────────────────────────────────────────────────────────────

class QuantumCircuit:
    """
    Fluent builder for quantum circuits.

    Usage:
        circuit = QuantumCircuit(n_qubits=3)
        circuit.h(0).cnot(0, 1).cnot(0, 2)
        result = circuit.run(shots=1024)
    """

    MAX_GATES = 500  # Safety limit for user-submitted circuits

    def __init__(self, n_qubits: int, name: str = "", max_gates: Optional[int] = None):
        # This is a static sanity bound only -- the real, RAM-aware safety
        # check runs in run() via check_memory() (dynamic: depends on actual
        # available system memory at request time, not just qubit count).
        if n_qubits < 1 or n_qubits > MAX_QUBITS:
            raise ValueError(f"n_qubits must be between 1 and {MAX_QUBITS}, got {n_qubits}.")
        self.n_qubits = n_qubits
        self.name = name or f"{n_qubits}-qubit circuit"
        self.max_gates = max_gates if max_gates is not None else self.MAX_GATES
        self._ops: List[Operation] = []

    # ── Single-Qubit Gates ────────────────────────────────────────────────

    def h(self, qubit: int) -> "QuantumCircuit":
        return self._add("H", [qubit])

    def x(self, qubit: int) -> "QuantumCircuit":
        return self._add("X", [qubit])

    def y(self, qubit: int) -> "QuantumCircuit":
        return self._add("Y", [qubit])

    def z(self, qubit: int) -> "QuantumCircuit":
        return self._add("Z", [qubit])

    def s(self, qubit: int) -> "QuantumCircuit":
        return self._add("S", [qubit])

    def t(self, qubit: int) -> "QuantumCircuit":
        return self._add("T", [qubit])

    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        return self._add("RX", [qubit], [theta])

    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        return self._add("RY", [qubit], [theta])

    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        return self._add("RZ", [qubit], [theta])

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> "QuantumCircuit":
        return self._add("U", [qubit], [theta, phi, lam])

    def p(self, phi: float, qubit: int) -> "QuantumCircuit":
        return self._add("P", [qubit], [phi])

    # ── Two-Qubit Gates ───────────────────────────────────────────────────

    def cnot(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._add("CNOT", [ctrl, tgt])

    def cx(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._add("CNOT", [ctrl, tgt])

    def cz(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._add("CZ", [ctrl, tgt])

    def swap(self, q1: int, q2: int) -> "QuantumCircuit":
        return self._add("SWAP", [q1, q2])

    def crx(self, theta: float, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._add("CRX", [ctrl, tgt], [theta])

    def cry(self, theta: float, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._add("CRY", [ctrl, tgt], [theta])

    def crz(self, theta: float, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._add("CRZ", [ctrl, tgt], [theta])

    def cp(self, phi: float, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self._add("CP", [ctrl, tgt], [phi])

    def rzz(self, theta: float, q1: int, q2: int) -> "QuantumCircuit":
        return self._add("RZZ", [q1, q2], [theta])

    # ── Three-Qubit & Multi-Controlled Gates ───────────────────────────────

    def ccx(self, c1: int, c2: int, tgt: int) -> "QuantumCircuit":
        return self._add("CCX", [c1, c2, tgt])

    def mcx(self, controls: Sequence[int], tgt: int) -> "QuantumCircuit":
        return self._add("MCX", list(controls) + [tgt])

    def mcz(self, qubits: Sequence[int]) -> "QuantumCircuit":
        return self._add("MCZ", list(qubits))

    def mcp(self, phi: float, qubits: Sequence[int]) -> "QuantumCircuit":
        return self._add("MCP", list(qubits), [phi])

    # ── Black-box operations ──────────────────────────────────────────────

    def barrier(self, label: str = "") -> "QuantumCircuit":
        """No-op marker. `execute()` invokes its callback at every barrier."""
        self._ops.append(Operation("BARRIER", list(range(self.n_qubits)), label=label))
        return self

    def oracle(self, marked: Sequence[int], label: str = "Oracle") -> "QuantumCircuit":
        """Phase oracle flipping the sign of the marked basis states."""
        marked = sorted({int(m) for m in marked})
        if any(m < 0 or m >= 2 ** self.n_qubits for m in marked):
            raise IndexError("Marked state outside the register.")
        return self._add_special("ORACLE", list(range(self.n_qubits)), label, {"marked": marked})

    def diagonal(self, phases: np.ndarray, label: str = "Diagonal") -> "QuantumCircuit":
        """Full diagonal unitary given its 2^n (unit-modulus) entries."""
        phases = np.asarray(phases, dtype=complex)
        if phases.shape != (2 ** self.n_qubits,):
            raise ValueError("Diagonal needs 2^n entries.")
        return self._add_special("DIAGONAL", list(range(self.n_qubits)), label, {"diagonal": phases})

    def cperm(self, controls: Sequence[int], targets: Sequence[int],
              permutation: Sequence[int], label: str = "U") -> "QuantumCircuit":
        """Controlled permutation of the `targets` register (Shor's modular multiplier)."""
        controls, targets = list(controls), list(targets)
        return self._add_special(
            "CPERM", controls + targets, label,
            {"controls": controls, "targets": targets, "permutation": list(map(int, permutation))},
        )

    # ── Convenience: common multi-gate patterns ───────────────────────────

    def bell_state(self, q0: int = 0, q1: int = 1) -> "QuantumCircuit":
        """Prepare the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        return self.h(q0).cnot(q0, q1)

    def ghz_state(self) -> "QuantumCircuit":
        """Prepare the GHZ state for all qubits."""
        self.h(0)
        for i in range(self.n_qubits - 1):
            self.cnot(i, i + 1)
        return self

    def qft(self, qubits: Optional[List[int]] = None, swaps: bool = True) -> "QuantumCircuit":
        """
        Append the Quantum Fourier Transform over the specified qubits
        (first listed = most-significant bit):

            QFT|x⟩ = 1/√N · Σ_y e^{2πi·xy/N} |y⟩

        Built from H and controlled-phase CP(π/2^k) gates, then SWAPs to
        restore bit order (Nielsen & Chuang, Fig. 5.1).
        """
        qs = qubits if qubits is not None else list(range(self.n_qubits))
        n = len(qs)
        for i in range(n):
            self.h(qs[i])
            for j in range(i + 1, n):
                self.cp(math.pi / (2 ** (j - i)), qs[j], qs[i])
        if swaps:
            for i in range(n // 2):
                self.swap(qs[i], qs[n - 1 - i])
        return self

    def iqft(self, qubits: Optional[List[int]] = None, swaps: bool = True) -> "QuantumCircuit":
        """Inverse QFT — the QFT's gates in reverse order with negated angles."""
        qs = qubits if qubits is not None else list(range(self.n_qubits))
        n = len(qs)
        if swaps:
            for i in range(n // 2):
                self.swap(qs[i], qs[n - 1 - i])
        for i in reversed(range(n)):
            for j in reversed(range(i + 1, n)):
                self.cp(-math.pi / (2 ** (j - i)), qs[j], qs[i])
            self.h(qs[i])
        return self

    # ── Execution ─────────────────────────────────────────────────────────

    def execute(self, sv: QuantumStateVector,
                on_barrier: Optional[Callable[[str, QuantumStateVector], None]] = None,
                on_progress: Optional[Callable[[int, int], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None) -> QuantumStateVector:
        """
        Apply every operation to an existing state vector.

        on_barrier(label, sv) fires at each BARRIER (e.g. to record the
        success probability after every Grover iteration); on_progress(done,
        total) fires roughly every 2 % of the operations (WebSocket progress);
        should_stop() is asked before every operation and, once it says yes,
        the run ends with SimulationCancelled.
        """
        if sv.n_qubits != self.n_qubits:
            raise ValueError("State vector and circuit have different qubit counts.")
        total = len(self._ops)
        step = max(1, total // 50)
        for i, op in enumerate(self._ops):
            if should_stop is not None and should_stop():
                raise SimulationCancelled(f"Stopped after {i} of {total} gates.")
            if op.gate == "BARRIER":
                if on_barrier is not None:
                    on_barrier(op.label, sv)
            elif op.gate == "ORACLE":
                sv.apply_phase_oracle(op.meta["marked"])
            elif op.gate == "DIAGONAL":
                sv.apply_diagonal(op.meta["diagonal"])
            elif op.gate == "CPERM":
                sv.apply_controlled_permutation(op.meta["controls"], op.meta["targets"],
                                                op.meta["permutation"])
            else:
                sv.apply_gate(op.gate, op.qubits, op.params)
            if on_progress is not None and (i + 1) % step == 0:
                on_progress(i + 1, total)
        if on_progress is not None:
            on_progress(total, total)
        return sv

    def run(self, shots: int = 1024,
            on_progress: Optional[Callable[[int, int], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None,
            amplitude_states: int = 64, per_qubit: bool = False) -> SimulationResult:
        """
        Execute the circuit on a fresh QuantumStateVector.

        Args:
            shots: Number of measurement samples for histogram counts.
            amplitude_states: how many of the most likely states to report amplitudes for.
            per_qubit: also report the chance each qubit reads 1 (one extra pass over the state).

        Returns:
            SimulationResult with amplitudes, probabilities, counts, and metadata.
        """
        mem_report = check_memory(self.n_qubits)
        if not mem_report.is_safe:
            raise MemoryError(mem_report.warning)

        sv = QuantumStateVector(self.n_qubits, check_ram=False)
        t0 = time.perf_counter()
        self.execute(sv, on_progress=on_progress, should_stop=should_stop)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        n = self.n_qubits
        large = n > 20
        top = sv.top_states(min(MAX_REPORTED_STATES, sv.dim))
        top = [(i, p) for i, p in top if p > 1e-9]

        probabilities = {f"|{i:0{n}b}>": _sig(p) for i, p in sorted(top)}

        phases = [
            {
                "state":     f"|{i:0{n}b}>",
                "phase_rad": round(cmath.phase(sv._state[i]), 6),
                "phase_deg": round(math.degrees(cmath.phase(sv._state[i])), 4),
            }
            for i, _ in sorted(top[:MAX_REPORTED_PHASES])
        ]

        amplitudes = []
        for i, p in sorted(top[:amplitude_states]):
            amp = sv._state[i]
            amplitudes.append({
                "state": f"|{i:0{n}b}>", "index": int(i),
                "real": _sig(amp.real), "imag": _sig(amp.imag),
                "probability": _sig(p), "magnitude": _sig(abs(amp)),
                "phase_deg": round(math.degrees(cmath.phase(amp)), 4),
            })

        entangled = sv.is_entangled(max_qubits_checked=20 if not large else 2)
        best_i, best_p = top[0] if top else (0, 0.0)
        summary = {
            "n_qubits":       n,
            "dim":            sv.dim,
            "gate_count":     sv._gate_count,
            "norm":           round(sv.norm_squared(), 10),
            "is_entangled":   entangled,
            "max_prob_state": f"|{best_i:0{n}b}>",
            "max_prob":       _sig(best_p),
        }

        return SimulationResult(
            n_qubits=n,
            gate_count=self.gate_count(),
            elapsed_ms=elapsed_ms,
            amplitudes=amplitudes,
            probabilities=probabilities,
            counts=sv.measure_all(shots=shots),
            is_entangled=entangled,
            summary=summary,
            phases=phases,
            operations=[op.to_dict() for op in self._ops],
            memory_report={
                "n_qubits":     mem_report.n_qubits,
                "required_gb":  round(mem_report.required_gb, 4),
                "available_gb": round(mem_report.available_gb, 2),
                "is_safe":      mem_report.is_safe,
                "warning":      mem_report.warning,
            },
            depth=self.depth(),
            bloch=sv.bloch_vectors() if n <= 16 else [],
            truncated=len(top) >= MAX_REPORTED_STATES,
            qubit_ones=[_sig(x) for x in sv.qubit_one_probabilities()] if per_qubit else [],
        )

    # ── Circuit statistics ────────────────────────────────────────────────

    def gate_count(self) -> int:
        """Number of operations excluding barriers."""
        return sum(1 for op in self._ops if op.gate != "BARRIER")

    def count_ops(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for op in self._ops:
            if op.gate != "BARRIER":
                counts[op.gate] = counts.get(op.gate, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def depth(self) -> int:
        """Circuit depth: the length of the longest chain of operations sharing a qubit."""
        level = [0] * self.n_qubits
        for op in self._ops:
            if op.gate == "BARRIER":
                top = max(level) if level else 0
                level = [top] * self.n_qubits
                continue
            layer = max(level[q] for q in op.qubits) + 1
            for q in op.qubits:
                level[q] = layer
        return max(level) if level else 0

    def stats(self) -> Dict:
        two_plus = sum(1 for op in self._ops if op.gate != "BARRIER" and len(op.qubits) >= 2)
        return {
            "qubits": self.n_qubits,
            "gates": self.gate_count(),
            "depth": self.depth(),
            "multi_qubit_gates": two_plus,
            "ops": self.count_ops(),
        }

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "name":      self.name,
            "n_qubits":  self.n_qubits,
            "gate_count": self.gate_count(),
            "depth":     self.depth(),
            "operations": [op.to_dict() for op in self._ops],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QuantumCircuit":
        """Reconstruct a circuit from a serialised dict (e.g. from frontend)."""
        circuit = cls(n_qubits=data["n_qubits"], name=data.get("name", ""))
        for op in data.get("operations", []):
            circuit._add(op["gate"], op["qubits"], op.get("params", []))
        return circuit

    def to_qiskit(self, measure: bool = True, shots: int = 1024) -> str:
        """
        Export as runnable Qiskit code (for running on IBM hardware / Aer).
        Black-box operations are expanded into standard gates.
        """
        from .qiskit_export import circuit_to_qiskit
        return circuit_to_qiskit(self, measure=measure, shots=shots)

    # ── Internals ─────────────────────────────────────────────────────────

    def _check_capacity(self) -> None:
        if len(self._ops) >= self.max_gates:
            raise RuntimeError(
                f"Circuit exceeds maximum gate count of {self.max_gates}."
            )

    def _check_qubits(self, qubits: Sequence[int]) -> None:
        for q in qubits:
            if not isinstance(q, (int, np.integer)) or not (0 <= q < self.n_qubits):
                raise IndexError(
                    f"Qubit {q} out of range for {self.n_qubits}-qubit circuit."
                )
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Qubits of one gate must be distinct, got {list(qubits)}.")

    def _add(self, gate: str, qubits: List[int], params: List[float] = None) -> "QuantumCircuit":
        self._check_capacity()
        name = gate.upper()
        if name == "CX":
            name = "CNOT"
        qubits = [int(q) for q in qubits]
        params = [float(p) for p in (params or [])]

        arity = gate_arity(name)          # raises ValueError for unknown gates
        if name in MULTI_QUBIT_GATES:
            if len(qubits) < 1:
                raise ValueError(f"Gate '{name}' needs at least one qubit.")
        elif len(qubits) != arity:
            raise ValueError(f"Gate '{name}' expects exactly {arity} qubit(s), got {len(qubits)}.")
        expected = gate_param_count(name)
        if expected and len(params) != expected:
            raise ValueError(f"Gate '{name}' takes {expected} angle parameter(s), got {len(params)}.")
        if not expected:
            params = []
        self._check_qubits(qubits)

        self._ops.append(Operation(gate=name, qubits=qubits, params=params))
        return self

    def _add_special(self, gate: str, qubits: List[int], label: str, meta: Dict) -> "QuantumCircuit":
        self._check_capacity()
        self._check_qubits(qubits)
        self._ops.append(Operation(gate=gate, qubits=qubits, label=label, meta=meta))
        return self

    @property
    def operations(self) -> List[Operation]:
        return list(self._ops)

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        return f"<QuantumCircuit name='{self.name}' qubits={self.n_qubits} gates={len(self._ops)}>"


# ── Pre-built Circuit Factory ─────────────────────────────────────────────────

class CircuitLibrary:
    """
    Ready-made circuits used by the SFOD suite and educational simulator.
    """

    @staticmethod
    def bell_state(n_qubits: int = 2) -> QuantumCircuit:
        assert n_qubits >= 2
        return QuantumCircuit(n_qubits, "Bell State").bell_state(0, 1)

    @staticmethod
    def ghz_state(n_qubits: int = 3) -> QuantumCircuit:
        assert n_qubits >= 2
        return QuantumCircuit(n_qubits, f"GHZ-{n_qubits}").ghz_state()

    @staticmethod
    def grover_oracle_2qubit() -> QuantumCircuit:
        """
        2-qubit Grover search — marks target state |11⟩.
        Full circuit: H⊗2 → Oracle (CZ) → Diffuser.
        """
        circ = QuantumCircuit(2, "Grover-2q")
        # Superposition
        circ.h(0).h(1)
        # Oracle: flip phase of |11⟩
        circ.cz(0, 1)
        # Diffuser: 2|s⟩⟨s| - I
        circ.h(0).h(1)
        circ.x(0).x(1)
        circ.cz(0, 1)
        circ.x(0).x(1)
        circ.h(0).h(1)
        return circ

    @staticmethod
    def grover(n_qubits: int, target: int) -> QuantumCircuit:
        """
        n-qubit Grover search for one marked item, fully gate-level:
        oracle = X-conjugated multi-controlled Z, diffuser = H X MCZ X H.
        """
        n_iter = max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** n_qubits))))
        circ = QuantumCircuit(n_qubits, f"Grover-{n_qubits}q", max_gates=100_000)
        qs = list(range(n_qubits))
        for q in qs:
            circ.h(q)
        zeros = [q for q in qs if not (target >> (n_qubits - 1 - q)) & 1]
        for _ in range(n_iter):
            for q in zeros:
                circ.x(q)
            circ.mcz(qs)
            for q in zeros:
                circ.x(q)
            for q in qs:
                circ.h(q)
            for q in qs:
                circ.x(q)
            circ.mcz(qs)
            for q in qs:
                circ.x(q)
            for q in qs:
                circ.h(q)
        return circ

    @staticmethod
    def qft_circuit(n_qubits: int = 4) -> QuantumCircuit:
        circ = QuantumCircuit(n_qubits, f"QFT-{n_qubits}q", max_gates=100_000)
        circ.qft()
        return circ

    @staticmethod
    def parametric_ansatz(n_qubits: int, layers: int = 2) -> QuantumCircuit:
        """
        Hardware-efficient ansatz for VQE / QAOA.
        Each layer: RY on every qubit → CNOT chain.
        """
        circ = QuantumCircuit(n_qubits, f"Ansatz-{n_qubits}q-{layers}L", max_gates=100_000)
        for _ in range(layers):
            for q in range(n_qubits):
                circ.ry(math.pi / 4, q)
            for q in range(n_qubits - 1):
                circ.cnot(q, q + 1)
        return circ
