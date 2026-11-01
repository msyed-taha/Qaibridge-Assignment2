"""
Module 1 – Custom Simulation Kernel
gates.py  –  All supported quantum gate matrices

Gates are represented as NumPy complex128 arrays.
Single-qubit gates : H, X, Y, Z, S, T, I, SX, RX(θ), RY(θ), RZ(θ), P(φ), U(θ,φ,λ)
Two-qubit gates    : CNOT, CY, CZ, CH, SWAP, ISWAP, CP(φ), CRX(θ), CRY(θ), CRZ(θ),
                     RXX(θ), RYY(θ), RZZ(θ)
Three-qubit gates  : CCX (Toffoli), CCZ, CSWAP (Fredkin)

Multi-controlled gates (MCX / MCZ / MCP) and black-box operators (phase
oracles, controlled permutations, diagonal cost layers) have no fixed-size
matrix — they are applied directly on the state vector, see state_vector.py.
"""

import inspect

import numpy as np
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────────────────────
# Single-Qubit Constant Gates
# ──────────────────────────────────────────────────────────────────────────────

I_GATE = np.eye(2, dtype=complex)

H_GATE = np.array(
    [[1,  1],
     [1, -1]], dtype=complex
) / np.sqrt(2)

X_GATE = np.array(
    [[0, 1],
     [1, 0]], dtype=complex
)

Y_GATE = np.array(
    [[0, -1j],
     [1j,  0]], dtype=complex
)

Z_GATE = np.array(
    [[1,  0],
     [0, -1]], dtype=complex
)

S_GATE = np.array(
    [[1, 0],
     [0, 1j]], dtype=complex
)

S_DAG_GATE = np.array(
    [[1,  0],
     [0, -1j]], dtype=complex
)

T_GATE = np.array(
    [[1, 0],
     [0, np.exp(1j * np.pi / 4)]], dtype=complex
)

T_DAG_GATE = np.array(
    [[1, 0],
     [0, np.exp(-1j * np.pi / 4)]], dtype=complex
)

SX_GATE = np.array(
    [[1 + 1j, 1 - 1j],
     [1 - 1j, 1 + 1j]], dtype=complex
) / 2   # √X — the native gate of IBM hardware

SXDG_GATE = SX_GATE.conj().T   # √X† — undoes an SX gate

# ──────────────────────────────────────────────────────────────────────────────
# Single-Qubit Parametric Gates
# ──────────────────────────────────────────────────────────────────────────────

def RX(theta: float) -> np.ndarray:
    """Rotation around X-axis by angle theta (radians)."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c,       -1j * s],
         [-1j * s,  c     ]], dtype=complex
    )


def RY(theta: float) -> np.ndarray:
    """Rotation around Y-axis by angle theta (radians)."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -s],
         [s,  c]], dtype=complex
    )


def RZ(theta: float) -> np.ndarray:
    """Rotation around Z-axis by angle theta (radians)."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0                        ],
         [0,                        np.exp(1j * theta / 2)  ]], dtype=complex
    )


def U_GATE(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    Generic single-qubit unitary (IBM U-gate).
    U(θ, φ, λ) = [[cos(θ/2),        -e^{iλ}·sin(θ/2)],
                  [e^{iφ}·sin(θ/2),  e^{i(φ+λ)}·cos(θ/2)]]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c,                              -np.exp(1j * lam) * s     ],
         [np.exp(1j * phi) * s,            np.exp(1j * (phi + lam)) * c]], dtype=complex
    )


def P_GATE(phi: float) -> np.ndarray:
    """Phase gate: adds phase e^{iφ} to |1> state."""
    return np.array(
        [[1, 0                ],
         [0, np.exp(1j * phi)]], dtype=complex
    )


# ──────────────────────────────────────────────────────────────────────────────
# Two-Qubit Constant Gates
# ──────────────────────────────────────────────────────────────────────────────

CNOT_GATE = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]], dtype=complex
)

CZ_GATE = np.array(
    [[1, 0, 0,  0],
     [0, 1, 0,  0],
     [0, 0, 1,  0],
     [0, 0, 0, -1]], dtype=complex
)

SWAP_GATE = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]], dtype=complex
)

ISWAP_GATE = np.array(
    [[1,  0,  0, 0],
     [0,  0, 1j, 0],
     [0, 1j,  0, 0],
     [0,  0,  0, 1]], dtype=complex
)

CY_GATE = np.array(
    [[1, 0, 0,   0],
     [0, 1, 0,   0],
     [0, 0, 0, -1j],
     [0, 0, 1j,  0]], dtype=complex
)

CH_GATE = np.block([
    [np.eye(2, dtype=complex), np.zeros((2, 2), dtype=complex)],
    [np.zeros((2, 2), dtype=complex), H_GATE],
])

# ──────────────────────────────────────────────────────────────────────────────
# Two-Qubit Parametric Controlled Gates
# ──────────────────────────────────────────────────────────────────────────────

def CRX(theta: float) -> np.ndarray:
    """Controlled-RX gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0,       0,        0      ],
         [0, 1,       0,        0      ],
         [0, 0,       c,  -1j * s      ],
         [0, 0, -1j * s,        c      ]], dtype=complex
    )


def CRY(theta: float) -> np.ndarray:
    """Controlled-RY gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0,  0,  0],
         [0, 1,  0,  0],
         [0, 0,  c, -s],
         [0, 0,  s,  c]], dtype=complex
    )


def CRZ(theta: float) -> np.ndarray:
    """Controlled-RZ gate."""
    return np.array(
        [[1, 0, 0,                        0                        ],
         [0, 1, 0,                        0                        ],
         [0, 0, np.exp(-1j * theta / 2),  0                        ],
         [0, 0, 0,                        np.exp(1j * theta / 2)   ]], dtype=complex
    )


def CP(phi: float) -> np.ndarray:
    """
    Controlled-phase gate: adds e^{iφ} to |11⟩ only.

    This (not CRZ) is the rotation the Quantum Fourier Transform needs.
    CRZ(φ) = CP(φ) · (P(-φ/2) on the control), so building a QFT from CRZ
    leaves extra relative phases on the control qubits.
    """
    return np.diag([1, 1, 1, np.exp(1j * phi)]).astype(complex)


def RZZ(theta: float) -> np.ndarray:
    """exp(-iθ/2 · Z⊗Z) — the two-body term of an Ising cost Hamiltonian (QAOA)."""
    a = np.exp(-1j * theta / 2)
    b = np.exp(1j * theta / 2)
    return np.diag([a, b, b, a]).astype(complex)


def RXX(theta: float) -> np.ndarray:
    """exp(-iθ/2 · X⊗X)."""
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    return np.array(
        [[c, 0, 0, s],
         [0, c, s, 0],
         [0, s, c, 0],
         [s, 0, 0, c]], dtype=complex
    )


def RYY(theta: float) -> np.ndarray:
    """exp(-iθ/2 · Y⊗Y)."""
    c, s = np.cos(theta / 2), 1j * np.sin(theta / 2)
    return np.array(
        [[c,  0,  0, s],
         [0,  c, -s, 0],
         [0, -s,  c, 0],
         [s,  0,  0, c]], dtype=complex
    )


def CU(theta: float, phi: float, lam: float, gamma: float) -> np.ndarray:
    """
    Controlled-U with a phase on the control (Qiskit's CUGate, OpenQASM `cu`):
    |0⟩⟨0| ⊗ I + e^{iγ}|1⟩⟨1| ⊗ U(θ, φ, λ).  With γ = 0 this is `cu3`.
    """
    m = np.eye(4, dtype=complex)
    m[2:, 2:] = np.exp(1j * gamma) * U_GATE(theta, phi, lam)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Three-Qubit Constant Gates
# ──────────────────────────────────────────────────────────────────────────────

CCX_GATE = np.eye(8, dtype=complex)          # Toffoli: flip target iff both controls are 1
CCX_GATE[[6, 7]] = CCX_GATE[[7, 6]]

CCZ_GATE = np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(complex)

CSWAP_GATE = np.eye(8, dtype=complex)        # Fredkin: swap targets iff control is 1
CSWAP_GATE[[5, 6]] = CSWAP_GATE[[6, 5]]


# ──────────────────────────────────────────────────────────────────────────────
# Gate Registry — maps string names → matrix (or callable)
# ──────────────────────────────────────────────────────────────────────────────

SINGLE_QUBIT_GATES: Dict[str, Any] = {
    "I":     I_GATE,
    "H":     H_GATE,
    "X":     X_GATE,
    "Y":     Y_GATE,
    "Z":     Z_GATE,
    "S":     S_GATE,
    "SDAG":  S_DAG_GATE,
    "SDG":   S_DAG_GATE,  # alias (Qiskit spelling)
    "T":     T_GATE,
    "TDAG":  T_DAG_GATE,
    "TDG":   T_DAG_GATE,  # alias (Qiskit spelling)
    "SX":    SX_GATE,
    "SXDG":  SXDG_GATE,
    # Parametric (callables)
    "RX":    RX,
    "RY":    RY,
    "RZ":    RZ,
    "U":     U_GATE,
    "P":     P_GATE,
}

TWO_QUBIT_GATES: Dict[str, Any] = {
    "CNOT":  CNOT_GATE,
    "CX":    CNOT_GATE,  # alias
    "CY":    CY_GATE,
    "CZ":    CZ_GATE,
    "CH":    CH_GATE,
    "SWAP":  SWAP_GATE,
    "ISWAP": ISWAP_GATE,
    # Parametric
    "CRX":   CRX,
    "CRY":   CRY,
    "CRZ":   CRZ,
    "CP":    CP,
    "CPHASE": CP,        # alias
    "RXX":   RXX,
    "RYY":   RYY,
    "RZZ":   RZZ,
    "CU":    CU,
}

THREE_QUBIT_GATES: Dict[str, Any] = {
    "CCX":     CCX_GATE,
    "TOFFOLI": CCX_GATE,  # alias
    "CCZ":     CCZ_GATE,
    "CSWAP":   CSWAP_GATE,
    "FREDKIN": CSWAP_GATE,  # alias
}

# Gates that act on any number of qubits; applied by dedicated state-vector
# routines rather than a matrix (see QuantumStateVector.apply_gate).
#   MCX(controls…, target)   MCZ(qubits…)   MCP(qubits…; φ)
MULTI_QUBIT_GATES = ("MCX", "MCZ", "MCP")


def gate_arity(name: str) -> int:
    """Number of qubits a fixed-size gate acts on (0 = variable, e.g. MCZ)."""
    n = name.upper()
    if n in SINGLE_QUBIT_GATES:
        return 1
    if n in TWO_QUBIT_GATES:
        return 2
    if n in THREE_QUBIT_GATES:
        return 3
    if n in MULTI_QUBIT_GATES:
        return 0
    raise ValueError(f"Unknown gate: '{name}'.")


def gate_param_count(name: str) -> int:
    """How many angle parameters a gate takes (0 for constant gates)."""
    n = name.upper()
    if n == "MCP":
        return 1
    for table in (SINGLE_QUBIT_GATES, TWO_QUBIT_GATES, THREE_QUBIT_GATES):
        if n in table:
            g = table[n]
            return len(inspect.signature(g).parameters) if callable(g) else 0
    if n in MULTI_QUBIT_GATES:
        return 0
    raise ValueError(f"Unknown gate: '{name}'.")


def get_gate_matrix(name: str, params: list = None) -> np.ndarray:
    """
    Resolve a gate name (and optional parameters) to its matrix.

    Examples:
        get_gate_matrix("H")         → H_GATE
        get_gate_matrix("RX", [1.57]) → RX(1.57)
        get_gate_matrix("CNOT")      → CNOT_GATE
    """
    params = list(params or [])
    name_upper = name.upper()

    for table in (SINGLE_QUBIT_GATES, TWO_QUBIT_GATES, THREE_QUBIT_GATES):
        if name_upper in table:
            gate = table[name_upper]
            if not callable(gate):
                return gate
            expected = gate_param_count(name_upper)
            if len(params) != expected:
                raise ValueError(
                    f"Gate '{name_upper}' takes {expected} angle parameter(s), got {len(params)}."
                )
            return gate(*[float(p) for p in params])

    raise ValueError(f"Unknown gate: '{name}'. "
                     f"Supported: {list(SINGLE_QUBIT_GATES) + list(TWO_QUBIT_GATES) + list(THREE_QUBIT_GATES) + list(MULTI_QUBIT_GATES)}")
