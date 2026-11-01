"""
Module 1 – Custom Simulation Kernel
memory_manager.py  –  RAM estimation and safety checks

The state vector for n qubits = 2^n complex128 numbers.
Each complex128 = 16 bytes  →  total = 2^n × 16 bytes.
"""

import os

import psutil
from dataclasses import dataclass
from typing import Optional, Tuple

# Set per server: a machine with more memory can raise both (QAI_MAX_QUBITS, QAI_MAX_RAM_GB).
MAX_QUBITS = int(os.getenv("QAI_MAX_QUBITS", "28"))          # 2^28 × 16 B = 4.3 GB
MAX_RAM_CAP_GB = float(os.getenv("QAI_MAX_RAM_GB", "5"))     # never allocate more than this, whatever RAM is free
SAFETY_MARGIN = 0.85      # Use at most 85 % of available RAM
BYTES_PER_AMPLITUDE = 16  # complex128


@dataclass
class MemoryReport:
    n_qubits: int
    state_vector_size: int          # 2^n
    required_bytes: int
    required_gb: float
    available_bytes: int
    available_gb: float
    total_ram_gb: float
    is_safe: bool
    warning: Optional[str] = None


def estimate_required_bytes(n_qubits: int, bytes_per_amplitude: int = BYTES_PER_AMPLITUDE) -> int:
    """Return bytes needed for the state-vector (16 per complex amplitude; 8 for a real-valued one)."""
    return (2 ** n_qubits) * bytes_per_amplitude


def check_memory(n_qubits: int, bytes_per_amplitude: int = BYTES_PER_AMPLITUDE) -> MemoryReport:
    """
    Full memory safety check for a given qubit count.

    Two limits are enforced:
      1. Absolute ceiling: required GB must not exceed MAX_RAM_CAP_GB (5 GB).
      2. Safety margin: required must be ≤ SAFETY_MARGIN × available RAM.

    Returns a MemoryReport with all relevant statistics and a boolean
    `is_safe` flag that the simulation kernel uses before allocating.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if n_qubits > MAX_QUBITS:
        raise ValueError(
            f"n_qubits={n_qubits} exceeds this server's limit of {MAX_QUBITS} "
            f"(~{MAX_RAM_CAP_GB:.0f} GB). A server with more memory can raise QAI_MAX_QUBITS and QAI_MAX_RAM_GB."
        )

    vm = psutil.virtual_memory()
    required = estimate_required_bytes(n_qubits, bytes_per_amplitude)
    available = vm.available

    # Hard cap: reject if simulation would exceed 5 GB even on a large-RAM machine
    exceeds_cap = (required / 1e9) > MAX_RAM_CAP_GB
    # Safety margin: reject if not enough free RAM
    exceeds_available = required > available * SAFETY_MARGIN

    is_safe = not exceeds_cap and not exceeds_available

    warning = None
    if exceeds_cap:
        warning = (
            f"Simulation of {n_qubits} qubits requires "
            f"{required / 1e9:.2f} GB which exceeds the configured cap of "
            f"{MAX_RAM_CAP_GB:.0f} GB. Reduce qubit count."
        )
    elif exceeds_available:
        warning = (
            f"Insufficient RAM: simulation of {n_qubits} qubits requires "
            f"{required / 1e9:.2f} GB but only "
            f"{available * SAFETY_MARGIN / 1e9:.2f} GB is safely available "
            f"(total available: {available / 1e9:.2f} GB)."
        )
    elif required > available * 0.60:
        warning = (
            f"High RAM usage: {required / 1e9:.2f} GB required out of "
            f"{available / 1e9:.2f} GB available. Simulation will proceed but "
            "may be slow due to memory pressure."
        )

    return MemoryReport(
        n_qubits=n_qubits,
        state_vector_size=2 ** n_qubits,
        required_bytes=required,
        required_gb=required / 1e9,
        available_bytes=available,
        available_gb=available / 1e9,
        total_ram_gb=vm.total / 1e9,
        is_safe=is_safe,
        warning=warning,
    )


def ram_table() -> list[dict]:
    """
    Return a table of qubit counts → RAM requirements.
    Used by the frontend slider to show dynamic RAM estimates.
    """
    rows = []
    for n in range(1, MAX_QUBITS + 1):
        required = estimate_required_bytes(n)
        rows.append({
            "n_qubits": n,
            "state_vector_size": 2 ** n,
            "required_bytes": required,
            "required_mb": round(required / 1e6, 3),
            "required_gb": round(required / 1e9, 4),
        })
    return rows
