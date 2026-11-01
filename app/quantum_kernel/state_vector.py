"""
Module 1 – Custom Simulation Kernel
state_vector.py  –  Core quantum state-vector engine

Design principles
─────────────────
• Uses NumPy complex128 arrays for state storage.
• Gates are applied IN PLACE on strided views of the state vector — the
  flat 2^n array is reshaped (never copied) so every target qubit becomes its
  own axis of length 2, and only the touched sub-blocks are rewritten.
  Working memory is bounded to small chunks (≈16 MB), so the peak RAM of a
  simulation is ≈ one state vector, which is what lets the kernel reach the
  26–28 qubit range on a 16 GB laptop ("Memory Wall", FE-3).
• Fast paths: diagonal gates (Z, S, T, RZ, P, CZ, CP, RZZ …) are pure
  in-place phase multiplications; controlled gates only touch the
  control = 1 half; multi-controlled gates (MCX / MCZ) work on a single
  sub-view with all controls fixed to 1.
• Black-box operators used by the SFOD algorithms (Module 2): phase oracles,
  diagonal cost layers (QAOA) and controlled permutations (Shor's modular
  multiplication).
• Read-out (probabilities, sampling, top-k states, Bloch vectors) is also
  chunked, so large registers never materialise extra 2^n-sized arrays.
• All public methods validate inputs and raise informative exceptions.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .gates import get_gate_matrix, gate_arity, MULTI_QUBIT_GATES
from .memory_manager import check_memory

# Largest number of amplitudes processed at once by chunked routines
# (2^20 complex128 values = 16 MB).
_CHUNK = 1 << 20

# Controlled permutations build index arrays of the full state size, so they
# are limited to registers where that is cheap.
_MAX_PERMUTATION_QUBITS = 24


def _iter_chunks(view: np.ndarray, fixed_axes: Sequence[int]):
    """
    Yield tuples of basic slices that partition `view` into blocks whose
    non-fixed part holds at most _CHUNK elements. Blocks are cut along the
    largest free axis (one level of cutting is always enough for ≤ 28 qubits).
    """
    free = [a for a in range(view.ndim) if a not in fixed_axes]
    free_size = 1
    for a in free:
        free_size *= view.shape[a]
    if not free or free_size <= _CHUNK:
        yield (slice(None),) * view.ndim
        return
    cut = max(free, key=lambda a: view.shape[a])
    rest = free_size // view.shape[cut]
    step = max(1, _CHUNK // max(rest, 1))
    for start in range(0, view.shape[cut], step):
        sl = [slice(None)] * view.ndim
        sl[cut] = slice(start, min(view.shape[cut], start + step))
        yield tuple(sl)


class QuantumStateVector:
    """
    Proprietary state-vector simulator for n_qubits (1 – MAX_QUBITS).

    The quantum state is stored as a flat 1-D complex128 NumPy array of
    length 2^n.  Qubit ordering follows the big-endian convention:
    qubit 0 is the most-significant bit.

        |ψ⟩ = Σ_k  amplitude[k]  |k⟩
        where k is read as  q0 q1 q2 … q_{n-1}  in binary.
    """

    def __init__(self, n_qubits: int, check_ram: bool = True):
        if check_ram:
            report = check_memory(n_qubits)
            if not report.is_safe:
                raise MemoryError(report.warning)

        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._state: np.ndarray = self._zero_state()
        self._gate_count: int = 0
        self._elapsed_ms: float = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Initializers
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_array(cls, amplitudes: np.ndarray) -> "QuantumStateVector":
        """
        Wrap amplitudes computed elsewhere, without copying them: a real-valued state from the whole-round Grover
        engine (module2_sfod/grover.py), for instance, so it is sampled and read like any other state.
        """
        dim = int(amplitudes.size)
        if dim < 2 or dim & (dim - 1):
            raise ValueError("A state vector needs 2^n amplitudes.")
        sv = cls.__new__(cls)
        sv.n_qubits = dim.bit_length() - 1
        sv.dim = dim
        sv._state = amplitudes
        sv._gate_count = 0
        sv._elapsed_ms = 0.0
        return sv

    def _zero_state(self) -> np.ndarray:
        """Return |00…0⟩ state vector."""
        sv = np.zeros(self.dim, dtype=complex)
        sv[0] = 1.0
        return sv

    def reset(self) -> None:
        """Reset to |00…0⟩."""
        self._state = self._zero_state()
        self._gate_count = 0

    def set_state(self, amplitudes: Sequence[complex]) -> None:
        """
        Manually set the state vector.  Must have exactly 2^n elements
        and be normalised to unit norm.
        """
        if len(amplitudes) != self.dim:
            raise ValueError(
                f"Expected {self.dim} amplitudes for {self.n_qubits} qubits, "
                f"got {len(amplitudes)}."
            )
        sv = np.array(amplitudes, dtype=complex)
        norm = np.linalg.norm(sv)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"State vector must be normalised (norm={norm:.6f}).")
        self._state = sv

    # ──────────────────────────────────────────────────────────────────────
    # Views: expose target qubits as their own axes (no copy)
    # ──────────────────────────────────────────────────────────────────────

    def _target_view(self, qubits: Sequence[int]) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Reshape the state (as a view) so that each qubit in `qubits` gets its
        own axis of length 2. Returns (view, {qubit: axis}).

        For sorted targets t0 < t1 < … the view has shape
            (2^t0, 2, 2^(t1-t0-1), 2, …, 2^(n-1-t_last))
        which is exactly the C-order layout of the big-endian index.
        Free blocks of size 1 are left out, so the view never has more than
        n axes (NumPy allows at most 32).
        """
        shape: List[int] = []
        axis_of: Dict[int, int] = {}
        prev = -1
        for q in sorted(qubits):
            gap = q - prev - 1
            if gap:
                shape.append(1 << gap)
            axis_of[q] = len(shape)
            shape.append(2)
            prev = q
        tail = self.n_qubits - prev - 1
        if tail:
            shape.append(1 << tail)
        return self._state.reshape(shape), axis_of

    def _validate_distinct(self, qubits: Sequence[int]) -> None:
        for q in qubits:
            self._validate_qubit(q)
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Gate qubits must be distinct, got {list(qubits)}.")

    # ──────────────────────────────────────────────────────────────────────
    # Core Gate Application (in-place, chunked)
    # ──────────────────────────────────────────────────────────────────────

    def apply_matrix(self, gate: np.ndarray, qubits: Sequence[int]) -> None:
        """
        Apply a 2^k × 2^k unitary to `qubits` in place.

        `qubits[0]` is the most-significant operand of the matrix, e.g. for
        CNOT the matrix rows are ordered |control target⟩.
        """
        qubits = list(qubits)
        k = len(qubits)
        dim_k = 1 << k
        if gate.shape != (dim_k, dim_k):
            raise ValueError(f"A gate on {k} qubit(s) needs a {dim_k}×{dim_k} matrix, got {gate.shape}.")
        self._validate_distinct(qubits)

        view, axis_of = self._target_view(qubits)
        axes = [axis_of[q] for q in qubits]

        def at(s: int, base: Tuple) -> Tuple:
            sl = list(base)
            for j, a in enumerate(axes):
                sl[a] = (s >> (k - 1 - j)) & 1
            return tuple(sl)

        # Classify rows: identity rows are skipped, pure-diagonal rows are an
        # in-place phase multiply, and only the remaining rows need copies.
        # (Unitarity guarantees the columns of the two groups never overlap.)
        diag_rows, dense_rows = [], []
        for r in range(dim_k):
            off_diagonal = np.delete(gate[r], r)
            if not np.any(off_diagonal):
                if gate[r, r] != 1:
                    diag_rows.append(r)
            else:
                dense_rows.append(r)

        everything = (slice(None),) * view.ndim
        for r in diag_rows:
            view[at(r, everything)] *= gate[r, r]

        if dense_rows:
            row_terms = {r: [(c, gate[r, c]) for c in np.flatnonzero(gate[r])] for r in dense_rows}
            needed = sorted({c for terms in row_terms.values() for c, _ in terms})
            for base in _iter_chunks(view, axes):
                comps = {c: view[at(c, base)].copy() for c in needed}
                for r in dense_rows:
                    acc = None
                    for c, w in row_terms[r]:
                        term = comps[c] if w == 1 else w * comps[c]
                        acc = term if acc is None else acc + term
                    view[at(r, base)] = acc

        self._gate_count += 1

    # Backwards-compatible names used elsewhere in the project.
    def _apply_single(self, gate: np.ndarray, qubit: int) -> None:
        self.apply_matrix(gate, [qubit])

    def _apply_two(self, gate: np.ndarray, qubit1: int, qubit2: int) -> None:
        if qubit1 == qubit2:
            raise ValueError("qubit1 and qubit2 must be different.")
        self.apply_matrix(gate, [qubit1, qubit2])

    # ── Multi-controlled gates (any number of controls) ────────────────────

    def apply_mcx(self, controls: Sequence[int], target: int) -> None:
        """Multi-controlled X: flip `target` on the subspace where every control is |1⟩."""
        controls = list(controls)
        self._validate_distinct(controls + [target])
        view, axis_of = self._target_view(controls + [target])
        index = [slice(None)] * view.ndim
        for c in controls:
            index[axis_of[c]] = 1
        sub = view[tuple(index)]  # basic indexing → still a view
        t_axis = axis_of[target] - sum(1 for c in controls if axis_of[c] < axis_of[target])
        for base in _iter_chunks(sub, [t_axis]):
            s0, s1 = list(base), list(base)
            s0[t_axis], s1[t_axis] = 0, 1
            s0, s1 = tuple(s0), tuple(s1)
            tmp = sub[s0].copy()
            sub[s0] = sub[s1]
            sub[s1] = tmp
        self._gate_count += 1

    def apply_mcp(self, qubits: Sequence[int], phi: float) -> None:
        """Multi-controlled phase: multiply the |11…1⟩ component of `qubits` by e^{iφ}."""
        qubits = list(qubits)
        self._validate_distinct(qubits)
        view, axis_of = self._target_view(qubits)
        index = [slice(None)] * view.ndim
        for q in qubits:
            index[axis_of[q]] = 1
        view[tuple(index)] *= np.exp(1j * phi)
        self._gate_count += 1

    def apply_mcz(self, qubits: Sequence[int]) -> None:
        """Multi-controlled Z: flip the sign of the |11…1⟩ component of `qubits`."""
        qubits = list(qubits)
        self._validate_distinct(qubits)
        view, axis_of = self._target_view(qubits)
        index = [slice(None)] * view.ndim
        for q in qubits:
            index[axis_of[q]] = 1
        view[tuple(index)] *= -1
        self._gate_count += 1

    # ── Black-box operators (oracles) ──────────────────────────────────────

    def apply_phase_oracle(self, marked: Iterable[int]) -> None:
        """
        Grover phase oracle  O|x⟩ = (−1)^{f(x)} |x⟩  where f(x)=1 for the
        marked basis states. A diagonal unitary; its gate-level form is
        X-conjugated multi-controlled Z per marked state.
        """
        idx = np.unique(np.asarray(list(marked), dtype=np.int64))
        if idx.size == 0:
            self._gate_count += 1
            return
        if idx[0] < 0 or idx[-1] >= self.dim:
            raise IndexError(f"Marked states must be in [0, {self.dim - 1}].")
        self._state[idx] *= -1
        self._gate_count += 1

    def apply_diagonal(self, diagonal: np.ndarray) -> None:
        """
        Multiply the state by a full diagonal unitary (length 2^n), e.g. the
        QAOA phase separator exp(−iγ·H_C) for a diagonal cost Hamiltonian.
        Mathematically identical to its RZ/RZZ gate decomposition.
        """
        diagonal = np.asarray(diagonal)
        if diagonal.shape != (self.dim,):
            raise ValueError(f"Diagonal must have {self.dim} entries, got {diagonal.shape}.")
        self._state *= diagonal
        self._gate_count += 1

    def apply_controlled_permutation(self, controls: Sequence[int], targets: Sequence[int],
                                     permutation: Sequence[int]) -> None:
        """
        Apply the permutation unitary U|t⟩ = |perm[t]⟩ to the `targets`
        register (targets[0] = most-significant bit), only where every control
        qubit is |1⟩. Used for Shor's controlled modular multiplication.
        """
        controls, targets = list(controls), list(targets)
        self._validate_distinct(controls + targets)
        if self.n_qubits > _MAX_PERMUTATION_QUBITS:
            raise MemoryError(
                f"Controlled permutations are limited to {_MAX_PERMUTATION_QUBITS} qubits."
            )
        k = len(targets)
        perm = np.asarray(permutation, dtype=np.int64)
        if perm.shape != (1 << k,) or not np.array_equal(np.sort(perm), np.arange(1 << k)):
            raise ValueError(f"Permutation must be a bijection on 0…{(1 << k) - 1}.")

        n = self.n_qubits
        idx = np.arange(self.dim, dtype=np.int64)
        t = np.zeros(self.dim, dtype=np.int64)
        for q in targets:
            t = (t << 1) | ((idx >> (n - 1 - q)) & 1)
        new_t = perm[t]
        new_idx = idx.copy()
        for j, q in enumerate(targets):
            pos = n - 1 - q
            new_idx &= ~np.int64(1 << pos)
            new_idx |= ((new_t >> (k - 1 - j)) & 1) << pos
        if controls:
            cmask = 0
            for c in controls:
                cmask |= 1 << (n - 1 - c)
            new_idx = np.where((idx & cmask) == cmask, new_idx, idx)

        new_state = np.empty_like(self._state)
        new_state[new_idx] = self._state
        self._state = new_state
        self._gate_count += 1

    # ──────────────────────────────────────────────────────────────────────
    # Public Gate Interface
    # ──────────────────────────────────────────────────────────────────────

    def apply_gate(self, name: str, qubits: List[int], params: List[float] = None) -> None:
        """
        Universal gate application entry-point.

        Args:
            name:   Gate name string (case-insensitive), e.g. "H", "CNOT", "RX", "CCX", "MCZ".
            qubits: List of target qubits.  Length must match gate type
                    (MCX: controls…, target — MCZ / MCP: any number).
            params: Optional rotation angles / phase parameters in radians.
        """
        params = list(params or [])
        upper = name.upper()

        if upper in MULTI_QUBIT_GATES:
            if not qubits:
                raise ValueError(f"'{upper}' needs at least one qubit.")
            if upper == "MCX":
                self.apply_mcx(qubits[:-1], qubits[-1])
            elif upper == "MCZ":
                self.apply_mcz(qubits)
            else:
                if len(params) != 1:
                    raise ValueError("'MCP' takes exactly 1 angle parameter.")
                self.apply_mcp(qubits, float(params[0]))
            return

        arity = gate_arity(upper)
        if len(qubits) != arity:
            raise ValueError(f"Gate '{upper}' expects exactly {arity} qubit(s), got {len(qubits)}.")
        self.apply_matrix(get_gate_matrix(upper, params), qubits)

    # Convenience wrappers
    def h(self, qubit: int) -> None:            self.apply_gate("H",    [qubit])
    def x(self, qubit: int) -> None:            self.apply_gate("X",    [qubit])
    def y(self, qubit: int) -> None:            self.apply_gate("Y",    [qubit])
    def z(self, qubit: int) -> None:            self.apply_gate("Z",    [qubit])
    def s(self, qubit: int) -> None:            self.apply_gate("S",    [qubit])
    def t(self, qubit: int) -> None:            self.apply_gate("T",    [qubit])
    def rx(self, theta: float, qubit: int) -> None: self.apply_gate("RX", [qubit], [theta])
    def ry(self, theta: float, qubit: int) -> None: self.apply_gate("RY", [qubit], [theta])
    def rz(self, theta: float, qubit: int) -> None: self.apply_gate("RZ", [qubit], [theta])
    def cnot(self, ctrl: int, tgt: int) -> None: self.apply_gate("CNOT", [ctrl, tgt])
    def cz(self, ctrl: int, tgt: int) -> None:   self.apply_gate("CZ",   [ctrl, tgt])
    def swap(self, q1: int, q2: int) -> None:    self.apply_gate("SWAP", [q1, q2])
    def ccx(self, c1: int, c2: int, tgt: int) -> None: self.apply_gate("CCX", [c1, c2, tgt])

    # ──────────────────────────────────────────────────────────────────────
    # Measurement
    # ──────────────────────────────────────────────────────────────────────

    def probabilities(self) -> np.ndarray:
        """Return probability of each basis state (array of length 2^n)."""
        return np.abs(self._state) ** 2

    def norm_squared(self) -> float:
        """Σ|amplitude|² computed chunk-wise (should be 1 up to rounding)."""
        total = 0.0
        for start in range(0, self.dim, _CHUNK):
            chunk = self._state[start:start + _CHUNK]
            total += float(np.vdot(chunk, chunk).real)
        return total

    def marginal_probabilities(self, qubits: Sequence[int]) -> np.ndarray:
        """
        Probability distribution of measuring only `qubits` (in the given
        order, first = most-significant bit of the returned index).
        """
        qubits = list(qubits)
        self._validate_distinct(qubits)
        probs = self.probabilities().reshape([2] * self.n_qubits)
        others = tuple(q for q in range(self.n_qubits) if q not in qubits)
        marg = probs.sum(axis=others) if others else probs
        # remaining axes are in ascending qubit order — reorder to `qubits`
        ascending = sorted(qubits)
        marg = np.transpose(marg, [ascending.index(q) for q in qubits])
        return marg.reshape(-1)

    def expectation_diagonal(self, values: np.ndarray) -> float:
        """⟨ψ|D|ψ⟩ for a diagonal observable D given by its 2^n diagonal entries."""
        return float(np.dot(self.probabilities(), np.asarray(values, dtype=float)))

    def sample_indices(self, shots: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw `shots` measurement outcomes (basis-state indices), chunk-wise."""
        rng = rng or np.random.default_rng()
        if shots <= 0:
            return np.zeros(0, dtype=np.int64)
        if self.dim <= 4 * _CHUNK:
            p = self.probabilities()
            return rng.choice(self.dim, size=shots, p=p / p.sum())

        # Large registers: inverse-CDF sampling over chunks (two passes).
        starts = list(range(0, self.dim, _CHUNK))
        sums = np.array([float(np.vdot(self._state[s:s + _CHUNK], self._state[s:s + _CHUNK]).real)
                         for s in starts])
        total = sums.sum()
        u = np.sort(rng.random(shots)) * total
        bounds = np.concatenate([[0.0], np.cumsum(sums)])
        out = []
        for i, s in enumerate(starts):
            lo, hi = bounds[i], bounds[i + 1]
            sel = u[(u >= lo) & (u < hi)] if i < len(starts) - 1 else u[u >= lo]
            if sel.size == 0:
                continue
            cdf = np.cumsum(np.abs(self._state[s:s + _CHUNK]) ** 2)
            local = np.searchsorted(cdf, sel - lo, side="right")
            out.append(s + np.minimum(local, cdf.size - 1))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def measure_all(self, shots: int = 1024) -> Dict[str, int]:
        """
        Sample the probability distribution `shots` times.

        Returns a dict mapping basis state strings to counts,
        e.g. {'|000>': 512, '|111>': 512}.
        """
        indices = self.sample_indices(shots)
        uniq, cnt = np.unique(indices, return_counts=True)
        return {f"|{int(i):0{self.n_qubits}b}>": int(c) for i, c in zip(uniq, cnt)}

    def top_states(self, k: int) -> List[Tuple[int, float]]:
        """The k most probable basis states as (index, probability), highest first."""
        k = max(1, min(k, self.dim))
        best_idx = np.zeros(0, dtype=np.int64)
        best_p = np.zeros(0)
        for start in range(0, self.dim, _CHUNK):
            p = np.abs(self._state[start:start + _CHUNK]) ** 2
            if p.size > k:
                part = np.argpartition(p, p.size - k)[p.size - k:]
            else:
                part = np.arange(p.size)
            best_idx = np.concatenate([best_idx, part + start])
            best_p = np.concatenate([best_p, p[part]])
            if best_p.size > k:
                keep = np.argpartition(best_p, best_p.size - k)[best_p.size - k:]
                best_idx, best_p = best_idx[keep], best_p[keep]
        order = np.argsort(-best_p, kind="stable")
        return [(int(best_idx[i]), float(best_p[i])) for i in order]

    def qubit_marginal(self, qubit: int) -> Tuple[float, float, complex]:
        """
        Reduced single-qubit density matrix entries (ρ00, ρ11, ρ01),
        computed chunk-wise without copying the state.
        """
        self._validate_qubit(qubit)
        v = self._state.reshape(1 << qubit, 2, 1 << (self.n_qubits - qubit - 1))
        p0 = p1 = 0.0
        c01 = 0j
        for base in _iter_chunks(v, [1]):
            a0 = v[base[0], 0, base[2]]
            a1 = v[base[0], 1, base[2]]
            p0 += float(np.sum(np.abs(a0) ** 2))
            p1 += float(np.sum(np.abs(a1) ** 2))
            c01 += complex(np.sum(a0 * np.conj(a1)))
        return p0, p1, c01

    def bloch_vector(self, qubit: int) -> Dict[str, float]:
        """
        Bloch vector (x, y, z) of one qubit's reduced state. Its length is 1
        for a pure (unentangled) qubit and < 1 when the qubit is entangled
        with the rest of the register.
        """
        p0, p1, c01 = self.qubit_marginal(qubit)
        x, y, z = 2 * c01.real, -2 * c01.imag, p0 - p1
        purity = p0 * p0 + p1 * p1 + 2 * abs(c01) ** 2
        return {
            "qubit": qubit,
            "x": round(x, 6), "y": round(y, 6), "z": round(z, 6),
            "p0": round(p0, 6), "p1": round(p1, 6),
            "purity": round(purity, 6),
            "length": round(float(np.sqrt(x * x + y * y + z * z)), 6),
        }

    def qubit_one_probabilities(self) -> np.ndarray:
        """
        The chance that each qubit reads 1, for all qubits in one chunked pass
        (no full-size temporary, so it also works at 28 qubits).

        Chunks are aligned blocks of 2^low amplitudes: inside a block the last
        `low` qubits vary (summed per axis), the first n − low are fixed by the
        block number (the block's total goes to every qubit set in it).
        """
        n = self.n_qubits
        ones = np.zeros(n)
        size = min(self.dim, _CHUNK)
        low = size.bit_length() - 1
        high = n - low
        for block, start in enumerate(range(0, self.dim, size)):
            p = np.abs(self._state[start:start + size]) ** 2
            total = float(p.sum())
            for q in range(high):
                if (block >> (high - 1 - q)) & 1:
                    ones[q] += total
            for k in range(low):
                ones[high + k] += float(p.reshape(1 << k, 2, -1)[:, 1, :].sum())
        return ones

    def bloch_vectors(self) -> List[Dict[str, float]]:
        return [self.bloch_vector(q) for q in range(self.n_qubits)]

    def measure_qubit(self, qubit: int, collapse: bool = True) -> Tuple[int, float]:
        """
        Measure a single qubit.  Returns (outcome, probability).

        If collapse=True, the state vector is projected onto the measurement
        outcome and renormalised in place (simulates wavefunction collapse).
        """
        p0, p1, _ = self.qubit_marginal(qubit)
        total = p0 + p1
        prob_1 = p1 / total if total > 0 else 0.0
        outcome = int(np.random.random() < prob_1)
        prob = prob_1 if outcome == 1 else 1.0 - prob_1

        if collapse:
            v = self._state.reshape(1 << qubit, 2, 1 << (self.n_qubits - qubit - 1))
            v[:, 1 - outcome, :] = 0
            if prob > 1e-12:
                self._state /= np.sqrt(prob * total)

        return outcome, prob

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics & Visualisation Helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_amplitudes(self, max_states: int = 64) -> List[Dict]:
        """
        Return amplitude data for frontend visualisation.

        Limits output to `max_states` basis states (the most probable ones),
        listed in basis-index order.
        """
        top = sorted(self.top_states(max_states))
        result = []
        for idx, prob in top:
            amp = self._state[idx]
            result.append({
                "state":       f"|{idx:0{self.n_qubits}b}>",
                "index":       int(idx),
                "real":        round(float(amp.real), 8),
                "imag":        round(float(amp.imag), 8),
                "probability": round(float(prob), 8),
                "magnitude":   round(float(np.abs(amp)), 8),
                "phase_deg":   round(float(np.degrees(np.angle(amp))), 4),
            })
        return result

    def is_entangled(self, max_qubits_checked: int = 20) -> bool:
        """
        Entanglement test for the (pure) register state.

        A pure n-qubit state is a full product state if and only if every
        single-qubit reduced state is pure (purity = 1). If any qubit's
        reduced state is mixed, that qubit is entangled with the others.
        For very large registers only the first `max_qubits_checked` qubits
        are examined to bound the cost.
        """
        if self.n_qubits < 2:
            return False
        for q in range(min(self.n_qubits, max_qubits_checked)):
            p0, p1, c01 = self.qubit_marginal(q)
            purity = p0 * p0 + p1 * p1 + 2 * abs(c01) ** 2
            if purity < 1 - 1e-9:
                return True
        return False

    def is_superposition(self, qubit: int = 0) -> bool:
        """
        Returns True if the given qubit is in superposition
        (probability of |0> is not 0 or 1).
        """
        _, p1, _ = self.qubit_marginal(qubit)
        return not (np.isclose(p1, 0.0, atol=1e-9) or np.isclose(p1, 1.0, atol=1e-9))

    def fidelity(self, other: "QuantumStateVector") -> float:
        """Compute |⟨ψ|φ⟩|² between this state and another."""
        if self.dim != other.dim:
            raise ValueError("State vectors have different dimensions.")
        return float(np.abs(np.vdot(self._state, other._state)) ** 2)

    def summary(self) -> Dict:
        """Return a JSON-serialisable summary of the current state."""
        (best_idx, best_p), = self.top_states(1)
        return {
            "n_qubits":    self.n_qubits,
            "dim":         self.dim,
            "gate_count":  self._gate_count,
            "norm":        round(self.norm_squared(), 10),
            "is_entangled": self.is_entangled(),
            "max_prob_state": f"|{best_idx:0{self.n_qubits}b}>",
            "max_prob":    round(best_p, 8),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _validate_qubit(self, qubit: int) -> None:
        if not (0 <= qubit < self.n_qubits):
            raise IndexError(
                f"Qubit index {qubit} out of range for {self.n_qubits}-qubit system "
                f"(valid: 0–{self.n_qubits - 1})."
            )

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()
