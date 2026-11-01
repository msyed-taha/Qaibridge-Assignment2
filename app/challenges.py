from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Challenge:
    id: str
    title: str
    concept: str
    num_qubits: int
    goal: str
    hint: str
    expected: dict[int, float]
    requires_entanglement: bool = False
    required_gate: str | None = None


CHALLENGES = (
    Challenge(
        id="flip",
        title="Flip the bit",
        concept="Pauli-X",
        num_qubits=1,
        goal="Make the qubit read 1 with certainty.",
        hint="Apply an X gate to qubit 0.",
        expected={1: 1.0},
    ),
    Challenge(
        id="coin",
        title="Quantum coin",
        concept="Superposition",
        num_qubits=1,
        goal="Give the outcomes 0 and 1 equal probability.",
        hint="Apply an H gate to qubit 0.",
        expected={0: 0.5, 1: 0.5},
    ),
    Challenge(
        id="bell",
        title="Entangle two qubits",
        concept="Entanglement",
        num_qubits=2,
        goal="Prepare 00 and 11 with equal probability.",
        hint="Apply H to qubit 0, then CNOT from qubit 0 to qubit 1.",
        expected={0: 0.5, 3: 0.5},
        requires_entanglement=True,
    ),
    Challenge(
        id="uniform",
        title="All at once",
        concept="Quantum parallelism",
        num_qubits=3,
        goal="Give each of the 8 outcomes a 12.5% probability.",
        hint="Apply H to each of the three qubits.",
        expected={state: 0.125 for state in range(8)},
    ),
    Challenge(
        id="and",
        title="A reversible AND",
        concept="Toffoli / reversible logic",
        num_qubits=3,
        goal="Use reversible gates to prepare state 111 with certainty.",
        hint="Apply X to qubits 0 and 1, then CCX with those controls and qubit 2 as target.",
        expected={7: 1.0},
        required_gate="CCX",
    ),
)

CHALLENGES_BY_ID = {challenge.id: challenge for challenge in CHALLENGES}
PROBABILITY_TOLERANCE = 0.02


def evaluate_challenge(challenge: Challenge, result: dict[str, Any]) -> bool:
    probabilities = result["probabilities"]
    expected = {
        f"|{state:0{challenge.num_qubits}b}>": probability
        for state, probability in challenge.expected.items()
    }
    all_outcomes_match = all(
        abs(probabilities.get(f"|{state:0{challenge.num_qubits}b}>", 0.0) - probability)
        <= PROBABILITY_TOLERANCE
        for state, probability in challenge.expected.items()
    )
    unexpected_outcomes_are_zero = all(
        abs(probability) <= PROBABILITY_TOLERANCE
        for label, probability in probabilities.items()
        if label not in expected
    )
    entanglement_matches = (
        not challenge.requires_entanglement or result["summary"]["is_entangled"]
    )
    gates_match = (
        challenge.required_gate is None
        or challenge.required_gate in {operation["gate"] for operation in result["operations"]}
    )
    return (
        all_outcomes_match
        and unexpected_outcomes_are_zero
        and entanglement_matches
        and gates_match
    )
