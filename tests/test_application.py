import pytest


def simulate(client, gates, num_qubits=1, challenge_id=None):
    return client.post(
        "/api/simulations",
        json={
            "num_qubits": num_qubits,
            "gates": gates,
            "shots": 128,
            "challenge_id": challenge_id,
        },
    )


def test_homepage_serves_the_application(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Quantum Learning Lab" in response.text


def test_health_endpoint_is_available(client):
    assert client.get("/api/health").json() == {"status": "healthy"}


def test_challenge_list_is_available_without_login(client):
    response = client.get("/api/challenges")
    assert response.status_code == 200
    assert {challenge["id"] for challenge in response.json()} == {
        "flip", "coin", "bell", "uniform", "and",
    }


def test_empty_circuit_stays_in_zero_state_and_is_saved(client):
    response = simulate(client, []).json()
    assert response["result"]["probabilities"] == {"|0>": 1.0}
    assert response["run_id"] == 1


def test_pauli_x_flips_zero_to_one(client):
    response = simulate(client, [{"gate": "X", "qubit": 0}])
    assert response.status_code == 200
    assert response.json()["result"]["probabilities"] == {"|1>": 1.0}


def test_hadamard_creates_equal_superposition(client):
    response = simulate(client, [{"gate": "H", "qubit": 0}])
    assert response.json()["result"]["probabilities"] == {"|0>": 0.5, "|1>": 0.5}


def test_rotation_gate_uses_requested_angle(client):
    response = simulate(client, [{"gate": "RY", "qubit": 0, "angle": 1.0471975512}])
    probabilities = response.json()["result"]["probabilities"]
    assert probabilities["|0>"] == pytest.approx(0.75, abs=1e-6)
    assert probabilities["|1>"] == pytest.approx(0.25, abs=1e-6)


def test_bell_circuit_creates_entanglement(client):
    response = simulate(
        client,
        [
            {"gate": "H", "qubit": 0},
            {"gate": "CNOT", "qubit": 1, "control_qubit": 0},
        ],
        num_qubits=2,
    )
    result = response.json()["result"]
    assert result["probabilities"] == {"|00>": 0.5, "|11>": 0.5}
    assert result["summary"]["is_entangled"] is True


def test_flip_challenge_records_completion_in_history(client):
    response = simulate(client, [{"gate": "X", "qubit": 0}], challenge_id="flip")
    assert response.status_code == 200
    assert response.json()["challenge_passed"] is True
    assert client.get("/api/challenges").json()[0]["completed"] is True


def test_bell_challenge_requires_both_expected_probabilities_and_entanglement(client):
    response = simulate(
        client,
        [
            {"gate": "H", "qubit": 0},
            {"gate": "CNOT", "qubit": 1, "control_qubit": 0},
        ],
        num_qubits=2,
        challenge_id="bell",
    )
    assert response.json()["challenge_passed"] is True


def test_history_contains_persisted_gate_list_and_challenge(client):
    simulate(client, [{"gate": "X", "qubit": 0}], challenge_id="flip")
    history = client.get("/api/history").json()
    assert len(history) == 1
    assert history[0]["challenge_id"] == "flip"
    assert history[0]["challenge_passed"] is True
    assert history[0]["gates"][0]["gate"] == "X"


def test_history_limit_is_applied(client):
    for _ in range(3):
        simulate(client, [])
    assert len(client.get("/api/history?limit=2").json()) == 2


def test_unknown_challenge_is_rejected(client):
    response = simulate(client, [], challenge_id="missing")
    assert response.status_code == 422


def test_challenge_requires_its_defined_qubit_count(client):
    response = simulate(client, [], num_qubits=2, challenge_id="flip")
    assert response.status_code == 422


@pytest.mark.parametrize(
    "gate,num_qubits",
    [
        ([{"gate": "UNKNOWN", "qubit": 0}], 1),
        ([{"gate": "H", "qubit": 1}], 1),
        ([{"gate": "CNOT", "qubit": 0, "control_qubit": 0}], 2),
        ([{"gate": "CCX", "qubit": 2, "control_qubit": 0, "control2": 0}], 3),
        ([{"gate": "H", "qubit": 0, "angle": 1.0}], 1),
    ],
)
def test_invalid_gate_operations_are_rejected(client, gate, num_qubits):
    response = simulate(client, gate, num_qubits=num_qubits)
    assert response.status_code == 422


def test_circuit_over_qubit_limit_is_rejected(client):
    response = client.post("/api/simulations", json={"num_qubits": 7, "gates": []})
    assert response.status_code == 422


def test_circuit_over_gate_limit_is_rejected(client):
    gates = [{"gate": "X", "qubit": 0}] * 41
    response = simulate(client, gates)
    assert response.status_code == 422


def test_history_limit_outside_valid_range_is_rejected(client):
    assert client.get("/api/history?limit=100").status_code == 422
