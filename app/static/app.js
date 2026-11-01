const state = {
  numQubits: 1,
  gates: [],
  challengeId: null,
  challenges: [],
};

const $ = (id) => document.getElementById(id);
const targetSelect = $("target-qubit");
const controlSelect = $("control-qubit");
const secondControlSelect = $("second-control");
const gateSelect = $("gate-name");

function isRotation(gate) {
  return ["RX", "RY", "RZ"].includes(gate);
}

function needsControls(gate) {
  return ["CNOT", "CZ", "SWAP"].includes(gate);
}

function updateGateControls() {
  const gate = gateSelect.value;
  $("control-field").classList.toggle("hidden", !needsControls(gate) && gate !== "CCX");
  $("second-control-field").classList.toggle("hidden", gate !== "CCX");
  $("angle-field").classList.toggle("hidden", !isRotation(gate));
}

function refreshQubitOptions() {
  const options = Array.from({ length: state.numQubits }, (_, qubit) =>
    `<option value="${qubit}">Qubit ${qubit}</option>`).join("");
  targetSelect.innerHTML = options;
  controlSelect.innerHTML = options;
  secondControlSelect.innerHTML = options;
  if (state.numQubits > 1) controlSelect.value = "1";
  if (state.numQubits > 2) secondControlSelect.value = "2";
}

function gateDescription(gate) {
  if (gate.gate === "CCX") return `CCX · controls ${gate.control_qubit}, ${gate.control2} → target ${gate.qubit}`;
  if (needsControls(gate.gate)) return `${gate.gate} · control ${gate.control_qubit} → target ${gate.qubit}`;
  if (isRotation(gate.gate)) return `${gate.gate}(${Number(gate.angle).toFixed(3)}) · qubit ${gate.qubit}`;
  return `${gate.gate} · qubit ${gate.qubit}`;
}

function renderGates() {
  $("empty-circuit").classList.toggle("hidden", state.gates.length > 0);
  $("gate-list").innerHTML = state.gates.map((gate, index) =>
    `<li class="gate-item"><span>${index + 1}. ${gateDescription(gate)}</span>
      <button class="remove-gate" type="button" data-remove="${index}" aria-label="Remove gate ${index + 1}">Remove</button></li>`
  ).join("");
  document.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => {
      state.gates.splice(Number(button.dataset.remove), 1);
      renderGates();
    });
  });
}

function addGate() {
  if (state.gates.length >= 40) {
    setStatus("The circuit limit is 40 gates.");
    return;
  }
  const gate = gateSelect.value;
  const operation = { gate, qubit: Number(targetSelect.value) };
  if (needsControls(gate) || gate === "CCX") operation.control_qubit = Number(controlSelect.value);
  if (gate === "CCX") operation.control2 = Number(secondControlSelect.value);
  if (isRotation(gate)) operation.angle = Number($("gate-angle").value);
  if ((needsControls(gate) || gate === "CCX") && operation.control_qubit === operation.qubit) {
    setStatus("Choose a control qubit different from the target.");
    return;
  }
  if (gate === "CCX" && (operation.control2 === operation.qubit || operation.control2 === operation.control_qubit)) {
    setStatus("CCX needs three different qubits.");
    return;
  }
  state.gates.push(operation);
  renderGates();
  setStatus("");
}

function setStatus(message, isError = false) {
  const element = $("simulation-status");
  element.textContent = message;
  element.style.color = isError ? "var(--danger)" : "";
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || body.error || `Request failed (${response.status}).`);
  return body;
}

function renderChallenges() {
  $("challenge-list").innerHTML = state.challenges.map((challenge) => `
    <article class="challenge-card">
      <p class="challenge-concept">${challenge.concept}</p>
      <h3>${challenge.title}</h3>
      <p>${challenge.goal}</p>
      <p><strong>Hint:</strong> ${challenge.hint}</p>
      ${challenge.completed ? '<p class="completion">Completed ✓</p>' : ""}
      <button class="button button-secondary" type="button" data-challenge="${challenge.id}">Start challenge</button>
    </article>`).join("");
  document.querySelectorAll("[data-challenge]").forEach((button) => {
    button.addEventListener("click", () => {
      const challenge = state.challenges.find((item) => item.id === button.dataset.challenge);
      state.challengeId = challenge.id;
      state.numQubits = challenge.num_qubits;
      $("qubit-count").value = String(challenge.num_qubits);
      state.gates = [];
      refreshQubitOptions();
      renderGates();
      setStatus(`Challenge selected: ${challenge.title}. Build its circuit above.`);
      $("simulator").scrollIntoView({ behavior: "smooth" });
    });
  });
}

async function loadChallenges() {
  state.challenges = await requestJson("/api/challenges");
  renderChallenges();
}

function renderResult(response) {
  const result = response.result;
  const probabilities = Object.entries(result.probabilities)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([label, probability]) => `<tr><td>${label.replaceAll(">", "⟩")}</td><td>${(probability * 100).toFixed(2)}%</td></tr>`)
    .join("");
  const challengeText = response.challenge_id
    ? `<span>${response.challenge_passed ? "Challenge passed ✓" : "Challenge not complete yet"}</span>`
    : "";
  const panel = $("result-panel");
  panel.innerHTML = `
    <h3>Simulation result</h3>
    <div class="result-summary">
      <span>${result.n_qubits} qubits</span>
      <span>${result.gate_count} gates</span>
      <span>${result.elapsed_ms.toFixed(2)} ms</span>
      <span>${result.summary.is_entangled ? "Entangled" : "Not entangled"}</span>
      <span>Saved run #${response.run_id}</span>${challengeText}
    </div>
    <table class="probability-table"><thead><tr><th>State</th><th>Probability</th></tr></thead><tbody>${probabilities}</tbody></table>`;
  panel.classList.remove("hidden");
}

async function runSimulation() {
  const button = $("run-circuit");
  button.disabled = true;
  setStatus("Simulating and saving your run…");
  try {
    const response = await requestJson("/api/simulations", {
      method: "POST",
      body: JSON.stringify({
        num_qubits: state.numQubits,
        gates: state.gates,
        shots: Number($("shot-count").value),
        challenge_id: state.challengeId,
      }),
    });
    renderResult(response);
    setStatus("Run completed and saved.");
    await Promise.all([loadHistory(), loadChallenges()]);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function loadHistory() {
  const runs = await requestJson("/api/history?limit=20");
  const history = $("history-list");
  if (runs.length === 0) {
    history.innerHTML = '<p class="empty-state">No runs yet. Simulate a circuit to create your first saved run.</p>';
    return;
  }
  history.innerHTML = runs.map((run) => {
    const challenge = state.challenges.find((item) => item.id === run.challenge_id);
    const title = challenge ? challenge.title : "Free circuit";
    const result = run.challenge_passed === true ? " · Challenge passed" :
      run.challenge_passed === false ? " · Challenge not complete" : "";
    const date = run.created_at ? new Date(run.created_at).toLocaleString() : "Just now";
    return `<article class="history-item"><strong>#${run.id} · ${title}${result}</strong>
      <span>${run.num_qubits} qubits · ${run.gate_count} gates · ${date}</span></article>`;
  }).join("");
}

async function checkApi() {
  try {
    await requestJson("/api/health");
    $("api-status").textContent = "API connected";
    $("api-status").style.color = "var(--accent)";
  } catch {
    $("api-status").textContent = "API unavailable";
    $("api-status").style.color = "var(--danger)";
  }
}

$("qubit-count").addEventListener("change", (event) => {
  state.numQubits = Number(event.target.value);
  state.gates = [];
  state.challengeId = null;
  refreshQubitOptions();
  renderGates();
  setStatus("Circuit cleared after changing its qubit count.");
});
gateSelect.addEventListener("change", updateGateControls);
$("add-gate").addEventListener("click", addGate);
$("clear-gates").addEventListener("click", () => {
  state.gates = [];
  state.challengeId = null;
  renderGates();
  setStatus("Circuit cleared.");
});
$("run-circuit").addEventListener("click", runSimulation);
$("refresh-history").addEventListener("click", () => loadHistory().catch((error) => setStatus(error.message, true)));

refreshQubitOptions();
updateGateControls();
renderGates();
Promise.all([checkApi(), loadChallenges()]).then(loadHistory).catch((error) => {
  $("challenge-list").innerHTML = `<p class="empty-state">${error.message}</p>`;
  $("history-list").innerHTML = `<p class="empty-state">History unavailable: ${error.message}</p>`;
});
