const API_BASE = "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);

const healthDot = $("healthDot");
const healthText = $("healthText");
const statApi = $("statApi");
const queryForm = $("queryForm");
const queryInput = $("queryInput");
const kInput = $("kInput");
const searchButton = $("searchButton");
const sampleButton = $("sampleButton");
const searchStatus = $("searchStatus");
const resultsList = $("resultsList");
const insertForm = $("insertForm");
const insertId = $("insertId");
const insertVector = $("insertVector");
const insertStatus = $("insertStatus");
const deleteForm = $("deleteForm");
const deleteId = $("deleteId");
const deleteStatus = $("deleteStatus");

function setStatus(element, message, kind = "") {
  element.textContent = message;
  element.className = "status-line " + kind;
}

function parseVector(text) {
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("Vector must be valid JSON, e.g. [0.1, -0.2, ...].");
  }

  if (!Array.isArray(value)) {
    throw new Error("Vector must be a JSON array.");
  }

  if (value.length !== 128) {
    throw new Error(`Expected 128 dimensions, received ${value.length}.`);
  }

  if (!value.every((x) => typeof x === "number" && Number.isFinite(x))) {
    throw new Error("Every vector component must be a finite number.");
  }

  return value;
}

async function apiFetch(path, options = {}) {
  const response = await fetch(API_BASE + path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });

  const raw = await response.text();
  let body = null;
  try { body = raw ? JSON.parse(raw) : null; } catch {}

  if (!response.ok) {
    const detail = body?.detail;
    throw new Error(
      typeof detail === "string" ? detail :
      detail ? JSON.stringify(detail) :
      `${response.status} ${response.statusText}`
    );
  }

  return body;
}

async function checkHealth() {
  try {
    const data = await apiFetch("/health");
    healthDot.className = "ok";
    healthText.textContent = "API is healthy";
    statApi.textContent = "Online";
    statApi.className = "stat-value ok";
    return data;
  } catch (error) {
    healthDot.className = "bad";
    healthText.textContent = "API unavailable";
    statApi.textContent = "Offline";
    statApi.className = "stat-value bad";
    return null;
  }
}

function renderResults(results) {
  resultsList.innerHTML = "";

  if (!Array.isArray(results) || results.length === 0) {
    resultsList.innerHTML = '<p class="empty-note">No results returned.</p>';
    return;
  }

  results.forEach((result, index) => {
    const row = document.createElement("div");
    row.className = "result-row";

    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = `${index + 1}.`;

    const id = document.createElement("span");
    id.className = "id";
    id.textContent = result.id ?? "—";

    const score = document.createElement("span");
    score.className = "score";
    score.textContent =
      typeof result.score === "number" ? result.score.toFixed(6) : String(result.score ?? "—");

    row.append(rank, id, score);
    resultsList.appendChild(row);
  });
}

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(searchStatus, "");

  try {
    const vector = parseVector(queryInput.value.trim());
    const k = Math.max(1, Math.min(100, Number.parseInt(kInput.value, 10) || 10));

    searchButton.disabled = true;
    setStatus(searchStatus, "Searching IVF-Flat…");

    const started = performance.now();
    const data = await apiFetch("/search", {
      method: "POST",
      body: JSON.stringify({vector, k}),
    });
    const elapsed = performance.now() - started;

    renderResults(data?.results ?? data);
    setStatus(searchStatus, `Search completed in ${elapsed.toFixed(2)} ms.`, "ok");
  } catch (error) {
    renderResults([]);
    setStatus(searchStatus, error.message, "error");
  } finally {
    searchButton.disabled = false;
  }
});

sampleButton.addEventListener("click", () => {
  const vector = Array.from({length: 128}, (_, i) =>
    i === 0 ? 1 : 0
  );
  queryInput.value = JSON.stringify(vector);
});

insertForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(insertStatus, "");

  try {
    const id = Number(insertId.value);
    if (!Number.isInteger(id) || id < 0) {
      throw new Error("Vector ID must be a non-negative integer.");
    }

    const vector = parseVector(insertVector.value.trim());

    const data = await apiFetch("/insert", {
      method: "POST",
      body: JSON.stringify({id, vector}),
    });

    setStatus(insertStatus, data?.message || `Vector ${id} inserted.`, "ok");
    insertVector.value = "";
  } catch (error) {
    setStatus(insertStatus, error.message, "error");
  }
});

deleteForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(deleteStatus, "");

  try {
    const id = Number(deleteId.value);
    if (!Number.isInteger(id) || id < 0) {
      throw new Error("Vector ID must be a non-negative integer.");
    }

    const data = await apiFetch(`/delete/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });

    setStatus(deleteStatus, data?.message || `Vector ${id} deleted.`, "ok");
  } catch (error) {
    setStatus(deleteStatus, error.message, "error");
  }
});

checkHealth();