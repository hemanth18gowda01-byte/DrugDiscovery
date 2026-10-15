const MAX_FILE_BYTES = 1024 * 1024 * 1024;
const MAX_ROWS = 5000;
const BATCH_SIZE = 100;
const API_BATCH_SIZE = 100;
const REQUIRED_COLUMNS = ["#Organism group", "AMR genotypes"];

const elements = {
  apiPill: document.querySelector("#api-pill"),
  apiStatus: document.querySelector("#api-status-label"),
  modelVersion: document.querySelector("#model-version"),
  antibiotic: document.querySelector("#antibiotic-select"),
  manualOrganism: document.querySelector("#manual-organism"),
  manualGenotypes: document.querySelector("#manual-genotypes"),
  manualAnalyze: document.querySelector("#manual-analyze-button"),
  fileInput: document.querySelector("#file-input"),
  dropzone: document.querySelector("#dropzone"),
  dropTitle: document.querySelector("#drop-title"),
  dropSubtitle: document.querySelector("#drop-subtitle"),
  fileMeta: document.querySelector("#drop-file-meta"),
  analyze: document.querySelector("#analyze-button"),
  buttonLabel: document.querySelector(".button-label"),
  progressWrap: document.querySelector("#progress-wrap"),
  progressLabel: document.querySelector("#progress-label"),
  progressPercent: document.querySelector("#progress-percent"),
  progressBar: document.querySelector("#progress-bar"),
  formError: document.querySelector("#form-error"),
  emptyState: document.querySelector("#empty-state"),
  reportContent: document.querySelector("#report-content"),
  resultsCount: document.querySelector("#results-count"),
  summary: document.querySelector("#result-summary"),
  warningBox: document.querySelector("#warning-box"),
  warningList: document.querySelector("#warning-list"),
  tableCount: document.querySelector("#table-count"),
  resultsBody: document.querySelector("#results-body"),
  jsonOutput: document.querySelector("#json-output"),
  toast: document.querySelector("#toast"),
};

let selectedFile = null;
let apiOnline = false;
let toastTimer;
const Papa = window.Papa;

function setApiStatus(status, label) {
  elements.apiPill.classList.remove("is-online", "is-offline");
  if (status === "online") {
    elements.apiPill.classList.add("is-online");
    elements.modelVersion.textContent = label || "MODEL ONLINE";
  } else if (status === "offline") {
    elements.apiPill.classList.add("is-offline");
    elements.modelVersion.textContent = "MODEL OFFLINE";
  } else {
    elements.modelVersion.textContent = "MODEL CONNECTING";
  }
  elements.apiStatus.textContent =
    status === "online" ? "API online" : status === "offline" ? "API offline" : "Checking API";
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      ...options,
      headers: {
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    throw new Error(`Could not reach the prediction API: ${error.message}`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    throw new Error(`API returned an unexpected response (${response.status}).`);
  }
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `API request failed (${response.status}).`);
  }
  return payload;
}

async function checkApi() {
  try {
    const health = await requestJson("/api/health");
    if (health.status !== "ok" || health.model !== "loaded") {
      throw new Error("The API responded, but the trained model is unavailable.");
    }
    apiOnline = true;
    setApiStatus("online", `${health.feature_count.toLocaleString()} FEATURES`);
    const options = await requestJson("/api/options");
    populateAntibiotics(options.antibiotics);
  } catch (error) {
    apiOnline = false;
    setApiStatus("offline");
    elements.antibiotic.innerHTML = '<option value="">API unavailable</option>';
    elements.antibiotic.disabled = true;
    elements.analyze.disabled = true;
    elements.manualAnalyze.disabled = true;
    if (!elements.formError.hidden) {
      showError(error.message);
    }
  }
}

function populateAntibiotics(antibiotics) {
  if (!Array.isArray(antibiotics) || antibiotics.length === 0) {
    throw new Error("The model did not return any trained antibiotic options.");
  }
  const fragment = document.createDocumentFragment();
  for (const antibiotic of antibiotics) {
    const option = document.createElement("option");
    option.value = antibiotic;
    option.textContent = antibiotic;
    fragment.append(option);
  }
  elements.antibiotic.replaceChildren(fragment);
  elements.antibiotic.disabled = false;
  // Use a trained default so an already-entered isolate or selected file can
  // be submitted immediately. The user can still choose any model antibiotic.
  elements.antibiotic.value = antibiotics.includes("ciprofloxacin")
    ? "ciprofloxacin"
    : antibiotics[0];
  updateAnalyzeButton();
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setFile(file) {
  clearError();
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showError("Choose a CSV export from NCBI isolates.csv.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError("The selected file is larger than the 1 GB upload limit.");
    return;
  }
  selectedFile = file;
  elements.dropTitle.textContent = file.name;
  elements.dropSubtitle.textContent = "Click to choose a different file";
  elements.fileMeta.textContent = `${formatBytes(file.size)} · CSV`;
  elements.dropzone.classList.add("has-file");
  updateAnalyzeButton();
}

function updateAnalyzeButton() {
  elements.analyze.disabled =
    !apiOnline || !selectedFile || !elements.antibiotic.value;
  elements.manualAnalyze.disabled =
    !apiOnline || !elements.antibiotic.value || !elements.manualOrganism.value.trim();
}

function showError(message) {
  elements.formError.textContent = message;
  elements.formError.hidden = false;
  showToast(message);
}

function clearError() {
  elements.formError.textContent = "";
  elements.formError.hidden = true;
  elements.toast.classList.remove("is-visible");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.classList.remove("is-visible");
  }, 4800);
}

function setProgress(done, total, label) {
  const percentage = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  elements.progressLabel.textContent = label;
  elements.progressPercent.textContent = `${percentage}%`;
  elements.progressBar.style.width = `${percentage}%`;
}

function collectColumns(row) {
  return Object.keys(row || {}).map((column) => column.trim());
}

function checkCsvColumns(headers) {
  const normalized = new Set(headers.map((header) => header.trim()));
  const missing = REQUIRED_COLUMNS.filter((column) => !normalized.has(column));
  if (missing.length) {
    throw new Error(`CSV is missing required columns: ${missing.join(", ")}.`);
  }
  if (!normalized.has("BioSample")) {
    showToast("No BioSample column found; result rows will use row numbers.");
  }
}

async function sendPredictionBatch(antibiotic, samples) {
  return requestJson("/api/predict", {
    method: "POST",
    body: JSON.stringify({ antibiotic, samples }),
  });
}

function updateReport(results, summary, allWarnings, sourceRows, antibiotic) {
  elements.emptyState.hidden = true;
  elements.reportContent.hidden = false;
  elements.resultsCount.textContent =
    `${antibiotic.toUpperCase()} · ${results.length.toLocaleString()} READ`;
  const total = Math.max(1, summary.resistant + summary.susceptible);
  const resistancePercent = (summary.resistant / total) * 100;
  const meanResistanceScore =
    results.reduce((sum, result) => sum + (result.probabilities.R || 0), 0) /
    Math.max(results.length, 1);
  elements.summary.innerHTML = "";
  const cards = [
    { label: "ISOLATES READ", value: results.length.toLocaleString(), style: "" },
    {
      label: "PREDICTED RESISTANT",
      value: `${summary.resistant.toLocaleString()} · ${resistancePercent.toFixed(1)}%`,
      style: "is-resistant",
    },
    {
      label: "MEAN R SCORE",
      value: `${(meanResistanceScore * 100).toFixed(1)}%`,
      style: "is-susceptible",
    },
  ];
  for (const card of cards) {
    const element = document.createElement("div");
    element.className = `summary-card ${card.style}`;
    const label = document.createElement("div");
    label.className = "summary-label";
    label.textContent = card.label;
    const value = document.createElement("div");
    value.className = "summary-value";
    value.textContent = card.value;
    element.append(label, value);
    elements.summary.append(element);
  }

  elements.warningList.replaceChildren();
  const warnings = [...allWarnings];
  if (sourceRows > results.length) {
    warnings.push(
      `This preview analyzed ${results.length.toLocaleString()} of ${sourceRows.toLocaleString()} readable rows.`,
    );
  }
  if (results.length > 200) {
    warnings.push(
      `The table displays the first 200 predictions; summary counts include all ${results.length.toLocaleString()} analyzed isolates.`,
    );
  }
  const uniqueWarnings = [...new Set(warnings)];
  if (uniqueWarnings.length) {
    for (const warning of uniqueWarnings.slice(0, 8)) {
      const item = document.createElement("li");
      item.textContent = warning;
      elements.warningList.append(item);
    }
    if (uniqueWarnings.length > 8) {
      const item = document.createElement("li");
      item.textContent = `${uniqueWarnings.length - 8} additional warning types were recorded.`;
      elements.warningList.append(item);
    }
    elements.warningBox.hidden = false;
  } else {
    elements.warningBox.hidden = true;
  }

  elements.tableCount.textContent = `${results.length.toLocaleString()} prediction${results.length === 1 ? "" : "s"}`;
  elements.resultsBody.replaceChildren();
  const tableFragment = document.createDocumentFragment();
  for (const result of results.slice(0, 200)) {
    const row = document.createElement("tr");
    const id = document.createElement("td");
    id.textContent = result.id || "Unknown";
    id.title = result.id || "";
    const call = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `prediction-tag ${result.prediction === "R" ? "is-resistant" : "is-susceptible"}`;
    badge.textContent = result.prediction;
    call.append(badge);
    const score = document.createElement("td");
    score.textContent = `${((result.probabilities.R || 0) * 100).toFixed(1)}%`;
    const note = document.createElement("td");
    note.className = result.warnings.length ? "table-warning" : "";
    note.textContent = result.warnings.length ? "Check warnings" : "—";
    note.title = result.warnings.join("\n");
    row.append(id, call, score, note);
    tableFragment.append(row);
  }
  elements.resultsBody.append(tableFragment);
  const jsonRows = results.map(({ id, warnings, ...prediction }) => prediction);
  elements.jsonOutput.textContent = JSON.stringify(
    jsonRows.length === 1 ? jsonRows[0] : jsonRows,
    null,
    2,
  );
}

async function analyzeFile() {
  clearError();
  if (!apiOnline) {
    showError("Prediction API is offline. Start the local server and try again.");
    return;
  }
  if (!selectedFile) {
    showError("Choose an NCBI isolate CSV first.");
    return;
  }
  const antibiotic = elements.antibiotic.value;
  if (!antibiotic) {
    showError("Choose an antibiotic to predict.");
    return;
  }

  elements.analyze.disabled = true;
  elements.buttonLabel.textContent = "Analyzing…";
  elements.progressWrap.hidden = false;
  setProgress(0, selectedFile.size, "Reading isolate CSV");
  const resultRows = [];
  const globalWarnings = new Set();
  let pendingSamples = [];
  let parsedRows = 0;
  let readableRows = 0;
  let skippedRowsWithoutOrganism = 0;
  let activeParser;
  let batchError = null;
  let headerChecked = false;
  let reachedRowLimit = false;

  const flushBatch = async () => {
    if (!pendingSamples.length) return;
    const samples = pendingSamples;
    pendingSamples = [];
    try {
      const response = await sendPredictionBatch(antibiotic, samples);
      resultRows.push(...response.results);
      if (response.note) globalWarnings.add(response.note);
      for (const result of response.results) {
        for (const warning of result.warnings) globalWarnings.add(warning);
      }
      setProgress(
        resultRows.length,
        MAX_ROWS,
        `Predicted ${resultRows.length.toLocaleString()} isolates`,
      );
    } catch (error) {
      batchError = error;
      if (activeParser) activeParser.abort();
      throw error;
    }
  };

  try {
    await new Promise((resolve, reject) => {
      let parseSettled = false;
      const settleReject = (error) => {
        if (parseSettled) return;
        parseSettled = true;
        reject(error);
      };
      Papa.parse(selectedFile, {
        header: true,
        skipEmptyLines: "greedy",
        step: (stepResult, parser) => {
          activeParser = parser;
          parser.pause();
          const row = stepResult.data;
          try {
            if (!headerChecked) {
              checkCsvColumns(collectColumns(row));
              headerChecked = true;
            }
            parsedRows += 1;
            if (parsedRows > MAX_ROWS) {
              reachedRowLimit = true;
              parser.abort();
              return;
            }
            const organism = String(row["#Organism group"] || "").trim();
            const genotypeText = String(row["AMR genotypes"] || "").trim();
            if (!organism) {
              skippedRowsWithoutOrganism += 1;
              parser.resume();
              return;
            }
            readableRows += 1;
            pendingSamples.push({
              id: String(row.BioSample || row.Strain || `row-${parsedRows}`).trim(),
              organism,
              genotypes: genotypeText,
            });
            setProgress(parsedRows, MAX_ROWS, `Reading row ${parsedRows.toLocaleString()}`);
            if (pendingSamples.length >= API_BATCH_SIZE) {
              flushBatch()
                .then(() => parser.resume())
                .catch(settleReject);
            } else {
              parser.resume();
            }
          } catch (error) {
            settleReject(error);
          }
        },
        complete: async () => {
          if (parseSettled) return;
          parseSettled = true;
          try {
            await flushBatch();
            if (!headerChecked) {
              throw new Error("The selected CSV is empty or has no header row.");
            }
            if (!readableRows) {
              throw new Error("No rows with an organism name were found.");
            }
            resolve();
          } catch (error) {
            reject(error);
          }
        },
        error: (error) => settleReject(new Error(`Could not read CSV: ${error.message}`)),
      });
    });
    if (batchError) throw batchError;
    if (reachedRowLimit) {
      globalWarnings.add(
        `Reached the ${MAX_ROWS.toLocaleString()}-row preview limit; remaining CSV rows were not analyzed.`,
      );
    }
    if (skippedRowsWithoutOrganism) {
      globalWarnings.add(
        `${skippedRowsWithoutOrganism.toLocaleString()} CSV rows without an organism value were skipped.`,
      );
    }
    if (!resultRows.length) {
      throw new Error("No eligible isolate rows were found in the selected CSV.");
    }
    const summary = {
      resistant: resultRows.filter((result) => result.prediction === "R").length,
      susceptible: resultRows.filter((result) => result.prediction === "S").length,
    };
    updateReport(resultRows, summary, [...globalWarnings], readableRows, antibiotic);
    elements.progressLabel.textContent = `Complete · ${resultRows.length.toLocaleString()} isolates`;
    elements.progressPercent.textContent = "100%";
    elements.progressBar.style.width = "100%";
    document.querySelector("#results-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error.message || "Analysis failed. Check the file and API, then retry.");
  } finally {
    elements.analyze.disabled = false;
    elements.buttonLabel.textContent = "Analyze isolates";
    updateAnalyzeButton();
  }
}

async function analyzeManual() {
  clearError();
  const antibiotic = elements.antibiotic.value;
  const organism = elements.manualOrganism.value.trim();
  const genotypes = elements.manualGenotypes.value.trim();
  if (!apiOnline) return showError("Prediction API is offline. Start the local server and try again.");
  if (!antibiotic) return showError("Choose an antibiotic to predict.");
  if (!organism) return showError("Enter an organism group for the isolate.");

  elements.manualAnalyze.disabled = true;
  const originalLabel = elements.manualAnalyze.innerHTML;
  elements.manualAnalyze.textContent = "Predicting…";
  try {
    const response = await sendPredictionBatch(antibiotic, [{
      id: "manual isolate",
      organism,
      genotypes,
    }]);
    const warnings = [response.note, ...response.results.flatMap((result) => result.warnings)].filter(Boolean);
    updateReport(response.results, response.summary, warnings, 1, antibiotic);
    document.querySelector("#results-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error.message || "Prediction failed. Check the input and try again.");
  } finally {
    elements.manualAnalyze.innerHTML = originalLabel;
    updateAnalyzeButton();
  }
}

elements.fileInput.addEventListener("change", (event) => {
  setFile(event.target.files[0]);
});
elements.antibiotic.addEventListener("change", updateAnalyzeButton);
elements.manualOrganism.addEventListener("input", updateAnalyzeButton);
elements.manualAnalyze.addEventListener("click", analyzeManual);
elements.analyze.addEventListener("click", analyzeFile);
elements.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});
for (const eventName of ["dragenter", "dragover"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("is-dragging");
  });
}
elements.dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  setFile(file);
});

setApiStatus("checking");
checkApi();
window.setInterval(checkApi, 30000);
