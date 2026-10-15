/* React client for the local CARD prediction API. */
const { createElement: h, useEffect, useMemo, useRef, useState } = React;

const MAX_ROWS = 5000;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.body ? { "Content-Type": "application/json" } : {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
  return body;
}

function PredictionCard({ result, index }) {
  const resistant = result.prediction === "R";
  return h("article", { className: "prediction-card" },
    h("div", { className: "prediction-card-head" },
      h("span", { className: "prediction-number" }, `ISOLATE ${index + 1}`),
      h("span", { className: `prediction-call ${resistant ? "is-resistant" : "is-susceptible"}` }, `${result.prediction} · ${result.prediction_label}`),
    ),
    h("div", { className: "prediction-fields" },
      h("div", null, h("span", null, "ORGANISM"), h("strong", null, result.organism)),
      h("div", null, h("span", null, "ANTIBIOTIC"), h("strong", null, result.antibiotic)),
      h("div", null, h("span", null, "GENOTYPE CALLS"), h("strong", null, result.genotype_calls_supplied)),
      h("div", null, h("span", null, "RESISTANCE SCORE"), h("strong", null, `${((result.probabilities.R || 0) * 100).toFixed(1)}%`)),
      h("div", null, h("span", null, "SUSCEPTIBLE SCORE"), h("strong", null, `${((result.probabilities.S || 0) * 100).toFixed(1)}%`)),
    ),
    h("p", { className: "prediction-note" }, result.note),
    result.warnings?.length > 0 && h("ul", { className: "prediction-warnings" }, result.warnings.map((warning, warningIndex) => h("li", { key: warningIndex }, warning))),
  );
}

function App() {
  const fileRef = useRef(null);
  const [status, setStatus] = useState("Connecting");
  const [antibiotics, setAntibiotics] = useState([]);
  const [antibiotic, setAntibiotic] = useState("");
  const [organism, setOrganism] = useState("");
  const [genotypes, setGenotypes] = useState("");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);

  useEffect(() => {
    Promise.all([api("/api/health"), api("/api/options")])
      .then(([health, options]) => {
        setStatus(`${health.feature_count.toLocaleString()} features online`);
        setAntibiotics(options.antibiotics);
        setAntibiotic(options.antibiotics.includes("ciprofloxacin") ? "ciprofloxacin" : options.antibiotics[0] || "");
      })
      .catch((requestError) => { setStatus("API offline"); setError(requestError.message); });
  }, []);

  const submit = async (samples) => {
    setBusy(true); setError("");
    try { setResponse(await api("/api/predict", { method: "POST", body: JSON.stringify({ antibiotic, samples }) })); }
    catch (requestError) { setError(requestError.message); }
    finally { setBusy(false); }
  };

  const predictManual = () => {
    if (!organism.trim()) return setError("Enter an organism group first.");
    submit([{ id: "manual-isolate", organism: organism.trim(), genotypes: genotypes.trim() }]);
  };

  const predictFile = () => {
    if (!file) return setError("Choose an isolates CSV file first.");
    if (!window.Papa) return setError("CSV reader is still loading. Please try again.");
    setError(""); setBusy(true);
    window.Papa.parse(file, {
      header: true, skipEmptyLines: "greedy", complete: async ({ data, meta }) => {
        const headers = meta.fields || [];
        const required = ["#Organism group", "AMR genotypes"];
        const missing = required.filter((name) => !headers.includes(name));
        if (missing.length) { setError(`CSV is missing: ${missing.join(", ")}.`); setBusy(false); return; }
        const samples = data.slice(0, MAX_ROWS).filter((row) => String(row["#Organism group"] || "").trim()).map((row, index) => ({
          id: String(row.BioSample || row.Strain || `row-${index + 1}`),
          organism: String(row["#Organism group"]).trim(),
          genotypes: String(row["AMR genotypes"] || "").trim(),
        }));
        if (!samples.length) { setError("No rows with an organism group were found."); setBusy(false); return; }
        try { setResponse(await api("/api/predict", { method: "POST", body: JSON.stringify({ antibiotic, samples }) })); }
        catch (requestError) { setError(requestError.message); }
        finally { setBusy(false); }
      }, error: () => { setError("Could not read this CSV file."); setBusy(false); },
    });
  };

  const warnings = response ? [response.note, ...response.results.flatMap((result) => result.warnings || [])].filter(Boolean) : [];
  const disabled = busy || !antibiotic;

  return h("main", { className: "react-shell" },
    h("header", { className: "react-topbar" }, h("span", { className: "react-brand" }, "CARDINAL."), h("span", { className: "react-status" }, status)),
    h("section", { className: "react-hero" }, h("p", null, "ANTIMICROBIAL RESISTANCE / RESEARCH"), h("h1", null, "Genome signal, ", h("em", null, "mapped.")), h("p", { className: "react-subtitle" }, "A React + Three.js research interface for AMR phenotype prediction.")),
    h("section", { className: "react-grid" },
      h("article", { className: "react-panel" },
        h("p", { className: "react-kicker" }, "01 / INPUT"), h("h2", null, "Run a prediction"),
        h("label", null, "ANTIBIOTIC", h("select", { value: antibiotic, onChange: (event) => setAntibiotic(event.target.value), disabled: !antibiotics.length }, antibiotics.map((item) => h("option", { value: item, key: item }, item)))),
        h("div", { className: "react-divider" }, "MANUAL ISOLATE"),
        h("label", null, "ORGANISM GROUP", h("input", { value: organism, onChange: (event) => setOrganism(event.target.value), placeholder: "Salmonella enterica" })),
        h("label", null, "AMR GENOTYPES (OPTIONAL)", h("textarea", { value: genotypes, onChange: (event) => setGenotypes(event.target.value), placeholder: "gyrA_D87Y=POINT,parC_S80I=POINT" })),
        h("button", { className: "react-primary", onClick: predictManual, disabled: disabled || !organism.trim() }, busy ? "Predicting…" : "Predict isolate →"),
        h("div", { className: "react-divider" }, "OR CSV BATCH"),
        h("input", { ref: fileRef, type: "file", accept: ".csv,text/csv", hidden: true, onChange: (event) => setFile(event.target.files[0] || null) }),
        h("button", { className: "react-file", onClick: () => fileRef.current?.click() }, file ? `Selected: ${file.name}` : "Choose isolates.csv"),
        h("button", { className: "react-secondary", onClick: predictFile, disabled: disabled || !file }, busy ? "Analyzing…" : "Analyze CSV →"),
        error && h("p", { className: "react-error", role: "alert" }, error),
      ),
      h("article", { className: "react-panel react-results" },
        h("p", { className: "react-kicker" }, "02 / OUTPUT"), h("h2", null, "Prediction report"),
        response && h("div", { className: "react-summary" }, `${response.summary.resistant} resistant · ${response.summary.susceptible} susceptible`),
        warnings.length > 0 && h("div", { className: "react-warning" }, h("strong", null, "Model warning"), h("ul", null, warnings.slice(0, 5).map((warning, index) => h("li", { key: index }, warning)))),
        response
          ? h("div", { className: "prediction-list" }, response.results.slice(0, 100).map((result, index) => h(PredictionCard, { result, index, key: `${result.id}-${index}` })), response.results.length > 100 && h("p", { className: "prediction-limit" }, `Showing 100 of ${response.results.length} predictions.`))
          : h("div", { className: "report-empty" }, "Enter an isolate or upload a CSV to view its full prediction report."),
      ),
    ),
    h("footer", null, "Research use only — not validated for clinical treatment decisions."),
  );
}

ReactDOM.createRoot(document.querySelector("#root")).render(h(App));
