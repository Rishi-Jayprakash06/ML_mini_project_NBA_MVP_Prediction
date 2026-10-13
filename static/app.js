/**
 * NBA MVP PREDICTOR - FRONTEND CONTROLLER
 * Handles interactive feature synchronization, preset loading, API calls, and analytics rendering.
 */

const FEATURES = [
  "g",
  "mp_per_game",
  "fg_percent",
  "x3p_percent",
  "ft_percent",
  "trb_per_game",
  "ast_per_game",
  "stl_per_game",
  "blk_per_game",
  "pts_per_game"
];

let globalMetadata = null;
let globalPresets = null;
let debounceTimer = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  setupInputListeners();
  await loadMetadata();
  window.runPrediction();
});

// Setup two-way data binding between sliders and numeric inputs
function setupInputListeners() {
  FEATURES.forEach(feat => {
    const numInput = document.getElementById(`input-${feat}`);
    const rangeInput = document.getElementById(`range-${feat}`);

    if (numInput && rangeInput) {
      // Slider moves -> update number input
      rangeInput.addEventListener("input", (e) => {
        numInput.value = e.target.value;
        onFeatureValueChanged();
      });

      // Number input typed -> update slider
      numInput.addEventListener("input", (e) => {
        rangeInput.value = e.target.value;
        onFeatureValueChanged();
      });
    }
  });

  // Algorithm or Mode selection changes
  const modelSelector = document.getElementById("model-selector");
  if (modelSelector) {
    modelSelector.addEventListener("change", () => window.runPrediction());
  }

  const modeSelector = document.getElementById("mode-selector");
  if (modeSelector) {
    modeSelector.addEventListener("change", () => window.runPrediction());
  }
}

// Debounce live predictions
function onFeatureValueChanged() {
  const liveToggle = document.getElementById("live-predict-toggle");
  if (liveToggle && liveToggle.checked) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      window.runPrediction();
    }, 180);
  }
}

// Fetch metadata and presets from Flask API
async function loadMetadata() {
  try {
    const res = await fetch("/api/metadata");
    const data = await res.json();
    globalMetadata = data.metadata;
    globalPresets = data.presets;

    // Attach click handlers to preset buttons
    setupPresetButtons();
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

// Setup preset clicks
function setupPresetButtons() {
  const buttons = document.querySelectorAll(".preset-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      if (globalPresets && globalPresets[presetKey]) {
        applyPreset(globalPresets[presetKey]);

        // Toggle active button class
        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
      }
    });
  });
}

// Apply selected preset stats to inputs
function applyPreset(preset) {
  const stats = preset.stats;
  for (const [feat, val] of Object.entries(stats)) {
    const numInput = document.getElementById(`input-${feat}`);
    const rangeInput = document.getElementById(`range-${feat}`);
    if (numInput) numInput.value = val;
    if (rangeInput) rangeInput.value = val;
  }
  window.runPrediction();
}

// Gather features from inputs
function getCurrentFeatures() {
  const feats = {};
  FEATURES.forEach(f => {
    const el = document.getElementById(`input-${f}`);
    feats[f] = el ? parseFloat(el.value) || 0 : 0;
  });
  return feats;
}

// Main Prediction Execution
window.runPrediction = async function() {
  const features = getCurrentFeatures();
  const modelName = document.getElementById("model-selector")?.value || "rf";
  const mode = document.getElementById("mode-selector")?.value || "full";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features: features,
        model: modelName,
        mode: mode
      })
    });

    const data = await res.json();
    if (data.success && data.result) {
      renderResults(data.result);
    } else {
      console.error("Prediction error:", data.error);
    }
  } catch (err) {
    console.error("Network or inference error:", err);
  }
};

// Render results onto UI
function renderResults(result) {
  const verdictCard = document.getElementById("verdict-card");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictSubtext = document.getElementById("verdict-subtext");
  const verdictModelLabel = document.getElementById("verdict-model-label");
  const scoreVal = document.getElementById("score-val");
  const scoreBar = document.getElementById("score-bar");
  const alertBox = document.getElementById("eligibility-alert");
  const alertMsg = document.getElementById("alert-message");

  // 1. Primary Verdict Banner
  if (result.is_mvp) {
    verdictCard.classList.remove("non-mvp-state");
    verdictIcon.textContent = "🏆";
    verdictTitle.textContent = "PREDICTED MVP: YES";
    verdictTitle.style.color = "#ffffff";
    verdictSubtext.textContent = "Player exhibits historic Most Valuable Player caliber statistical dominance.";
  } else {
    verdictCard.classList.add("non-mvp-state");
    verdictIcon.textContent = "❌";
    verdictTitle.textContent = "PREDICTED MVP: NO";
    verdictTitle.style.color = "#fca5a5";
    verdictSubtext.textContent = "Statistical production does not reach the historical MVP winner benchmark.";
  }

  if (verdictModelLabel) {
    verdictModelLabel.textContent = `${result.model_title.toUpperCase()} CLASSIFICATION`;
  }

  // Score Progress Meter
  const probPct = result.prob_percentage;
  if (scoreVal) scoreVal.textContent = `${probPct.toFixed(1)}%`;
  if (scoreBar) scoreBar.style.width = `${Math.min(100, Math.max(0, probPct))}%`;

  // Eligibility Warning
  if (alertBox && alertMsg) {
    if (result.eligibility_warning) {
      alertMsg.textContent = result.eligibility_warning;
      alertBox.style.display = "flex";
    } else {
      alertBox.style.display = "none";
    }
  }

  // 2. Models Consensus Boxes
  renderConsensusBoxes(result.models_consensus);

  // 3. Benchmarks Comparison
  renderBenchmarks(result.benchmarks);

  // 4. Feature Importance
  renderImportance(result.feature_importance);

  // 5. Automated Insights
  renderInsights(result.insights);
}

// Render Consensus Grid
function renderConsensusBoxes(consensus) {
  if (!consensus) return;

  // RF
  const rfPill = document.getElementById("pill-rf");
  const rfProb = document.getElementById("prob-rf");
  const rfTier = document.getElementById("tier-rf");
  if (rfPill && consensus.rf) {
    rfPill.textContent = consensus.rf.is_mvp ? "MVP: YES" : "MVP: NO";
    rfPill.className = consensus.rf.is_mvp ? "model-verdict-pill mvp-pill" : "model-verdict-pill non-mvp-pill";
    if (rfProb) rfProb.textContent = `${consensus.rf.prob_pct.toFixed(1)}%`;
    if (rfTier) rfTier.textContent = consensus.rf.rank_grade;
  }

  // LogReg
  const logregPill = document.getElementById("pill-logreg");
  const logregProb = document.getElementById("prob-logreg");
  const logregTier = document.getElementById("tier-logreg");
  if (logregPill && consensus.logreg) {
    logregPill.textContent = consensus.logreg.is_mvp ? "MVP: YES" : "MVP: NO";
    logregPill.className = consensus.logreg.is_mvp ? "model-verdict-pill mvp-pill" : "model-verdict-pill non-mvp-pill";
    if (logregProb) logregProb.textContent = `${consensus.logreg.prob_pct.toFixed(1)}%`;
    if (logregTier) logregTier.textContent = consensus.logreg.rank_grade;
  }

  // Naive Bayes
  const nbPill = document.getElementById("pill-nb");
  const nbProb = document.getElementById("prob-nb");
  const nbTier = document.getElementById("tier-nb");
  if (nbPill && consensus.nb) {
    nbPill.textContent = consensus.nb.is_mvp ? "MVP: YES" : "MVP: NO";
    nbPill.className = consensus.nb.is_mvp ? "model-verdict-pill mvp-pill" : "model-verdict-pill non-mvp-pill";
    if (nbProb) nbProb.textContent = `${consensus.nb.prob_pct.toFixed(1)}%`;
    if (nbTier) nbTier.textContent = consensus.nb.rank_grade;
  }
}

// Render Comparison vs MVP Averages
function renderBenchmarks(benchmarks) {
  const container = document.getElementById("benchmarks-container");
  if (!container || !benchmarks) return;

  container.innerHTML = "";

  const order = ["pts_per_game", "mp_per_game", "g", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game", "fg_percent", "x3p_percent", "ft_percent"];

  order.forEach(f => {
    const item = benchmarks[f];
    if (!item) return;

    let deltaClass = "delta-near";
    let deltaSign = item.diff_pct > 0 ? "+" : "";
    if (item.status === "above") deltaClass = "delta-above";
    if (item.status === "below") deltaClass = "delta-below";

    const card = document.createElement("div");
    card.className = "benchmark-card";
    card.innerHTML = `
      <div class="bench-header">
        <span>${item.short}</span>
        <span class="bench-delta ${deltaClass}">${deltaSign}${item.diff_pct}%</span>
      </div>
      <div class="bench-val">${item.user_val}</div>
      <div class="bench-avg">MVP Avg: ${item.mvp_avg}</div>
    `;
    container.appendChild(card);
  });
}

// Render Feature Importance
function renderImportance(importanceList) {
  const container = document.getElementById("importance-container");
  if (!container || !importanceList) return;

  container.innerHTML = "";

  const maxImp = Math.max(...importanceList.map(x => x.importance), 1);

  importanceList.slice(0, 6).forEach(item => {
    const row = document.createElement("div");
    row.className = "imp-row";
    const widthPct = Math.min(100, (item.importance / maxImp) * 100);

    row.innerHTML = `
      <div class="imp-label" title="${item.name}">${item.name}</div>
      <div class="imp-bar-track">
        <div class="imp-bar-fill" style="width: ${widthPct}%"></div>
      </div>
      <div class="imp-pct">${item.importance.toFixed(1)}%</div>
    `;
    container.appendChild(row);
  });
}

// Render Insights
function renderInsights(insights) {
  const container = document.getElementById("insights-container");
  if (!container || !insights) return;

  container.innerHTML = "";
  insights.forEach(text => {
    const li = document.createElement("li");
    li.textContent = text;
    container.appendChild(li);
  });
}

// Reset button
window.resetToExample = function() {
  if (globalMetadata) {
    FEATURES.forEach(f => {
      const ex = globalMetadata[f]?.example || 0;
      const numInput = document.getElementById(`input-${f}`);
      const rangeInput = document.getElementById(`range-${f}`);
      if (numInput) numInput.value = ex;
      if (rangeInput) rangeInput.value = ex;
    });

    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
    window.runPrediction();
  }
};
