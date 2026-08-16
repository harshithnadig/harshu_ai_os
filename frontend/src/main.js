// Harshu AI OS — Neural Mission Control Frontend Runtime
const API_BASE_URL = "http://127.0.0.1:8000";

const form = document.getElementById("ask-form");
const question = document.getElementById("question");
const submitButton = document.getElementById("submit-button");
const modeButtons = document.querySelectorAll(".mode-btn:not(.disabled)");
const modeIndicator = document.getElementById("mode-indicator");
const routeIndicator = document.getElementById("route-indicator");
const liveRouteNode = document.getElementById("live-route-node");

const result = document.getElementById("result");
const answer = document.getElementById("answer");
const metadata = document.getElementById("metadata");
const errorPanel = document.getElementById("error");
const errorMessage = document.getElementById("error-message");

// Normal Mode: Web Search Tool Elements
const toolExecution = document.getElementById("tool-execution");
const toolQuery = document.getElementById("tool-query");
const toolSources = document.getElementById("tool-sources");

// RAG Mode Elements
const ragEvidence = document.getElementById("rag-evidence");
const ragStatusBanner = document.getElementById("rag-status-banner");
const timeRetrieval = document.getElementById("time-retrieval");
const timeJudge = document.getElementById("time-judge");
const timeGeneration = document.getElementById("time-generation");
const timeTotal = document.getElementById("time-total");
const citations = document.getElementById("citations");
const ragJudgeReason = document.getElementById("rag-judge-reason");
const ragChunkCount = document.getElementById("rag-chunk-count");
const ragIds = document.getElementById("rag-ids");
const ragDistances = document.getElementById("rag-distances");
const ragMetadata = document.getElementById("rag-metadata");
const context = document.getElementById("context");

// Inspector Drawer Elements
const inspectorDrawer = document.getElementById("inspector-drawer");
const inspectorToggleBtn = document.getElementById("inspector-toggle-btn");
const inspectorCloseBtn = document.getElementById("inspector-close-btn");
const inspectorBackdrop = document.getElementById("inspector-backdrop");

let activeMode = "ask";
let latestRequestId = 0;

// Mode Descriptions
const modeLabels = {
  ask: "MODE: NORMAL DIRECT",
  rag: "MODE: GROUNDED RAG",
};

// Mode Switcher Handlers
modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    setMode(btn.dataset.mode);
  });
});

function setMode(mode) {
  if (!mode || mode === activeMode) return;
  activeMode = mode;

  modeButtons.forEach((btn) => {
    const isSelected = btn.dataset.mode === mode;
    btn.classList.toggle("active", isSelected);
    btn.setAttribute("aria-checked", String(isSelected));
  });

  if (modeIndicator) {
    modeIndicator.textContent = modeLabels[mode] || "MODE: UNKNOWN";
  }

  if (liveRouteNode) {
    liveRouteNode.textContent = mode === "rag" ? "rag-pipeline" : "harshu-general";
  }
}

// Inspector Drawer Handlers
function toggleInspector(show) {
  const isOpen = typeof show === "boolean" ? show : !inspectorDrawer.classList.contains("open");
  inspectorDrawer.classList.toggle("open", isOpen);
  inspectorDrawer.setAttribute("aria-hidden", String(!isOpen));
  if (inspectorToggleBtn) {
    inspectorToggleBtn.setAttribute("aria-expanded", String(isOpen));
  }
}

if (inspectorToggleBtn) {
  inspectorToggleBtn.addEventListener("click", () => toggleInspector());
}
if (inspectorCloseBtn) {
  inspectorCloseBtn.addEventListener("click", () => toggleInspector(false));
}
if (inspectorBackdrop) {
  inspectorBackdrop.addEventListener("click", () => toggleInspector(false));
}

// Global Keyboard Shortcuts (⌥I for Inspector, Ctrl+K for Search Focus)
window.addEventListener("keydown", (e) => {
  if (e.altKey && (e.key === "i" || e.key === "I" || e.code === "KeyI")) {
    e.preventDefault();
    toggleInspector();
  }
  if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
    e.preventDefault();
    question.focus();
  }
  if (e.key === "Escape" && inspectorDrawer.classList.contains("open")) {
    toggleInspector(false);
  }
});

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  question.disabled = isLoading;
  modeButtons.forEach((btn) => {
    btn.disabled = isLoading;
  });

  submitButton.innerHTML = isLoading
    ? '<span class="spinner" aria-hidden="true"></span><span>Thinking...</span>'
    : '<span>Execute</span><span class="btn-arrow" aria-hidden="true">→</span>';
}

function clearOutput() {
  result.hidden = true;
  clearError();

  toolExecution.hidden = true;
  toolSources.replaceChildren();
  toolQuery.textContent = "";

  ragEvidence.hidden = true;
  ragStatusBanner.className = "grounding-banner";
  ragStatusBanner.replaceChildren();
  citations.replaceChildren();
  context.textContent = "";
  ragMetadata.textContent = "";
  ragIds.textContent = "";
  ragDistances.textContent = "";
  ragJudgeReason.textContent = "";
}

function clearError() {
  errorPanel.hidden = true;
  errorMessage.textContent = "";
}

function addMeta(label, value, fragment) {
  const chip = document.createElement("span");
  chip.className = "meta-chip";
  chip.textContent = `${label}: ${value}`;
  if (fragment) {
    fragment.appendChild(chip);
  } else {
    metadata.append(chip);
  }
}

function renderWebSource(source, index, fragment) {
  const row = document.createElement("div");
  row.className = "web-source-row";

  const num = document.createElement("span");
  num.className = "source-num";
  num.textContent = `${index + 1}.`;

  const anchor = document.createElement("a");
  anchor.className = "source-anchor";
  anchor.textContent = source.title || source.url || "External Web Source";
  anchor.href = source.url || "#";
  anchor.target = "_blank";
  anchor.rel = "noopener noreferrer";

  const host = document.createElement("span");
  host.className = "source-host";
  try {
    const parsed = new URL(source.url);
    host.textContent = parsed.hostname.replace(/^www\./, "");
  } catch {
    host.textContent = source.url || "";
  }

  row.append(num, anchor, host);
  fragment.appendChild(row);
}

function renderCitation(citation, index, fragment) {
  const box = document.createElement("article");
  box.className = "citation-box";

  const title = document.createElement("h3");
  title.textContent = citation.source || "Knowledge Base Document";

  const details = document.createElement("dl");
  const fields = [
    ["Chunk ID", citation.chunk_id],
    ["Index", citation.chunk_index ?? "—"],
    ["Distance", Number(citation.distance).toFixed(4)],
  ];

  fields.forEach(([term, val]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = val;
    details.append(dt, dd);
  });

  box.setAttribute("aria-label", `Source ${index + 1}: ${title.textContent}`);
  box.append(title, details);
  fragment.appendChild(box);
}

function friendlyError(response, body) {
  if (response.status === 503) {
    return body?.detail ?? "The AI service is temporarily unavailable. Please verify API keys and network.";
  }
  if (response.status >= 500) {
    return "The Harshu AI OS backend encountered an unexpected error. Please check server logs.";
  }
  return body?.detail ?? "Please check your prompt and try again.";
}

// Form Submission Handler
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = question.value.trim();
  if (!prompt) return;

  const requestId = ++latestRequestId;
  clearOutput();
  setLoading(true);

  try {
    const endpoint = activeMode === "rag" ? "/ask/rag" : "/ask";
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: prompt }),
    });

    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error(friendlyError(response, body));

    clearError();
    answer.textContent = body.answer;

    // Build metadata chips with DocumentFragment for performance
    const metaFragment = document.createDocumentFragment();

    if (activeMode === "ask") {
      addMeta("Mode", "Normal", metaFragment);
      addMeta("Model", body.model, metaFragment);
      addMeta("Complexity", body.complexity, metaFragment);

      if (body.tool_used) {
        addMeta("Tool", "Web search used", metaFragment);
        toolQuery.textContent = body.tool_query || "(empty query)";
        toolSources.replaceChildren();

        const sourcesList = body.tool_sources ?? [];
        if (sourcesList.length > 0) {
          const sourcesFragment = document.createDocumentFragment();
          sourcesList.forEach((src, idx) => renderWebSource(src, idx, sourcesFragment));
          toolSources.appendChild(sourcesFragment);
        } else {
          const empty = document.createElement("p");
          empty.className = "empty-note";
          empty.textContent = "No external URLs returned by search.";
          toolSources.appendChild(empty);
        }
        toolExecution.hidden = false;
      } else {
        addMeta("Tool", "No tool used", metaFragment);
        toolExecution.hidden = true;
      }
      ragEvidence.hidden = true;
    } else {
      // RAG Mode
      addMeta("Mode", "RAG", metaFragment);
      addMeta("Model", body.model, metaFragment);
      addMeta("Complexity", body.complexity, metaFragment);
      addMeta("Grounding", body.abstained ? "Abstained" : "Passed", metaFragment);

      // Grounding decision banner
      ragStatusBanner.className = `grounding-banner ${body.abstained ? "banner-abstained" : "banner-passed"}`;
      const statusIcon = document.createElement("span");
      statusIcon.className = "banner-icon";
      statusIcon.textContent = body.abstained ? "⚠" : "✓";

      const statusText = document.createElement("div");
      const statusTitle = document.createElement("strong");
      statusTitle.textContent = body.abstained
        ? "Grounding Verdict: Abstained"
        : "Grounding Verdict: Passed (Verified Context)";
      const statusDesc = document.createElement("p");
      statusDesc.textContent = body.abstained
        ? `Reason: ${body.abstention_reason || "Insufficient supporting knowledge in vector store."}`
        : "Answer synthesized strictly from retrieved and verified knowledge chunks.";

      statusText.append(statusTitle, statusDesc);
      ragStatusBanner.append(statusIcon, statusText);

      // Timings
      timeRetrieval.textContent = `${Number(body.retrieval_ms || 0).toFixed(1)} ms`;
      timeJudge.textContent = `${Number(body.judge_ms || 0).toFixed(1)} ms`;
      timeGeneration.textContent = `${Number(body.generation_ms || 0).toFixed(1)} ms`;
      timeTotal.textContent = `${Number(body.total_ms || 0).toFixed(1)} ms`;

      // Citations
      citations.replaceChildren();
      const citationsList = body.citations ?? [];
      if (citationsList.length > 0) {
        const citationsFragment = document.createDocumentFragment();
        citationsList.forEach((cit, idx) => renderCitation(cit, idx, citationsFragment));
        citations.appendChild(citationsFragment);
      } else {
        const noCitations = document.createElement("p");
        noCitations.className = "empty-note";
        noCitations.textContent = body.abstained
          ? "No supporting citations (pipeline abstained due to threshold distance)."
          : "No citations recorded.";
        citations.appendChild(noCitations);
      }

      // Debug Details
      ragJudgeReason.textContent = body.judge_reason || "None recorded";
      ragChunkCount.textContent = String(body.ids?.length ?? 0);
      ragIds.textContent = (body.ids ?? []).join(", ") || "None";
      ragDistances.textContent = (body.distances ?? []).map((d) => Number(d).toFixed(4)).join(", ") || "None";
      ragMetadata.textContent = JSON.stringify(body.metadatas ?? [], null, 2);
      context.textContent = body.context ?? "No retrieved context returned.";

      toolExecution.hidden = true;
      ragEvidence.hidden = false;
    }

    metadata.replaceChildren(metaFragment);
    result.hidden = false;

    // Smoothly scroll result into viewport
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    if (requestId !== latestRequestId) return;
    errorMessage.textContent =
      err instanceof TypeError
        ? "Unable to connect to the backend server. Verify that FastAPI is running at http://127.0.0.1:8000."
        : err.message;
    errorPanel.hidden = false;
  } finally {
    setLoading(false);
  }
});
