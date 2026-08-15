import "./styles.css";

// The frontend stays deliberately thin: the backend owns routing, retrieval,
// provider calls, and tool dispatch; this file renders their typed response.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const modeDescriptions = {
  ask: "Direct AI response with automatic web search tool if required",
  rag: "Retrieves indexed knowledge notes, checks sufficiency, and grounds answer",
};

const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const modeButtons = document.querySelectorAll(".mode-button");
const modeHint = document.querySelector("#mode-hint");
const result = document.querySelector("#result");
const answer = document.querySelector("#answer");
const metadata = document.querySelector("#metadata");

// Tool Execution Elements (Normal Mode)
const toolExecution = document.querySelector("#tool-execution");
const toolQuery = document.querySelector("#tool-query");
const toolSources = document.querySelector("#tool-sources");

// RAG Elements
const ragEvidence = document.querySelector("#rag-evidence");
const ragStatusBanner = document.querySelector("#rag-status-banner");
const timeRetrieval = document.querySelector("#time-retrieval");
const timeJudge = document.querySelector("#time-judge");
const timeGeneration = document.querySelector("#time-generation");
const timeTotal = document.querySelector("#time-total");
const citations = document.querySelector("#citations");
const ragJudgeReason = document.querySelector("#rag-judge-reason");
const ragChunkCount = document.querySelector("#rag-chunk-count");
const ragIds = document.querySelector("#rag-ids");
const ragDistances = document.querySelector("#rag-distances");
const ragMetadata = document.querySelector("#rag-metadata");
const context = document.querySelector("#context");

const errorPanel = document.querySelector("#error");
const errorMessage = document.querySelector("#error-message");

let activeMode = "ask";
let latestRequestId = 0;

modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

function setMode(mode) {
  activeMode = mode;
  modeButtons.forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-checked", String(selected));
  });
  modeHint.textContent = modeDescriptions[mode];
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  question.disabled = isLoading;
  modeButtons.forEach((button) => {
    button.disabled = isLoading;
  });
  submitButton.innerHTML = isLoading
    ? '<span class="spinner" aria-hidden="true"></span><span>Thinking</span>'
    : '<span>Ask Harshu</span><b aria-hidden="true">↗</b>';
}

function clearOutput() {
  result.hidden = true;
  clearError();
  toolExecution.hidden = true;
  toolSources.replaceChildren();
  toolQuery.textContent = "";

  ragEvidence.hidden = true;
  ragStatusBanner.className = "rag-status-banner";
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

function addMeta(label, value) {
  const chip = document.createElement("span");
  chip.textContent = `${label}: ${value}`;
  metadata.append(chip);
}

function renderWebSource(source, index) {
  // Use safe DOM methods for untrusted live web search data.
  const item = document.createElement("div");
  item.className = "web-source-item";

  const num = document.createElement("span");
  num.className = "source-index";
  num.textContent = `${index + 1}.`;

  const link = document.createElement("a");
  link.className = "web-source-link";
  link.textContent = source.title || source.url || "Web Source";
  link.href = source.url || "#";
  link.target = "_blank";
  link.rel = "noopener noreferrer";

  const urlPreview = document.createElement("span");
  urlPreview.className = "source-url-preview";
  try {
    const parsed = new URL(source.url);
    urlPreview.textContent = parsed.hostname;
  } catch {
    urlPreview.textContent = source.url;
  }

  item.append(num, link, urlPreview);
  toolSources.append(item);
}

function renderCitation(citation, index) {
  // Use textContent instead of HTML interpolation for retrieved document data.
  const card = document.createElement("article");
  card.className = "citation-card";
  const title = document.createElement("h3");
  title.textContent = citation.source || "Unknown source";
  const details = document.createElement("dl");
  const values = [
    ["Chunk ID", citation.chunk_id],
    ["Index", citation.chunk_index ?? "—"],
    ["Distance", Number(citation.distance).toFixed(4)],
  ];
  values.forEach(([term, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    details.append(dt, dd);
  });
  card.setAttribute("aria-label", `Source ${index + 1}: ${title.textContent}`);
  card.append(title, details);
  citations.append(card);
}

function friendlyError(response, body) {
  if (response.status === 503) {
    return body?.detail ?? "The AI service is temporarily unavailable. Please try again shortly.";
  }
  if (response.status >= 500) {
    return "The backend had a problem processing your request. Please try again.";
  }
  return body?.detail ?? "Please check your question and try again.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
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
    metadata.replaceChildren();

    if (activeMode === "ask") {
      addMeta("Mode", "Normal");
      addMeta("Model", body.model);
      addMeta("Complexity", body.complexity);

      if (body.tool_used) {
        addMeta("Tool", "Web search used");
        toolQuery.textContent = body.tool_query || "(empty query)";
        toolSources.replaceChildren();
        if (body.tool_sources && body.tool_sources.length > 0) {
          body.tool_sources.forEach(renderWebSource);
        } else {
          const emptyItem = document.createElement("p");
          emptyItem.className = "empty-note";
          emptyItem.textContent = "No external URLs returned by search.";
          toolSources.append(emptyItem);
        }
        toolExecution.hidden = false;
      } else {
        addMeta("Tool", "No tool used");
        toolExecution.hidden = true;
      }
      ragEvidence.hidden = true;
    } else {
      // RAG mode
      addMeta("Mode", "RAG");
      addMeta("Model", body.model);
      addMeta("Complexity", body.complexity);
      addMeta("Grounding", body.abstained ? "Abstained" : "Passed");

      // Grounding banner
      ragStatusBanner.className = `rag-status-banner ${body.abstained ? "status-abstained" : "status-passed"}`;
      const statusIcon = document.createElement("span");
      statusIcon.className = "banner-icon";
      statusIcon.textContent = body.abstained ? "⚠" : "✓";
      const statusText = document.createElement("div");
      const statusTitle = document.createElement("strong");
      statusTitle.textContent = body.abstained
        ? "Grounding Decision: Abstained"
        : "Grounding Decision: Passed (Context Supported)";
      const statusDesc = document.createElement("p");
      statusDesc.textContent = body.abstained
        ? `Reason: ${body.abstention_reason || "Insufficient supporting context"}`
        : "Answer generated strictly from verified context chunks.";
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
        citationsList.forEach(renderCitation);
      } else {
        const noCitations = document.createElement("p");
        noCitations.className = "empty-note";
        noCitations.textContent = body.abstained
          ? "No supporting citations (abstained due to distance or judge verdict)."
          : "No citations available.";
        citations.append(noCitations);
      }

      // Execution & Debug Details
      ragJudgeReason.textContent = body.judge_reason || "None recorded";
      ragChunkCount.textContent = String(body.ids?.length ?? 0);
      ragIds.textContent = (body.ids ?? []).join(", ") || "None";
      ragDistances.textContent = (body.distances ?? []).map((d) => Number(d).toFixed(4)).join(", ") || "None";
      ragMetadata.textContent = JSON.stringify(body.metadatas ?? [], null, 2);
      context.textContent = body.context ?? "No retrieved evidence was returned.";

      toolExecution.hidden = true;
      ragEvidence.hidden = false;
    }

    result.hidden = false;
  } catch (error) {
    if (requestId !== latestRequestId) return;
    errorMessage.textContent =
      error instanceof TypeError
        ? "We couldn’t reach the backend. Confirm it is running at http://127.0.0.1:8000."
        : error.message;
    errorPanel.hidden = false;
  } finally {
    setLoading(false);
  }
});
