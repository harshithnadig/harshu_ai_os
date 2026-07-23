import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const modeDescriptions = {
  ask: "Direct AI response",
  rag: "Searches indexed notes before responding",
};

const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const modeButtons = document.querySelectorAll(".mode-button");
const modeHint = document.querySelector("#mode-hint");
const result = document.querySelector("#result");
const answer = document.querySelector("#answer");
const metadata = document.querySelector("#metadata");
const ragEvidence = document.querySelector("#rag-evidence");
const citations = document.querySelector("#citations");
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
  modeButtons.forEach((button) => { button.disabled = isLoading; });
  submitButton.innerHTML = isLoading
    ? '<span class="spinner" aria-hidden="true"></span><span>Thinking</span>'
    : '<span>Ask Harshu</span><b aria-hidden="true">↗</b>';
}

function clearOutput() {
  result.hidden = true;
  clearError();
  ragEvidence.hidden = true;
  citations.replaceChildren();
  context.textContent = "";
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

function renderCitation(citation, index) {
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
  if (response.status === 503) return body?.detail ?? "The AI service is temporarily unavailable. Please try again shortly.";
  if (response.status >= 500) return "The backend had a problem processing your request. Please try again.";
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
    addMeta("Model", body.model);
    addMeta("Complexity", body.complexity);
    if (activeMode === "rag") {
      (body.citations ?? []).forEach(renderCitation);
      context.textContent = body.context ?? "No retrieved evidence was returned.";
      ragEvidence.hidden = false;
    }
    result.hidden = false;
  } catch (error) {
    if (requestId !== latestRequestId) return;
    errorMessage.textContent = error instanceof TypeError
      ? "We couldn’t reach the backend. Confirm it is running at http://127.0.0.1:8000."
      : error.message;
    errorPanel.hidden = false;
  } finally {
    setLoading(false);
  }
});
