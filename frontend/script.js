// --- Elements ---
const apiUrlInput = document.getElementById("apiUrl");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropPrompt = document.getElementById("dropPrompt");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const barFill = document.getElementById("barFill");

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
let selectedFile = null;

// --- File selection ---
function handleFile(file) {
  if (!file) return;
  if (!ALLOWED_TYPES.includes(file.type)) {
    showError(`Unsupported file type. Please use JPEG, PNG or WEBP.`);
    return;
  }
  selectedFile = file;

  // preview
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropPrompt.hidden = true;
  };
  reader.readAsDataURL(file);

  predictBtn.disabled = false;
  clearStatus();
  resultEl.hidden = true;
}

// Click to open picker
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

// Drag & drop
["dragenter", "dragover"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

// --- Prediction ---
predictBtn.addEventListener("click", classify);

async function classify() {
  if (!selectedFile) return;

  const base = apiUrlInput.value.trim().replace(/\/+$/, ""); // strip trailing slash
  if (!base) {
    showError("Please enter the API URL.");
    return;
  }

  predictBtn.disabled = true;
  resultEl.hidden = true;
  showStatus("Classifying…");

  const formData = new FormData();
  formData.append("file", selectedFile); // field name must be "file"

  try {
    const res = await fetch(`${base}/predict`, {
      method: "POST",
      body: formData, // do NOT set Content-Type; the browser sets the multipart boundary
    });

    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      if (res.status === 429) {
        detail = "Rate limit reached (10/min). Please wait a moment.";
      } else {
        try {
          const err = await res.json();
          if (err.detail) detail = err.detail;
        } catch (_) {
          /* non-JSON error body */
        }
      }
      showError(detail);
      return;
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(
      "Could not reach the API. Check the URL is running and that CORS is enabled."
    );
  } finally {
    predictBtn.disabled = false;
  }
}

function renderResult(data) {
  clearStatus();
  const pct = Math.round((data.confidence || 0) * 100);
  resultLabel.textContent = (data.label || "unknown").replace(/_/g, " ");
  resultConfidence.textContent = `${pct}% confidence`;
  barFill.style.width = "0%";
  resultEl.hidden = false;
  // animate the bar after it's visible
  requestAnimationFrame(() => (barFill.style.width = `${pct}%`));
}

// --- Status helpers ---
function showStatus(msg) {
  statusEl.textContent = msg;
  statusEl.classList.remove("error");
  statusEl.hidden = false;
}
function showError(msg) {
  statusEl.textContent = msg;
  statusEl.classList.add("error");
  statusEl.hidden = false;
}
function clearStatus() {
  statusEl.hidden = true;
  statusEl.textContent = "";
}
