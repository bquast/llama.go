// app.js — frontend for llm.go
//
// 1. Fetches SmolLM2-135M-Instruct-Q4_K_M.gguf from HuggingFace (~105 MB)
//    and caches it in the browser Cache API so subsequent loads are instant.
// 2. Instantiates llm.wasm (pure Go transformer).
// 3. Drives a minimal chat UI.

const MODEL_URL =
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/" +
  "SmolLM2-135M-Instruct-Q4_K_M.gguf";
const CACHE_NAME  = "llm-go-v1";
const MAX_TOKENS  = 200;
const TEMPERATURE = 0.7;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const log      = document.getElementById("log");
const input    = document.getElementById("input");
const sendBtn  = document.getElementById("send");
const status   = document.getElementById("status");

function setStatus(msg) { status.textContent = msg; }

function appendMsg(role, text) {
  const div   = document.createElement("div");
  div.className = "msg " + role;
  const label = document.createElement("span");
  label.className   = "label";
  label.textContent = role === "user" ? "you" : "llm.go";
  const body  = document.createElement("span");
  body.className   = "body";
  body.textContent = text;
  div.append(label, body);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return body;
}

// ── WASM boot ─────────────────────────────────────────────────────────────────
async function bootWasm() {
  setStatus("loading runtime…");
  const go = new Go(); // from wasm_exec.js

  // Attach the wasmReady listener BEFORE go.run() to avoid the race condition
  // where the event fires synchronously inside run().
  const wasmReady = new Promise(resolve =>
    document.addEventListener("wasmReady", resolve, { once: true })
  );

  const result = await WebAssembly.instantiateStreaming(fetch("llm.wasm"), go.importObject);
  go.run(result.instance);
  await wasmReady;
  setStatus("runtime ready");
}

// ── Model fetch + cache ───────────────────────────────────────────────────────
async function fetchModel() {
  const cache  = await caches.open(CACHE_NAME);
  const cached = await cache.match(MODEL_URL);
  if (cached) {
    setStatus("loading model from cache…");
    const buf = await cached.arrayBuffer();
    setStatus("model loaded from cache");
    return buf;
  }

  setStatus("downloading model (105 MB, cached after first load)…");
  const resp = await fetch(MODEL_URL);
  if (!resp.ok) throw new Error("fetch failed: " + resp.status);

  const total  = parseInt(resp.headers.get("content-length") || "0");
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) {
      const pct = ((received / total) * 100).toFixed(0);
      setStatus(`downloading… ${pct}%  (${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(0)} MB)`);
    }
  }

  const all = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }

  // Store for next visit
  await cache.put(MODEL_URL, new Response(all, {
    headers: { "content-type": "application/octet-stream" }
  }));

  setStatus("download complete");
  return all.buffer;
}

// ── Main init ─────────────────────────────────────────────────────────────────
async function init() {
  try {
    await bootWasm();
    const buf = await fetchModel();

    setStatus("parsing model…");
    const err = llm.load(buf);
    if (err) throw new Error("llm.load: " + err);

    setStatus("ready  ·  SmolLM2-135M-Instruct  ·  pure Go  ·  WASM");
    input.disabled   = false;
    sendBtn.disabled = false;
    input.focus();
  } catch (e) {
    setStatus("error: " + e.message);
    console.error(e);
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────
async function send() {
  const text = input.value.trim();
  if (!text || !llm.ready()) return;

  input.value      = "";
  input.disabled   = true;
  sendBtn.disabled = true;

  appendMsg("user", text);
  const replyEl = appendMsg("assistant", "…");
  setStatus("generating…");

  // llm.generate is synchronous (blocks the main thread while running).
  // We yield once so the browser can repaint the "…" placeholder.
  await new Promise(r => setTimeout(r, 0));
  try {
    const reply = llm.generate(text, MAX_TOKENS, TEMPERATURE);
    replyEl.textContent = reply || "(empty response)";
    setStatus("ready  ·  SmolLM2-135M-Instruct  ·  pure Go  ·  WASM");
  } catch (e) {
    replyEl.textContent = "error: " + e.message;
    setStatus("error");
  }

  input.disabled   = false;
  sendBtn.disabled = false;
  input.focus();
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});

init();