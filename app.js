// app.js — frontend for llama.go
//
// 1. Fetches SmolLM2-135M-Instruct-Q4_K_M.gguf from HuggingFace (~105 MB)
//    and caches it in the browser Cache API so subsequent loads are instant.
// 2. Instantiates llm.wasm (pure Go transformer).
// 3. Drives a minimal chat UI.

const MODEL_URL =
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/" +
  "SmolLM2-135M-Instruct-Q8_0.gguf";
const CACHE_NAME  = "llm-go-v2-q8";
const MAX_TOKENS  = 200;
const TEMPERATURE = 0.7;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const log      = document.getElementById("log");
const input    = document.getElementById("input");
const sendBtn  = document.getElementById("send");

function appendMsg(role, text) {
  const div   = document.createElement("div");
  div.className = "msg " + role;
  
  if (role !== "system") {
    const label = document.createElement("span");
    label.className   = "label";
    label.textContent = role === "user" ? "you" : "llama.go";
    div.appendChild(label);
  }
  
  const body  = document.createElement("span");
  body.className   = "body";
  body.textContent = text;
  div.appendChild(body);
  
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return body;
}

// ── WASM boot ─────────────────────────────────────────────────────────────────
async function bootWasm() {
  const msgBody = appendMsg("system", "loading runtime… ");
  const go = new Go(); // from wasm_exec.js

  // Attach the wasmReady listener BEFORE go.run() to avoid the race condition
  // where the event fires synchronously inside run().
  const wasmReady = new Promise(resolve =>
    document.addEventListener("wasmReady", resolve, { once: true })
  );

  const result = await WebAssembly.instantiateStreaming(fetch("llm.wasm"), go.importObject);
  go.run(result.instance);
  await wasmReady;
  msgBody.innerHTML = 'loading runtime… <span class="done">✅</span>';
}

// ── Model fetch + cache ───────────────────────────────────────────────────────
async function fetchModel() {
  const cache  = await caches.open(CACHE_NAME);
  const cached = await cache.match(MODEL_URL);
  if (cached) {
    const msgBody = appendMsg("system", "loading model from cache… ");
    const buf = await cached.arrayBuffer();
    msgBody.innerHTML = 'loading model from cache… <span class="done">✅</span>';
    return buf;
  }

  const msgBody = appendMsg("system", "downloading model… ");
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
      msgBody.innerHTML = `downloading model <span class="progress">[${pct}%]</span>`;
    }
  }

  const all = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }

  // Store for next visit
  await cache.put(MODEL_URL, new Response(all, {
    headers: { "content-type": "application/octet-stream" }
  }));

  msgBody.innerHTML = 'downloading model <span class="done">✅</span>';
  return all.buffer;
}

// ── Main init ─────────────────────────────────────────────────────────────────
async function init() {
  try {
    await bootWasm();
    const buf = await fetchModel();

    const msgBody = appendMsg("system", "parsing model… ");
    const err = llm.load(buf);
    if (err) throw new Error("llm.load: " + err);
    msgBody.innerHTML = 'parsing model… <span class="done">✅</span>';

    input.disabled   = false;
    sendBtn.disabled = false;
    input.focus();
  } catch (e) {
    appendMsg("system", "error: " + e.message);
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
  const replyEl = appendMsg("assistant", "🤔 thinking…");

  try {
    const finalReply = await llm.generate(text, MAX_TOKENS, TEMPERATURE, (partialText) => {
      replyEl.textContent = partialText;
      log.scrollTop = log.scrollHeight;
    });
    replyEl.textContent = finalReply || "(empty response)";
  } catch (e) {
    replyEl.textContent = "error: " + e.message;
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
