/* The workspace UI intentionally works independently of the optional 3D renderer. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  sources: [],
  graph: null,
  config: {},
  filter: "all",
  chatting: false,
  importing: false,
  source: null,
};
const icons = { text: "▤", image: "▧", video: "▷", audio: "♫" };
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const bytes = (n) =>
  n > 1048576
    ? (n / 1048576).toFixed(1) + " MB"
    : Math.max(1, Math.round(n / 1024)) + " KB";
const timestamp = (n) =>
  `${Math.floor(n / 60)}:${String(Math.floor(n % 60)).padStart(2, "0")}`;
function answerHTML(text) {
  return escapeHTML(text)
    .replace(/\*\*(.+?)\*\*/gs, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[S(\d+)\]/g, '<span class="inline-citation">[S$1]</span>');
}
let toastTimer;
function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.add("hidden"), 6000);
}
function status(message) {
  $("status-text").textContent = message;
}
async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok || data.error)
    throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
const post = (path, body = {}) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
function showDialog(id) {
  if (!$(id).open) $(id).showModal();
}
function showView(view) {
  document
    .querySelectorAll(".main-view")
    .forEach((node) =>
      node.classList.toggle("active", node.id === `view-${view}`),
    );
  document
    .querySelectorAll("[data-view]")
    .forEach((node) =>
      node.classList.toggle("active", node.dataset.view === view),
    );
  $("view-label").textContent =
    view === "map" ? "Memory map" : "Source library";
  if (view === "map") window.dispatchEvent(new Event("resize"));
}
function citations(container, sources, answer = "") {
  if (!sources?.length) return;
  const wrap = document.createElement("div");
  wrap.className = "answer-sources";
  const cited = new Set(
    [...answer.matchAll(/\[S(\d+)\]/g)].map((match) => Number(match[1])),
  );
  sources.forEach((source, index) => {
    if (cited.size ? !cited.has(index + 1) : index >= 3) return;
    const button = document.createElement("button");
    button.className = "citation";
    const at =
      source.start_seconds != null
        ? ` · ${timestamp(source.start_seconds)}`
        : "";
    button.innerHTML = `<span>S${index + 1}</span><span>${escapeHTML(source.name)}${at}</span>`;
    button.addEventListener("click", () =>
      openSource(source.source_id, source.start_seconds, source.id),
    );
    wrap.appendChild(button);
  });
  container.appendChild(wrap);
}
function message(role, content, sources = []) {
  $("chat-empty")?.remove();
  const item = document.createElement("article");
  item.className = `chat-msg ${role}`;
  item.innerHTML = `<div class="who">${role === "assistant" ? "✳ SECOND BRAIN" : role === "user" ? "YOU" : "WORKSPACE"}</div><div class="body"></div>`;
  if (role === "assistant")
    item.querySelector(".body").innerHTML = answerHTML(content);
  else item.querySelector(".body").textContent = content;
  citations(item, sources, content);
  $("chat-history").appendChild(item);
  $("chat-history").scrollTop = $("chat-history").scrollHeight;
  return item;
}
async function refresh() {
  const [sources, graph] = await Promise.all([
    request("/api/sources"),
    request("/api/graph"),
  ]);
  state.sources = sources.sources;
  state.graph = graph;
  $("nav-source-count").textContent = state.sources.length;
  $("nav-memory-count").textContent = graph.crystals.length;
  $("count-memories").textContent = graph.crystals.length;
  $("count-sources").textContent = graph.sources.length;
  const visibleKeys = new Set([
    ...graph.crystals.map((c) => `crystal_${c.id}`),
    ...graph.entities.map((e) => `entity_${e.id}`),
    ...graph.sources.map((s) => `source_${s.id}`),
  ]);
  $("count-links").textContent = new Set(
    graph.relations
      .filter(
        (r) =>
          r.relation !== "semantic_similarity" &&
          visibleKeys.has(`${r.source_type}_${r.source_id}`) &&
          visibleKeys.has(`${r.target_type}_${r.target_id}`),
      )
      .map(
        (r) =>
          `${r.source_type}_${r.source_id}:${r.target_type}_${r.target_id}:${r.relation}`,
      ),
  ).size;
  $("map-empty").classList.toggle(
    "hidden",
    graph.crystals.length + graph.sources.length > 0,
  );
  renderLibrary();
  renderRecent();
  window.dispatchEvent(new CustomEvent("brain:data", { detail: graph }));
  status("Saved locally · connections from your data");
}
function renderRecent() {
  $("recent-sources").replaceChildren();
  if (!state.sources.length) {
    $("recent-sources").innerHTML =
      '<p class="muted small">Your next idea starts here.</p>';
    return;
  }
  for (const source of state.sources.slice(0, 9)) {
    const button = document.createElement("button");
    button.className = "recent-source";
    button.innerHTML = `<span>${icons[source.kind]}</span><span>${escapeHTML(source.name)}</span>`;
    button.addEventListener("click", () => openSource(source.id));
    $("recent-sources").appendChild(button);
  }
}
function renderLibrary() {
  const sources = state.sources.filter(
    (source) => state.filter === "all" || source.kind === state.filter,
  );
  $("source-grid").replaceChildren();
  if (!sources.length) {
    $("source-grid").innerHTML =
      '<div class="library-empty">Nothing here yet.<br>Add a source and make it part of your memory.</div>';
    return;
  }
  sources.forEach((source) => {
    const card = document.createElement("button");
    card.className = "source-card";
    const cover =
      source.kind === "image"
        ? `<img src="${source.url}" alt="" loading="lazy">`
        : icons[source.kind];
    card.innerHTML = `<div class="source-cover">${cover}</div><div class="source-card-info"><div class="eyebrow">${source.kind} · ${bytes(source.bytes)}</div><h3>${escapeHTML(source.name)}</h3><p>${source.chunks} ${source.kind === "video" ? "frames" : source.kind === "audio" ? "segments" : "passages"} · ${new Date(source.created_ts).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</p></div>`;
    card.addEventListener("click", () => openSource(source.id));
    $("source-grid").appendChild(card);
  });
}
async function openSource(id, seconds, chunkId) {
  try {
    const source = await request(`/api/sources/${id}`);
    state.source = source;
    $("source-title").textContent = source.name;
    const fromYouTube = source.metadata.origin === "youtube";
    $("source-kind").textContent = fromYouTube
      ? `YOUTUBE TRANSCRIPT · ${source.metadata.channel || bytes(source.bytes)}`
      : `${source.kind.toUpperCase()} · ${bytes(source.bytes)}`;
    $("source-meta").textContent = source.metadata.coverage;
    if (fromYouTube) {
      const link = document.createElement("a");
      link.href = source.metadata.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "timecode";
      link.textContent = "Watch on YouTube ↗";
      $("source-meta").append(" ", link);
    }
    $("download-source").href = source.url;
    $("source-preview").replaceChildren();
    $("source-passages").replaceChildren();
    if (["image", "video", "audio"].includes(source.kind)) {
      const media = document.createElement(
        source.kind === "image" ? "img" : source.kind,
      );
      media.src = source.url;
      media.alt = source.name;
      if (source.kind !== "image") {
        media.controls = true;
        media.preload = "metadata";
        if (seconds != null)
          media.addEventListener(
            "loadedmetadata",
            () => {
              media.currentTime = seconds;
            },
            { once: true },
          );
      }
      $("source-preview").appendChild(media);
    }
    if (source.kind === "video") {
      const frames = document.createElement("div");
      frames.className = "frame-grid";
      for (const passage of source.passages) {
        const button = document.createElement("button");
        button.innerHTML = `<img src="${passage.preview_url}" alt="Frame at ${timestamp(passage.start_seconds)}" loading="lazy"><span>↗ ${timestamp(passage.start_seconds)}</span>`;
        button.addEventListener("click", () => {
          $("source-preview").querySelector("video").currentTime =
            passage.start_seconds;
        });
        frames.appendChild(button);
      }
      $("source-passages").appendChild(frames);
    } else {
      source.passages.forEach((passage, i) => {
        const details = document.createElement("details");
        details.className = "passage";
        const body = fromYouTube
          ? linkTimecodes(escapeHTML(passage.text), source.metadata.url)
          : escapeHTML(passage.text);
        details.innerHTML = `<summary>${source.kind === "audio" ? "Segment at " + timestamp(passage.start_seconds) : "Passage " + (i + 1)}</summary><pre>${body}</pre>`;
        details.dataset.chunkId = passage.id;
        details.open = chunkId
          ? passage.id === chunkId
          : i === 0 && source.kind === "text";
        if (source.kind === "audio") {
          const player = document.createElement("audio");
          player.src = passage.preview_url;
          player.controls = true;
          player.preload = "none";
          details.appendChild(player);
        }
        $("source-passages").appendChild(details);
      });
    }
    showDialog("source-dialog");
    if (chunkId)
      $("source-passages")
        .querySelector(`[data-chunk-id="${Number(chunkId)}"]`)
        ?.scrollIntoView({ block: "nearest" });
    window.dispatchEvent(
      new CustomEvent("brain:focus", { detail: `source_${id}` }),
    );
  } catch (error) {
    toast(error.message);
  }
}
async function pollJob(job, onProgress) {
  while (job.status === "running") {
    onProgress(job.message);
    await new Promise((resolve) => setTimeout(resolve, 800));
    job = await request(`/api/jobs/${job.id}`);
  }
  if (job.status === "error") throw new Error(job.error);
  return job.result;
}
async function importFiles(files) {
  if (state.importing) return toast("An import is already in progress.");
  const chosen = Array.from(files);
  if (!chosen.length) return;
  state.importing = true;
  showDialog("import-dialog");
  $("file-input").disabled = true;
  try {
    for (const file of chosen) {
      const item = document.createElement("div");
      item.className = "import-item";
      $("import-progress").appendChild(item);
      item.textContent = `Uploading ${file.name}…`;
      try {
        if (!file.size || file.size > 64 * 1024 * 1024)
          throw new Error("Choose a nonempty file under 64 MB.");
        const job = await request(
          `/api/sources?name=${encodeURIComponent(file.name)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: file,
          },
        );
        const result = await pollJob(job, (text) => {
          item.textContent = text;
          status(text);
        });
        item.classList.add("done");
        item.textContent = `${result.duplicate ? "Already saved" : "Saved"} · ${file.name} · ${result.chunks} searchable excerpts`;
      } catch (error) {
        item.classList.add("error");
        item.textContent = `${file.name}: ${error.message}`;
      }
    }
    await refresh();
  } finally {
    state.importing = false;
    $("file-input").disabled = false;
    $("file-input").value = "";
  }
}
async function importYouTube(url) {
  url = String(url || "").trim();
  if (!url) return toast("Paste a YouTube link first.");
  if (state.importing) return toast("An import is already in progress.");
  state.importing = true;
  showDialog("import-dialog");
  $("btn-youtube").disabled = true;
  $("file-input").disabled = true;
  const item = document.createElement("div");
  item.className = "import-item";
  item.textContent = "Fetching transcript…";
  $("import-progress").appendChild(item);
  try {
    const job = await post("/api/sources/youtube", { url });
    const result = await pollJob(job, (text) => {
      item.textContent = text;
      status(text);
    });
    item.classList.add("done");
    item.textContent = `${result.duplicate ? "Already saved" : "Saved"} · ${result.name} · ${result.chunks} searchable excerpts`;
    $("youtube-url").value = "";
    await refresh();
  } catch (error) {
    item.classList.add("error");
    item.textContent = `YouTube: ${error.message}`;
  } finally {
    state.importing = false;
    $("btn-youtube").disabled = !state.config.capabilities?.youtube;
    $("file-input").disabled = false;
  }
}
function applyCapabilities() {
  const ready = Boolean(state.config.capabilities?.youtube);
  $("youtube-url").disabled = !ready;
  $("btn-youtube").disabled = !ready;
  if (!ready)
    $("youtube-hint").textContent =
      'YouTube transcripts need yt-dlp. Install it with: pip install -e ".[youtube]", then restart.';
}
// "[00:12:34]" markers in a YouTube transcript become links that open the video at that moment.
function linkTimecodes(html, videoUrl) {
  return html.replace(
    /\[(\d{2}):(\d{2}):(\d{2})\]/g,
    (match, h, m, s) =>
      `<a class="timecode" href="${escapeHTML(videoUrl)}&amp;t=${+h * 3600 + +m * 60 + +s}s" target="_blank" rel="noopener noreferrer">${match}</a>`,
  );
}
async function saveNote(fromChat = false) {
  const text = (fromChat ? $("chat-input") : $("note-text")).value.trim();
  if (!text) return toast("Write a thought first.");
  if (state.importing) return toast("Wait for the current import to finish.");
  const name = fromChat
    ? "Note " + new Date().toISOString().slice(0, 19).replace(/:/g, "-")
    : $("note-title").value.trim() || "Untitled thought";
  const file = new File([text], name.replace(/\.txt$/i, "") + ".txt", {
    type: "text/plain",
  });
  await importFiles([file]);
  // Keep the text available if the job failed, so a retry doesn't lose it.
  if ($("import-progress").lastElementChild?.classList.contains("done")) {
    (fromChat ? $("chat-input") : $("note-text")).value = "";
    if (!fromChat) $("note-title").value = "";
  }
}
async function sendChat(event) {
  event?.preventDefault();
  const query = $("chat-input").value.trim();
  if (!query || state.chatting) return;
  state.chatting = true;
  $("btn-send").disabled = true;
  $("chat-input").value = "";
  message("user", query);
  $("chat-status").textContent = "Finding relevant memories…";
  let output;
  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Could not start chat.");
    }
    output = message("assistant", "");
    const body = output.querySelector(".body");
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "",
      scene = null,
      completed = false;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const event of events) {
        if (!event.startsWith("data: ")) continue;
        const data = JSON.parse(event.slice(6));
        if (data.type === "scene") {
          scene = data.scene;
          window.dispatchEvent(
            new CustomEvent("brain:highlight", { detail: scene }),
          );
          $("chat-status").textContent = "Connecting the dots…";
        }
        if (data.type === "token") {
          body.textContent += data.content;
          $("chat-history").scrollTop = $("chat-history").scrollHeight;
        }
        if (data.type === "error") throw new Error(data.message);
        if (data.type === "done") {
          completed = true;
          body.innerHTML = answerHTML(data.answer || body.textContent);
          citations(output, data.scene.sources, data.answer);
        }
      }
    }
    if (!completed)
      throw new Error("The connection ended before the answer finished.");
    await refresh();
    if (scene)
      window.dispatchEvent(
        new CustomEvent("brain:highlight", { detail: scene }),
      );
  } catch (error) {
    if (output && !output.querySelector(".body").textContent) output.remove();
    message("system", error.message);
  } finally {
    state.chatting = false;
    $("btn-send").disabled = false;
    $("chat-status").textContent = "Grounded in what you’ve saved.";
  }
}
async function search(event) {
  event?.preventDefault();
  const query = $("global-search").value.trim();
  if (!query) return;
  $("search-title").textContent = query;
  $("search-results").textContent = "Searching your sources and memories…";
  showDialog("search-dialog");
  try {
    const scene = await post("/api/search", { query });
    $("search-results").replaceChildren();
    for (const hit of scene.sources || []) {
      const item = document.createElement("button");
      item.className = "result-card";
      item.innerHTML = `<strong>${icons[hit.kind]} ${escapeHTML(hit.name)}</strong><p>${escapeHTML(hit.text.slice(0, 400))}</p><small>${hit.similarity != null ? `Semantic similarity ${hit.similarity.toFixed(3)}` : "Text match"}${hit.start_seconds != null ? " · " + timestamp(hit.start_seconds) : ""}</small>`;
      item.addEventListener("click", () => {
        $("search-dialog").close();
        openSource(hit.source_id, hit.start_seconds, hit.id);
      });
      $("search-results").appendChild(item);
    }
    for (const memory of (scene.crystals || []).slice(0, 5)) {
      const item = document.createElement("button");
      item.className = "result-card";
      item.innerHTML = `<strong>✧ ${escapeHTML(memory.title)}</strong><p>${escapeHTML((memory.content || memory.summary).slice(0, 500))}</p><small>Saved memory · #${memory.id}</small>`;
      item.addEventListener("click", () => {
        $("search-dialog").close();
        showView("map");
        window.dispatchEvent(
          new CustomEvent("brain:focus", { detail: `crystal_${memory.id}` }),
        );
      });
      $("search-results").appendChild(item);
    }
    if (!$("search-results").children.length)
      $("search-results").textContent =
        "No matching material yet. Try a different description.";
    window.dispatchEvent(new CustomEvent("brain:highlight", { detail: scene }));
  } catch (error) {
    $("search-results").textContent = error.message;
  }
}
async function settings() {
  showDialog("settings-dialog");
  $("settings-status").textContent = "Loading models…";
  try {
    const [config, models, stats] = await Promise.all([
      request("/api/config"),
      request("/api/models"),
      request("/api/stats"),
    ]);
    state.config = config;
    const embed = models.models.filter((name) => /embed|minilm/i.test(name));
    const chat = models.models.filter(
      (name) => !/embed|minilm|flux|z-image|klein|:cloud/i.test(name),
    );
    for (const [id, names, value] of [
      ["chat-model", chat, config.chat_model],
      ["embed-model", embed, config.embed_model],
    ]) {
      $(id).replaceChildren();
      if (id === "embed-model")
        $(id).add(new Option("No embeddings (text matching only)", ""));
      names.forEach((name) => $(id).add(new Option(name, name)));
      $(id).value = value;
    }
    $("settings-status").textContent =
      config.warning || "Ollama connected. Changes are applied locally.";
    $("stats-summary").innerHTML =
      `<div class="eyebrow">${stats.sources} SOURCES · ${stats.passages} EXCERPTS · ${stats.active_crystals} MEMORIES</div>`;
  } catch (error) {
    $("settings-status").textContent = error.message;
  }
}
async function saveSettings() {
  $("btn-save-settings").disabled = true;
  try {
    const result = await post("/api/config", {
      chat_model: $("chat-model").value,
      embed_model: $("embed-model").value,
    });
    if (result.id)
      await pollJob(
        result,
        (text) => ($("settings-status").textContent = text),
      );
    state.config = await request("/api/config");
    $("active-chat-model").textContent =
      state.config.chat_model || "Select a chat model";
    $("settings-status").textContent = "Models updated. Your memory is ready.";
    await refresh();
  } catch (error) {
    $("settings-status").textContent = error.message;
  } finally {
    $("btn-save-settings").disabled = false;
  }
}
function inspectNode(detail) {
  const { node, connections } = detail;
  const data = node.data;
  const panel = $("node-inspector");
  panel.replaceChildren();
  panel.classList.remove("hidden");
  panel.innerHTML = `<button class="icon-button" aria-label="Close node details">×</button><div class="eyebrow">${escapeHTML(node.type.toUpperCase())}</div><h3>${escapeHTML(data.name || data.title)}</h3><p>${escapeHTML(data.description || data.summary || `${data.mention_count} mentions in your saved memories`)}</p>`;
  panel
    .querySelector("button")
    .addEventListener("click", () => panel.classList.add("hidden"));
  if (node.type === "source") {
    const button = document.createElement("button");
    button.className = "secondary";
    button.textContent = "Open source ↗";
    button.style.marginTop = "12px";
    button.addEventListener("click", () => openSource(data.id));
    panel.appendChild(button);
  }
  for (const connection of connections.slice(0, 8)) {
    const row = document.createElement("div");
    row.className = "connection-row";
    row.textContent = `${connection.relation === "semantic_similarity" ? "similarity suggestion" : connection.relation.replaceAll("_", " ")} → ${connection.title}${connection.relation === "semantic_similarity" ? ` · ${connection.weight.toFixed(3)}` : ""}`;
    panel.appendChild(row);
  }
  if (!connections.length) {
    const row = document.createElement("div");
    row.className = "connection-row";
    row.textContent = "No stored connections yet.";
    panel.appendChild(row);
  }
}
async function initWorkspace() {
  document
    .querySelectorAll(".add-source")
    .forEach((button) =>
      button.addEventListener("click", () => showDialog("import-dialog")),
    );
  document
    .querySelectorAll(".close-dialog")
    .forEach((button) =>
      button.addEventListener("click", () => button.closest("dialog").close()),
    );
  document
    .querySelectorAll("[data-view]")
    .forEach((button) =>
      button.addEventListener("click", () => showView(button.dataset.view)),
    );
  $("view-label").addEventListener("click", () =>
    showView($("view-map").classList.contains("active") ? "library" : "map"),
  );
  document.querySelectorAll("[data-filter]").forEach((button) =>
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      document
        .querySelectorAll("[data-filter]")
        .forEach((b) => b.classList.toggle("active", b === button));
      renderLibrary();
    }),
  );
  $("file-input").addEventListener("change", (event) =>
    importFiles(event.target.files),
  );
  const drop = $("drop-zone");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("drag-over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("drag-over");
    if (!event.dataTransfer.files.length) {
      // A link dragged from a browser tab arrives as a URL, not a file.
      const link = (
        event.dataTransfer.getData("text/uri-list") ||
        event.dataTransfer.getData("text/plain")
      )
        .split(/\r?\n/)
        .find((line) => line.trim() && !line.startsWith("#"));
      if (link && /youtu\.?be/i.test(link)) {
        $("youtube-url").value = link.trim();
        return importYouTube(link);
      }
    }
    importFiles(event.dataTransfer.files);
  });
  $("youtube-form").addEventListener("submit", (event) => {
    event.preventDefault();
    importYouTube($("youtube-url").value);
  });
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter") $("file-input").click();
  });
  $("btn-save-import-note").addEventListener("click", () => saveNote());
  $("btn-ingest").addEventListener("click", () => saveNote(true));
  $("chat-form").addEventListener("submit", sendChat);
  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChat();
    }
  });
  document.querySelectorAll("[data-prompt]").forEach((button) =>
    button.addEventListener("click", () => {
      $("chat-input").value = button.dataset.prompt;
      $("chat-input").focus();
    }),
  );
  $("search-form").addEventListener("submit", search);
  ["btn-settings", "btn-settings-top", "btn-model-shortcut"].forEach((id) =>
    $(id).addEventListener("click", settings),
  );
  $("btn-save-settings").addEventListener("click", saveSettings);
  $("btn-ask-source").addEventListener("click", () => {
    $("source-dialog").close();
    $("chat-input").value =
      `Tell me about ${state.source.name}. Use the original source and cite it.`;
    $("chat-input").focus();
  });
  $("btn-consolidate").addEventListener("click", async () => {
    try {
      $("btn-consolidate").disabled = true;
      await post("/api/consolidate");
      await refresh();
      toast("Memory consolidation finished.");
    } catch (error) {
      toast(error.message);
    } finally {
      $("btn-consolidate").disabled = false;
    }
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "i") {
      event.preventDefault();
      showDialog("import-dialog");
    }
  });
  window.addEventListener("brain:select", (event) => inspectNode(event.detail));
  try {
    state.config = await request("/api/config");
    $("active-chat-model").textContent =
      state.config.chat_model || "Select a chat model";
    applyCapabilities();
    if (state.config.warning) toast(state.config.warning);
    const history = await request("/api/history");
    history.messages.forEach((item) =>
      message(item.role, item.content, item.sources),
    );
    await refresh();
    if (state.config.active_job) {
      showDialog("import-dialog");
      const item = document.createElement("div");
      item.className = "import-item";
      $("import-progress").appendChild(item);
      await pollJob(
        await request("/api/jobs/" + state.config.active_job),
        (text) => (item.textContent = text),
      );
      item.classList.add("done");
      item.textContent = "Finished.";
      await refresh();
    }
  } catch (error) {
    status("Connection unavailable");
    toast(error.message);
  }
}
initWorkspace();
