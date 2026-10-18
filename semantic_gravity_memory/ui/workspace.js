/* The workspace UI intentionally works independently of the optional 3D renderer. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  chatFocus: null,
  sources: [],
  graph: null,
  config: {},
  filter: "all",
  chatting: false,
  importing: false,
  source: null,
};
const icons = { text: "▤", image: "▧", video: "▷", audio: "♫", dataset: "▦" };
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
// Clicking the dimmed area around a dialog closes it. Clicks on the dialog's own padding
// also target the dialog, so compare with its box; and both the press and the release must
// be outside, so selecting text and letting go past the edge doesn't close it.
function closeOnOutsideClick(dialog) {
  let pressedOutside = false;
  const outside = (event) => {
    if (event.target !== dialog) return false;
    const box = dialog.getBoundingClientRect();
    return (
      event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom
    );
  };
  dialog.addEventListener("pointerdown", (event) => {
    pressedOutside = outside(event);
  });
  dialog.addEventListener("click", (event) => {
    if (pressedOutside && outside(event)) dialog.close();
    pressedOutside = false;
  });
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
    // A dataset's own size (its downloaded files), not the size of its summary document.
    card.innerHTML = `<div class="source-cover">${cover}</div><div class="source-card-info"><div class="eyebrow">${source.kind} · ${bytes(source.metadata?.bytes || source.bytes)}</div><h3>${escapeHTML(source.name)}</h3><p>${source.chunks} ${source.kind === "video" ? "frames" : source.kind === "audio" ? "segments" : "passages"} · ${new Date(source.created_ts).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</p></div>`;
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
    const fromGitHub = source.metadata.origin === "github";
    const fromHub = source.metadata.origin === "huggingface";
    $("source-kind").textContent = fromYouTube
      ? `YOUTUBE TRANSCRIPT · ${source.metadata.channel || bytes(source.bytes)}`
      : fromGitHub
        ? `GITHUB REPOSITORY · ${source.metadata.repo}`
        : fromHub
          ? `HUGGING FACE DATASET · ${source.metadata.dataset}`
          : `${source.kind.toUpperCase()} · ${bytes(source.bytes)}`;
    $("source-meta").textContent = source.metadata.coverage;
    if (fromYouTube || fromGitHub || fromHub) {
      const origin = originOf(source);
      const link = document.createElement("a");
      link.href = origin.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "timecode";
      link.textContent = origin.label;
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
          : fromGitHub
            ? linkFileLines(escapeHTML(passage.text), source.metadata)
            : escapeHTML(passage.text);
        const label =
          source.kind === "audio"
            ? "Segment at " + timestamp(passage.start_seconds)
            : passage.media_kind === "image"
              ? `Sample image ${i + 1}`
              : passage.media_kind === "audio"
                ? `Sample audio ${i + 1}`
                : "Passage " + (i + 1);
        details.innerHTML = `<summary>${label}</summary><pre>${body}</pre>`;
        details.dataset.chunkId = passage.id;
        details.open = chunkId
          ? passage.id === chunkId
          : i === 0 && ["text", "dataset"].includes(source.kind);
        // Audio segments, and a dataset's sample images and clips, preview inline.
        if (passage.preview_url && (source.kind === "audio" || passage.media_kind === "audio")) {
          const player = document.createElement("audio");
          player.src = passage.preview_url;
          player.controls = true;
          player.preload = "none";
          details.appendChild(player);
        } else if (passage.preview_url && passage.media_kind === "image") {
          const image = document.createElement("img");
          image.src = passage.preview_url;
          image.alt = passage.text;
          image.loading = "lazy";
          image.className = "passage-image";
          details.appendChild(image);
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
// Import notes don't pile up: starting an import (or closing the dialog) clears earlier
// finished notes, a "Saved" note fades after a few seconds, and an error stays until then.
function clearFinishedImports() {
  $("import-progress")
    .querySelectorAll(".import-item.done, .import-item.error")
    .forEach((item) => item.remove());
}
function finishImportItem(item, ok, text) {
  item.classList.add(ok ? "done" : "error");
  item.textContent = text;
  if (ok)
    setTimeout(() => {
      item.classList.add("fading");
      setTimeout(() => item.remove(), 400);
    }, 6000);
}
async function importFiles(files) {
  if (state.importing) return toast("An import is already in progress.");
  const chosen = Array.from(files);
  if (!chosen.length) return;
  state.importing = true;
  clearFinishedImports();
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
        finishImportItem(
          item,
          true,
          `${result.duplicate ? "Already saved" : "Saved"} · ${file.name} · ${result.chunks} searchable excerpts`,
        );
      } catch (error) {
        finishImportItem(item, false, `${file.name}: ${error.message}`);
      }
    }
    await refresh();
  } finally {
    state.importing = false;
    $("file-input").disabled = false;
    $("file-input").value = "";
  }
}
// One link field imports YouTube transcripts and GitHub repositories.
const YOUTUBE_HOSTS = ["youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "youtube-nocookie.com"];
function linkKind(url) {
  let text = String(url || "").trim();
  if (/^git@github\.com:/i.test(text)) return "github";
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(text)) text = "https://" + text;
  let host;
  try {
    host = new URL(text).hostname.toLowerCase().replace(/^www\./, "");
  } catch {
    return null;
  }
  if (YOUTUBE_HOSTS.includes(host)) return "youtube";
  if (host === "huggingface.co" || host === "hf.co") return "huggingface";
  return host === "github.com" ? "github" : null;
}
async function importLink(url) {
  url = String(url || "").trim();
  if (!url) return toast("Paste a YouTube or GitHub link first.");
  const kind = linkKind(url);
  if (!kind) return toast("Paste a YouTube video, GitHub repository, or Hugging Face dataset link.");
  if (kind === "youtube" && !state.config.capabilities?.youtube)
    return toast('YouTube transcripts need yt-dlp: pip install -e ".[youtube]", then restart.');
  if (state.importing) return toast("An import is already in progress.");
  state.importing = true;
  clearFinishedImports();
  showDialog("import-dialog");
  $("btn-link").disabled = true;
  $("file-input").disabled = true;
  const label = { youtube: "YouTube", github: "GitHub", huggingface: "Hugging Face" }[kind];
  const item = document.createElement("div");
  item.className = "import-item";
  item.textContent = {
    youtube: "Fetching transcript…",
    github: "Downloading repository…",
    huggingface: "Reading dataset…",
  }[kind];
  $("import-progress").appendChild(item);
  try {
    const job = await post(`/api/sources/${kind}`, { url });
    const result = await pollJob(job, (text) => {
      item.textContent = text;
      status(text);
    });
    finishImportItem(
      item,
      true,
      `${result.duplicate ? "Already saved" : "Saved"} · ${result.name} · ${result.chunks} searchable excerpts`,
    );
    $("link-url").value = "";
    await refresh();
  } catch (error) {
    finishImportItem(item, false, `${label}: ${error.message}`);
  } finally {
    state.importing = false;
    $("btn-link").disabled = false;
    $("file-input").disabled = false;
  }
}
function applyCapabilities() {
  if (!state.config.capabilities?.youtube)
    $("link-hint").textContent =
      'GitHub repositories import as-is. YouTube transcripts need yt-dlp: pip install -e ".[youtube]", then restart.';
}
// Where a source really comes from: the video, the repository, or the stored original file.
function originOf(source) {
  const meta = source.metadata || {};
  if (meta.origin === "youtube" && meta.url) return { url: meta.url, label: "Watch on YouTube ↗" };
  if (meta.origin === "github" && meta.url) return { url: meta.url, label: "Open on GitHub ↗" };
  if (meta.origin === "huggingface" && meta.url) return { url: meta.url, label: "Open on Hugging Face ↗" };
  return { url: source.url || `/api/sources/${source.id}/file`, label: "Open original ↗" };
}
const unescapeHTML = (html) =>
  html.replace(/&(amp|lt|gt|quot|#39);/g, (_, e) => ({ amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'" })[e]);
// "File: path (lines 10–42)" labels in a repository source link to those lines on GitHub.
function linkFileLines(html, meta) {
  const at = meta.commit || meta.ref || "HEAD";
  return html.replace(/File: (.+?) \(lines (\d+)–(\d+)\)/g, (match, path, a, b) => {
    const file = unescapeHTML(path).split("/").map(encodeURIComponent).join("/");
    const href = `https://github.com/${meta.repo}/blob/${encodeURIComponent(at)}/${file}#L${a}-L${b}`;
    return `<a class="timecode" href="${escapeHTML(href)}" target="_blank" rel="noopener noreferrer">${match}</a>`;
  });
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
  const focus = state.chatFocus;
  const sent = message("user", query);
  if (focus) {
    const tag = document.createElement("span");
    tag.className = "about-tag";
    tag.textContent = `about ${short(focus.title, 36)}`;
    sent.querySelector(".who").append(" · ", tag);
  }
  $("chat-status").textContent = "Finding relevant memories…";
  let output;
  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(focus ? { query, focus } : { query }),
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
// Side panels collapse to slim rails; the choice is remembered on this device.
function setPanelCollapsed(side, collapsed) {
  const left = side === "left";
  const name = left ? "sidebar" : "chat";
  const button = $(left ? "btn-toggle-left" : "btn-toggle-right");
  document
    .querySelector(".shell")
    .classList.toggle(left ? "left-collapsed" : "right-collapsed", collapsed);
  button.setAttribute("aria-expanded", String(!collapsed));
  button.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${name}`);
  button.title = `${collapsed ? "Expand" : "Collapse"} ${name} ( ${left ? "[" : "]"} )`;
  button.textContent = left === collapsed ? "›" : "‹";
  try {
    localStorage.setItem(`panel-${side}-collapsed`, collapsed ? "1" : "");
  } catch {
    /* storage unavailable (private window); the toggle still works */
  }
}
function togglePanel(side) {
  const shell = document.querySelector(".shell");
  setPanelCollapsed(
    side,
    !shell.classList.contains(side === "left" ? "left-collapsed" : "right-collapsed"),
  );
}
function restorePanels() {
  for (const side of ["left", "right"]) {
    let collapsed = false;
    try {
      collapsed = localStorage.getItem(`panel-${side}-collapsed`) === "1";
    } catch {
      /* storage unavailable */
    }
    setPanelCollapsed(side, collapsed);
  }
}
// Closing node details also lifts the map's focus, so it doesn't stay dimmed.
function hideNodeInspector() {
  const panel = $("node-inspector");
  if (panel.classList.contains("hidden")) return;
  panel.classList.add("hidden");
  window.dispatchEvent(new CustomEvent("brain:clear"));
}
const NODE_LABELS = { crystal: "MEMORY", entity: "CONCEPT", source: "SOURCE" };
const short = (text, n = 32) => (text.length > n ? text.slice(0, n - 1) + "…" : text);
function actionButton(className, label, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}
function inspectNode(detail) {
  const { node, connections } = detail;
  const data = node.data;
  const title = data.name || data.title || "Untitled";
  const panel = $("node-inspector");
  panel.replaceChildren();
  panel.classList.remove("hidden");
  panel.innerHTML = `<button class="icon-button" aria-label="Close node details">×</button><div class="eyebrow">${NODE_LABELS[node.type] || escapeHTML(node.type.toUpperCase())}</div><h3>${escapeHTML(title)}</h3><p>${escapeHTML(data.description || data.summary || `${data.mention_count} mentions in your saved memories`)}</p>`;
  panel.querySelector("button").addEventListener("click", hideNodeInspector);
  const actions = document.createElement("div");
  actions.className = "inspector-actions";
  // Open the real thing: a source's video, repository, or file; for a memory, its source.
  const origins =
    node.type === "source"
      ? [data]
      : connections.filter((c) => c.type === "source").map((c) => c.data);
  for (const source of origins.slice(0, 2)) {
    const origin = originOf(source);
    const link = document.createElement("a");
    link.className = "secondary";
    link.href = origin.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent =
      node.type === "source" ? origin.label : `Source: ${short(source.name, 26)} ↗`;
    link.title = node.type === "source" ? source.name : `${origin.label.replace(" ↗", "")}: ${source.name}`;
    actions.appendChild(link);
  }
  if (node.type === "source")
    actions.appendChild(actionButton("secondary", "Passages", () => openSource(data.id)));
  actions.appendChild(
    actionButton("secondary send-to-chat", "Send to chat ✳", () =>
      sendToChat(node, title, connections),
    ),
  );
  panel.appendChild(actions);
  for (const connection of connections.slice(0, 8)) {
    // Each connection is a step along the map: click to inspect that node instead.
    const row = actionButton("connection-row", "", () =>
      window.dispatchEvent(new CustomEvent("brain:focus", { detail: connection.key })),
    );
    row.textContent =
      connection.relation === "shared_words"
        ? `shares ${connection.words.map((w) => `“${w}”`).join(", ")} → ${connection.title}`
        : `${connection.relation === "semantic_similarity" ? "similarity suggestion" : connection.relation.replaceAll("_", " ")} → ${connection.title}${connection.relation === "semantic_similarity" ? ` · ${connection.weight.toFixed(3)}` : ""}`;
    row.title = "Show this on the map";
    panel.appendChild(row);
  }
  if (!connections.length) {
    const row = document.createElement("div");
    row.className = "connection-row";
    row.textContent = "No stored connections yet.";
    panel.appendChild(row);
  }
}
// "Send to chat" pins a map item to the conversation: until removed, questions include
// that item, its own passages (for a source), and its connections on the map.
function sendToChat(node, title, connections) {
  state.chatFocus = {
    type: node.type,
    id: node.data.id,
    title,
    connections: connections
      .slice(0, 12)
      .map((c) => ({
        title: c.title,
        relation: c.words ? `shares the words ${c.words.join(", ")} with` : c.relation,
        type: c.type,
      })),
  };
  renderChatFocus();
  setPanelCollapsed("right", false);
  $("chat-input").focus();
}
function renderChatFocus() {
  const chip = $("chat-focus");
  const focus = state.chatFocus;
  chip.classList.toggle("hidden", !focus);
  $("chat-input").placeholder = focus
    ? `Ask about “${short(focus.title, 40)}” or how it connects…`
    : "Ask, explore, remember…";
  if (!focus) return chip.replaceChildren();
  chip.innerHTML = `<span class="chip-kind">${NODE_LABELS[focus.type]}</span><span class="chip-title"></span><button type="button" class="chip-clear" aria-label="Stop asking about this">×</button>`;
  chip.querySelector(".chip-title").textContent = focus.title;
  chip.title = "Your next questions include this item and its connections";
  chip.querySelector("button").addEventListener("click", () => {
    state.chatFocus = null;
    renderChatFocus();
  });
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
  document.querySelectorAll("dialog").forEach(closeOnOutsideClick);
  $("import-dialog").addEventListener("close", clearFinishedImports);
  // The node details panel closes on a click anywhere else. This listens in the capture
  // phase, before the 3D map's own click handler, so clicking another node still reopens
  // the panel for that node. Drags (orbiting the map) and clicks inside dialogs don't count.
  let pressedAt = null;
  document.addEventListener(
    "pointerdown",
    (event) => (pressedAt = [event.clientX, event.clientY]),
    true,
  );
  document.addEventListener(
    "click",
    (event) => {
      const panel = $("node-inspector");
      if (panel.classList.contains("hidden") || panel.contains(event.target)) return;
      if (event.target.closest?.("dialog")) return;
      if (
        pressedAt &&
        Math.hypot(event.clientX - pressedAt[0], event.clientY - pressedAt[1]) > 5
      )
        return;
      hideNodeInspector();
    },
    true,
  );
  restorePanels();
  $("btn-toggle-left").addEventListener("click", () => togglePanel("left"));
  $("btn-toggle-right").addEventListener("click", () => togglePanel("right"));
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
      if (link && linkKind(link.trim())) {
        $("link-url").value = link.trim();
        return importLink(link);
      }
    }
    importFiles(event.dataTransfer.files);
  });
  $("link-form").addEventListener("submit", (event) => {
    event.preventDefault();
    importLink($("link-url").value);
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
    setPanelCollapsed("right", false);
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
      return;
    }
    // [ and ] fold the side panels; Escape closes node details. Never while typing.
    const typing = event.target.closest?.("input, textarea, select, [contenteditable]");
    const modifier = event.metaKey || event.ctrlKey || event.altKey;
    if (typing || modifier || document.querySelector("dialog[open]")) return;
    if (event.key === "[") togglePanel("left");
    else if (event.key === "]") togglePanel("right");
    else if (event.key === "Escape") hideNodeInspector();
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
      finishImportItem(item, true, "Finished.");
      await refresh();
    }
  } catch (error) {
    status("Connection unavailable");
    toast(error.message);
  }
}
initWorkspace();
