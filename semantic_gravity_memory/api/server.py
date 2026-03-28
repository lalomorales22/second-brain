"""
Semantic Gravity Memory — Web API Server

Lightweight HTTP server (stdlib only) that exposes the Memory API as JSON
endpoints and serves the Three.js 3D brain UI.

Features:
  - Streaming SSE responses for chat (tokens appear as they arrive)
  - Thinking/reasoning display (detects <think> tags from reasoning models)
  - Memory recall → grounded prompt → streamed response → auto-ingest

Usage::

    python -m semantic_gravity_memory.api.server          # default port 8487
    python -m semantic_gravity_memory.api.server --port 9000

"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import traceback
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Iterator, Optional
from urllib.parse import parse_qs, urlparse

from semantic_gravity_memory import Memory
from semantic_gravity_memory.utils import now_iso

UI_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "ui"))
DEFAULT_PORT = 8487
DEFAULT_OLLAMA_URL = "http://localhost:11434/api"

# Shared mutable state — set once at startup, read from handler threads.
_state: Dict[str, Any] = {
    "memory": None,
    "ollama_url": DEFAULT_OLLAMA_URL,
    "chat_model": "gpt-oss:20b",
    "embed_model": "all-minilm",
    "last_scene": {},
}


# =========================================================================
# Ollama helpers (stdlib only)
# =========================================================================

def ollama_chat(base_url: str, model: str, prompt: str) -> str:
    """Non-streaming chat — used as fallback."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


def ollama_chat_stream(base_url: str, model: str, prompt: str) -> Iterator[Dict]:
    """Streaming chat — yields chunks as they arrive from Ollama."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=600)
    try:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                yield chunk
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue
    finally:
        resp.close()


def ollama_models(base_url: str):
    req = urllib.request.Request(f"{base_url.rstrip('/')}/tags", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return sorted(
        [m.get("name", "") for m in data.get("models", []) if m.get("name")],
        key=str.lower,
    )


# =========================================================================
# JSON helpers
# =========================================================================

def _crystal_to_dict(c, storage=None) -> Dict[str, Any]:
    d = {
        "id": c.id,
        "created_ts": c.created_ts,
        "updated_ts": c.updated_ts,
        "title": c.title,
        "theme": c.theme,
        "summary": c.summary,
        "compressed_narrative": c.compressed_narrative,
        "self_state": c.self_state,
        "memory_type": c.memory_type,
        "confidence": c.confidence,
        "access_count": c.access_count,
        "last_accessed_ts": c.last_accessed_ts,
        "decay_rate": c.decay_rate,
        "version": c.version,
        "contradiction_state": c.contradiction_state,
        "future_implications": c.future_implications,
        "unresolved": c.unresolved,
        "salience": c.salience.to_dict(),
        "salience_combined": c.salience.combined(),
        "entity_ids": c.entity_ids,
        "parent_crystal_id": c.parent_crystal_id,
        "schema_id": c.schema_id,
        "grav_mass": 0.0,
    }
    # Read grav_mass directly from DB if storage available
    if storage and c.id is not None:
        try:
            row = storage.conn.execute(
                "SELECT grav_mass FROM crystals WHERE id=?", (c.id,)
            ).fetchone()
            if row and row[0] is not None:
                d["grav_mass"] = float(row[0])
        except Exception:
            pass
    return d


def _entity_to_dict(e) -> Dict[str, Any]:
    return {
        "id": e.id, "name": e.name, "kind": e.kind,
        "salience": e.salience, "mention_count": e.mention_count,
        "first_seen_ts": e.first_seen_ts, "last_seen_ts": e.last_seen_ts,
    }


def _relation_to_dict(r) -> Dict[str, Any]:
    return {
        "id": r.id, "source_type": r.source_type, "source_id": r.source_id,
        "target_type": r.target_type, "target_id": r.target_id,
        "relation": r.relation, "weight": r.weight,
    }


def _contradiction_to_dict(c) -> Dict[str, Any]:
    return {
        "id": c.id, "topic": c.topic, "claim_a": c.claim_a,
        "claim_b": c.claim_b, "resolution_state": c.resolution_state,
    }


# =========================================================================
# HTTP Handler
# =========================================================================

class BrainHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Only log errors, not every request
        if args and (str(args[0]).startswith("4") or str(args[0]).startswith("5")):
            # Suppress favicon 404 noise
            if "/favicon" not in (self.path or ""):
                super().log_message(format, *args)

    # ---------- helpers ----------

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str):
        self._send_json({"error": message}, status)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _serve_file(self, filepath: str):
        if not os.path.isfile(filepath):
            self.send_error(404)
            return
        mime, _ = mimetypes.guess_type(filepath)
        mime = mime or "application/octet-stream"
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, event_type: str, data: dict):
        """Send one Server-Sent Event."""
        payload = json.dumps({"type": event_type, **data}, ensure_ascii=False, default=str)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    @property
    def memory(self) -> Optional[Memory]:
        return _state["memory"]

    # ---------- GET routes ----------

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._serve_file(os.path.join(UI_DIR, "index.html"))
        elif path == "/favicon.ico":
            # Empty favicon — suppress 404
            self.send_response(204)
            self.end_headers()
        elif path.startswith("/static/"):
            relpath = path[len("/static/"):]
            self._serve_file(os.path.join(UI_DIR, relpath))
        elif path == "/api/stats":
            self._api_stats()
        elif path == "/api/graph":
            self._api_graph(parsed)
        elif path == "/api/models":
            self._api_models()
        elif path == "/api/export":
            self._api_export()
        elif path == "/api/config":
            self._api_get_config()
        else:
            self.send_error(404)

    # ---------- POST routes ----------

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/api/ingest":
            self._api_ingest()
        elif path == "/api/recall":
            self._api_recall()
        elif path == "/api/answer":
            self._api_answer_stream()
        elif path == "/api/consolidate":
            self._api_consolidate()
        elif path == "/api/feedback":
            self._api_feedback()
        elif path == "/api/config":
            self._api_set_config()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ---------- API: streaming answer ----------

    def _api_answer_stream(self):
        """Streaming answer: recall → format prompt → stream chat → ingest.

        Uses Server-Sent Events so the frontend sees tokens as they arrive.
        Detects <think>...</think> tags for reasoning models.
        """
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        body = self._read_json_body()
        query = body.get("query", "").strip()
        if not query:
            return self._send_error_json(400, "query is required")

        model = body.get("model") or _state["chat_model"]
        url = body.get("ollama_url") or _state["ollama_url"]
        self_state = body.get("self_state")

        # 1. Recall (synchronous — fast with gravitational retrieval)
        try:
            scene = self.memory.recall(query, self_state=self_state)
            _state["last_scene"] = scene
        except Exception as e:
            return self._send_error_json(500, f"recall failed: {e}")

        # 2. Format the grounded prompt
        prompt = self.memory._engine._format_prompt(scene, query)

        # 3. Start SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send scene immediately so frontend can highlight nodes
        self._send_sse("scene", {"scene": scene})

        # 4. Stream from Ollama
        full_response = ""
        in_thinking = False

        try:
            for chunk in ollama_chat_stream(url, model, prompt):
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                if not content:
                    if chunk.get("done"):
                        break
                    continue

                # Detect <think> tags for reasoning models
                # Process character by character through the content
                i = 0
                while i < len(content):
                    remaining = content[i:]

                    if not in_thinking and remaining.startswith("<think>"):
                        in_thinking = True
                        i += len("<think>")
                        self._send_sse("thinking_start", {})
                        continue

                    if in_thinking and remaining.startswith("</think>"):
                        in_thinking = False
                        i += len("</think>")
                        self._send_sse("thinking_end", {})
                        continue

                    # Find next tag boundary
                    if in_thinking:
                        next_tag = remaining.find("</think>")
                        if next_tag == -1:
                            self._send_sse("thinking", {"content": remaining})
                            full_response += remaining
                            break
                        else:
                            self._send_sse("thinking", {"content": remaining[:next_tag]})
                            full_response += remaining[:next_tag]
                            i += next_tag
                            continue
                    else:
                        next_tag = remaining.find("<think>")
                        if next_tag == -1:
                            self._send_sse("token", {"content": remaining})
                            full_response += remaining
                            break
                        elif next_tag > 0:
                            self._send_sse("token", {"content": remaining[:next_tag]})
                            full_response += remaining[:next_tag]
                            i += next_tag
                            continue
                        else:
                            i += 0  # Will be caught by startswith above
                            # Force advance to avoid infinite loop
                            if remaining.startswith("<think>"):
                                in_thinking = True
                                i += len("<think>")
                                self._send_sse("thinking_start", {})
                            else:
                                self._send_sse("token", {"content": "<"})
                                full_response += "<"
                                i += 1

                if chunk.get("done"):
                    break

        except Exception as e:
            self._send_sse("error", {"message": str(e)})
            return

        # 5. Ingest the assistant response into memory
        try:
            clean_response = full_response
            # Strip thinking tags from the stored memory
            import re
            clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL).strip()
            if clean_response:
                self.memory.ingest(
                    clean_response,
                    actor="assistant",
                    kind="chat_response",
                    context={"query": query, "activation_id": scene.get("activation_id")},
                )
        except Exception:
            pass  # Don't fail the response if ingest fails

        # 6. Send done
        self._send_sse("done", {"answer": full_response, "scene": scene})

    # ---------- API: other endpoints ----------

    def _api_stats(self):
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        self._send_json(self.memory.stats())

    def _api_graph(self, parsed):
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        qs = parse_qs(parsed.query)
        crystal_limit = int(qs.get("crystal_limit", ["100"])[0])
        entity_limit = int(qs.get("entity_limit", ["50"])[0])

        crystals = self.memory._storage.recent_crystals(limit=crystal_limit)
        entities = self.memory._storage.top_entities(limit=entity_limit)
        relations = self.memory._storage.all_relations()
        contradictions = self.memory._storage.open_contradictions()

        crystal_ids = {c.id for c in crystals}
        entity_ids = {e.id for e in entities}

        filtered_relations = []
        for r in relations:
            src_ok = (r.source_type == "crystal" and r.source_id in crystal_ids) or \
                     (r.source_type == "entity" and r.source_id in entity_ids)
            tgt_ok = (r.target_type == "crystal" and r.target_id in crystal_ids) or \
                     (r.target_type == "entity" and r.target_id in entity_ids)
            if src_ok and tgt_ok:
                filtered_relations.append(r)

        self._send_json({
            "crystals": [_crystal_to_dict(c, self.memory._storage) for c in crystals],
            "entities": [_entity_to_dict(e) for e in entities],
            "relations": [_relation_to_dict(r) for r in filtered_relations],
            "contradictions": [_contradiction_to_dict(c) for c in contradictions],
            "last_scene": _state.get("last_scene", {}),
        })

    def _api_models(self):
        try:
            models = ollama_models(_state["ollama_url"])
            self._send_json({"models": models})
        except Exception as e:
            self._send_json({"models": [], "error": str(e)})

    def _api_export(self):
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        self._send_json(self.memory.export())

    def _api_get_config(self):
        self._send_json({
            "ollama_url": _state["ollama_url"],
            "chat_model": _state["chat_model"],
            "embed_model": _state["embed_model"],
        })

    def _api_ingest(self):
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        body = self._read_json_body()
        text = body.get("text", "").strip()
        if not text:
            return self._send_error_json(400, "text is required")
        actor = body.get("actor", "user")
        kind = body.get("kind", "chat_message")
        context = body.get("context")
        event_id, crystal_id = self.memory.ingest(text, actor, kind, context)
        self._send_json({"event_id": event_id, "crystal_id": crystal_id})

    def _api_recall(self):
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        body = self._read_json_body()
        query = body.get("query", "").strip()
        if not query:
            return self._send_error_json(400, "query is required")
        self_state = body.get("self_state")
        scene = self.memory.recall(query, self_state=self_state)
        _state["last_scene"] = scene
        self._send_json(scene)

    def _api_consolidate(self):
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        log = self.memory.consolidate()
        self._send_json(log)

    def _api_feedback(self):
        if not self.memory:
            return self._send_error_json(503, "memory not initialized")
        body = self._read_json_body()
        activation_id = body.get("activation_id")
        quality = body.get("quality")
        if activation_id is None or quality is None:
            return self._send_error_json(400, "activation_id and quality required")
        self_state = body.get("self_state", "general")
        self.memory.feedback(int(activation_id), float(quality), self_state)
        self._send_json({"ok": True})

    def _api_set_config(self):
        body = self._read_json_body()
        if "ollama_url" in body:
            _state["ollama_url"] = body["ollama_url"]
        if "chat_model" in body:
            _state["chat_model"] = body["chat_model"]
        if "embed_model" in body:
            _state["embed_model"] = body["embed_model"]
        self._send_json({"ok": True})


# =========================================================================
# Server entrypoint
# =========================================================================

class ThreadedHTTPServer(HTTPServer):
    allow_reuse_address = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def run(port: int = DEFAULT_PORT, ollama_url: str = DEFAULT_OLLAMA_URL,
        embed_model: str = "all-minilm", chat_model: str = "gpt-oss:20b"):
    _state["ollama_url"] = ollama_url
    _state["chat_model"] = chat_model
    _state["embed_model"] = embed_model

    try:
        _state["memory"] = Memory(ollama_model=embed_model, ollama_url=ollama_url)
    except Exception:
        print("[warn] could not init with embeddings, falling back to no-embedding mode")
        _state["memory"] = Memory()

    server = ThreadedHTTPServer(("0.0.0.0", port), BrainHandler)
    print(f"\n  second brain \u2014 3D visualization")
    print(f"  http://localhost:{port}")
    print(f"  ollama: {ollama_url}  chat: {chat_model}  embed: {embed_model}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
        _state["memory"].close()
        server.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Second Brain \u2014 3D Web UI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--embed-model", default="all-minilm")
    parser.add_argument("--chat-model", default="gpt-oss:20b")
    args = parser.parse_args()
    run(port=args.port, ollama_url=args.ollama_url,
        embed_model=args.embed_model, chat_model=args.chat_model)


if __name__ == "__main__":
    main()
