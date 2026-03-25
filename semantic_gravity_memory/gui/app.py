#!/usr/bin/env python3
"""
Semantic Gravity Memory Lab — Desktop GUI

A sleek dark tkinter app built on top of the Memory API.
Tabs: Chat, Graph, Crystals, Stats, System.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:
    raise SystemExit("tkinter is required for the GUI")

from semantic_gravity_memory import Memory
from semantic_gravity_memory.embeddings.ollama import OllamaEmbedder
from semantic_gravity_memory.utils import now_iso, summarize_text, safe_json_dumps

APP_TITLE = "semantic gravity memory lab"
APP_DIR = os.path.join(os.path.expanduser("~"), ".semantic_gravity_memory")
EXPORT_DIR = os.path.join(APP_DIR, "exports")
DEFAULT_CHAT_MODEL = "gemma3"
DEFAULT_EMBED_MODEL = "all-minilm"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api"


# =========================================================================
# Force-directed graph layout (hand-built, no dependencies)
# =========================================================================


def force_directed_layout(
    nodes: List[Dict],
    edges: List[Tuple[int, int, float]],
    width: float,
    height: float,
    iterations: int = 60,
) -> Dict[int, Tuple[float, float]]:
    """Simple spring-electric force-directed layout.

    nodes: list of {"id": int, "size": float}
    edges: list of (source_id, target_id, weight)
    Returns {node_id: (x, y)}.
    """
    if not nodes:
        return {}

    cx, cy = width / 2, height / 2
    positions: Dict[int, List[float]] = {}
    for n in nodes:
        positions[n["id"]] = [
            cx + random.uniform(-width * 0.3, width * 0.3),
            cy + random.uniform(-height * 0.3, height * 0.3),
        ]

    id_set = {n["id"] for n in nodes}
    repulsion = 8000.0
    attraction = 0.005
    damping = 0.85
    velocities: Dict[int, List[float]] = {n["id"]: [0.0, 0.0] for n in nodes}

    for _ in range(iterations):
        forces: Dict[int, List[float]] = {n["id"]: [0.0, 0.0] for n in nodes}

        # Repulsion between all pairs
        node_ids = list(positions.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = node_ids[i], node_ids[j]
                dx = positions[a][0] - positions[b][0]
                dy = positions[a][1] - positions[b][1]
                dist = max(1.0, math.sqrt(dx * dx + dy * dy))
                force = repulsion / (dist * dist)
                fx, fy = force * dx / dist, force * dy / dist
                forces[a][0] += fx
                forces[a][1] += fy
                forces[b][0] -= fx
                forces[b][1] -= fy

        # Attraction along edges
        for src, tgt, w in edges:
            if src not in id_set or tgt not in id_set:
                continue
            dx = positions[tgt][0] - positions[src][0]
            dy = positions[tgt][1] - positions[src][1]
            dist = max(1.0, math.sqrt(dx * dx + dy * dy))
            force = attraction * dist * w
            fx, fy = force * dx / dist, force * dy / dist
            forces[src][0] += fx
            forces[src][1] += fy
            forces[tgt][0] -= fx
            forces[tgt][1] -= fy

        # Gravity toward center
        for nid in node_ids:
            dx = cx - positions[nid][0]
            dy = cy - positions[nid][1]
            forces[nid][0] += dx * 0.0005
            forces[nid][1] += dy * 0.0005

        # Update positions
        margin = 40
        for nid in node_ids:
            velocities[nid][0] = (velocities[nid][0] + forces[nid][0]) * damping
            velocities[nid][1] = (velocities[nid][1] + forces[nid][1]) * damping
            positions[nid][0] = max(margin, min(width - margin, positions[nid][0] + velocities[nid][0]))
            positions[nid][1] = max(margin, min(height - margin, positions[nid][1] + velocities[nid][1]))

    return {nid: (p[0], p[1]) for nid, p in positions.items()}


# =========================================================================
# Ollama chat helper
# =========================================================================


def ollama_chat(base_url: str, model: str, prompt: str) -> str:
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


# =========================================================================
# Application
# =========================================================================


class MemoryLabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        os.makedirs(EXPORT_DIR, exist_ok=True)

        self.title(APP_TITLE)
        self.geometry("1520x960")
        self.minsize(1200, 800)
        self.configure(bg="#050505")

        # State
        self.chat_model_var = tk.StringVar(value=DEFAULT_CHAT_MODEL)
        self.embed_model_var = tk.StringVar(value=DEFAULT_EMBED_MODEL)
        self.base_url_var = tk.StringVar(value=DEFAULT_OLLAMA_URL)
        self.status_var = tk.StringVar(value="ready")
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.memory: Optional[Memory] = None
        self.current_scene: Dict[str, Any] = {}

        self._init_memory()
        self._configure_style()
        self._build_ui()
        self.after(180, self._drain_log)

    # -----------------------------------------------------------------
    # Memory init
    # -----------------------------------------------------------------

    def _init_memory(self):
        try:
            embed_model = self.embed_model_var.get().strip()
            url = self.base_url_var.get().strip()
            self.memory = Memory(
                ollama_model=embed_model, ollama_url=url,
            )
        except Exception:
            self.memory = Memory()

    # -----------------------------------------------------------------
    # Style
    # -----------------------------------------------------------------

    def _configure_style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure(".", background="#050505", foreground="#f4f4f4", fieldbackground="#0f0f10")
        s.configure("TFrame", background="#050505")
        s.configure("TLabel", background="#050505", foreground="#f4f4f4", font=("Helvetica", 10))
        s.configure("Title.TLabel", font=("Helvetica", 18, "bold"), foreground="#ffffff")
        s.configure("Sub.TLabel", font=("Helvetica", 10), foreground="#bcbcbc")
        s.configure("TButton", background="#111214", foreground="#ffffff", padding=8)
        s.map("TButton", background=[("active", "#1b1c20")])
        s.configure("TEntry", foreground="#ffffff", fieldbackground="#111214", insertcolor="#ffffff")
        s.configure("TNotebook", background="#050505", borderwidth=0)
        s.configure("TNotebook.Tab", background="#0d0d0f", foreground="#d8d8d8", padding=(14, 8))
        s.map("TNotebook.Tab", background=[("selected", "#17181c")], foreground=[("selected", "#ffffff")])
        s.configure("Treeview", background="#0f1012", foreground="#ececec", fieldbackground="#0f1012",
                     rowheight=26, borderwidth=0)
        s.map("Treeview", background=[("selected", "#262a31")])
        s.configure("Treeview.Heading", background="#121318", foreground="#ffffff", relief="flat")

    # -----------------------------------------------------------------
    # UI build
    # -----------------------------------------------------------------

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # Header
        top = ttk.Frame(root)
        top.pack(fill="x", padx=18, pady=(16, 10))
        left_top = ttk.Frame(top)
        left_top.pack(side="left", fill="x", expand=True)
        ttk.Label(left_top, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(left_top, text="crystals \u2022 spreading activation \u2022 temporal gravity \u2022 contradiction tracking \u2022 consolidation",
                  style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

        right_top = ttk.Frame(top)
        right_top.pack(side="right")
        ttk.Label(right_top, text="chat model").grid(row=0, column=0, sticky="e", padx=4)
        ttk.Entry(right_top, textvariable=self.chat_model_var, width=16).grid(row=0, column=1, padx=4)
        ttk.Label(right_top, text="embed model").grid(row=0, column=2, sticky="e", padx=4)
        ttk.Entry(right_top, textvariable=self.embed_model_var, width=16).grid(row=0, column=3, padx=4)

        # Tabs
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=18, pady=(0, 10))

        self.chat_tab = ttk.Frame(nb)
        self.graph_tab = ttk.Frame(nb)
        self.crystals_tab = ttk.Frame(nb)
        self.stats_tab = ttk.Frame(nb)
        self.system_tab = ttk.Frame(nb)
        nb.add(self.chat_tab, text="chat + memory")
        nb.add(self.graph_tab, text="graph")
        nb.add(self.crystals_tab, text="crystals")
        nb.add(self.stats_tab, text="stats")
        nb.add(self.system_tab, text="system")

        self._build_chat()
        self._build_graph()
        self._build_crystals()
        self._build_stats()
        self._build_system()

        # Footer
        bot = ttk.Frame(root)
        bot.pack(fill="x", padx=18, pady=(0, 14))
        ttk.Label(bot, textvariable=self.status_var, style="Sub.TLabel").pack(side="left")
        ttk.Button(bot, text="refresh", command=self.refresh_all).pack(side="right")
        ttk.Button(bot, text="consolidate", command=self._consolidate).pack(side="right", padx=(0, 8))

    # ---- Chat tab ----
    def _build_chat(self):
        w = ttk.Frame(self.chat_tab)
        w.pack(fill="both", expand=True, padx=8, pady=8)
        w.columnconfigure(0, weight=3)
        w.columnconfigure(1, weight=2)
        w.rowconfigure(0, weight=1)

        left = ttk.Frame(w)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.chat_out = tk.Text(left, bg="#08090b", fg="#f6f6f6", insertbackground="#fff",
                                wrap="word", relief="flat", padx=16, pady=16, font=("Menlo", 11))
        self.chat_out.grid(row=0, column=0, sticky="nsew")
        self.chat_out.tag_configure("user", foreground="#ffffff")
        self.chat_out.tag_configure("assistant", foreground="#c8d7ff")
        self.chat_out.tag_configure("header", foreground="#8a8a8a")

        bar = ttk.Frame(left)
        bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        bar.columnconfigure(0, weight=1)
        self.chat_in = tk.Text(bar, height=4, bg="#101114", fg="#fff", insertbackground="#fff",
                               wrap="word", relief="flat", padx=12, pady=10, font=("Menlo", 11))
        self.chat_in.grid(row=0, column=0, sticky="ew")
        btns = ttk.Frame(bar)
        btns.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        ttk.Button(btns, text="send", command=self._send_chat).pack(fill="x")
        ttk.Button(btns, text="ingest only", command=self._ingest_only).pack(fill="x", pady=(8, 0))

        right = ttk.Frame(w)
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text="active scene", style="Sub.TLabel").pack(anchor="w")
        self.scene_text = tk.Text(right, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat",
                                  padx=14, pady=14, font=("Menlo", 10))
        self.scene_text.pack(fill="both", expand=True, pady=(6, 0))

    # ---- Graph tab ----
    def _build_graph(self):
        w = ttk.Frame(self.graph_tab)
        w.pack(fill="both", expand=True, padx=8, pady=8)
        w.columnconfigure(0, weight=1)
        w.rowconfigure(0, weight=1)
        self.graph_canvas = tk.Canvas(w, bg="#060708", highlightthickness=0)
        self.graph_canvas.grid(row=0, column=0, sticky="nsew")

    # ---- Crystals tab ----
    def _build_crystals(self):
        w = ttk.Frame(self.crystals_tab)
        w.pack(fill="both", expand=True, padx=8, pady=8)
        w.columnconfigure(0, weight=3)
        w.columnconfigure(1, weight=2)
        w.rowconfigure(0, weight=1)

        cols = ("id", "title", "state", "type", "access", "version")
        self.crystal_tree = ttk.Treeview(w, columns=cols, show="headings")
        for c, wd in [("id", 50), ("title", 300), ("state", 100), ("type", 80), ("access", 60), ("version", 60)]:
            self.crystal_tree.heading(c, text=c)
            self.crystal_tree.column(c, width=wd, anchor="w")
        self.crystal_tree.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.crystal_tree.bind("<<TreeviewSelect>>", self._show_crystal)

        self.crystal_detail = tk.Text(w, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat",
                                      padx=12, pady=12, font=("Menlo", 10))
        self.crystal_detail.grid(row=0, column=1, sticky="nsew")

    # ---- Stats tab ----
    def _build_stats(self):
        w = ttk.Frame(self.stats_tab)
        w.pack(fill="both", expand=True, padx=8, pady=8)
        self.stats_text = tk.Text(w, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat",
                                  padx=14, pady=14, font=("Menlo", 11))
        self.stats_text.pack(fill="both", expand=True)

    # ---- System tab ----
    def _build_system(self):
        w = ttk.Frame(self.system_tab)
        w.pack(fill="both", expand=True, padx=8, pady=8)
        w.rowconfigure(1, weight=1)
        w.columnconfigure(0, weight=1)

        ctl = ttk.Frame(w)
        ctl.grid(row=0, column=0, sticky="ew")
        ttk.Label(ctl, text="ollama url").grid(row=0, column=0, sticky="e", padx=4)
        ttk.Entry(ctl, textvariable=self.base_url_var, width=36).grid(row=0, column=1, padx=4)
        ttk.Button(ctl, text="test ollama", command=self._test_ollama).grid(row=0, column=2, padx=4)
        ttk.Button(ctl, text="pull chat", command=self._pull_chat).grid(row=0, column=3, padx=4)
        ttk.Button(ctl, text="pull embed", command=self._pull_embed).grid(row=0, column=4, padx=4)
        ttk.Button(ctl, text="export json", command=self._export).grid(row=0, column=5, padx=4)
        ttk.Button(ctl, text="reconnect", command=self._init_memory).grid(row=0, column=6, padx=4)

        self.sys_log = tk.Text(w, bg="#08090b", fg="#d7d7d7", wrap="word", relief="flat",
                               padx=12, pady=12, font=("Menlo", 10))
        self.sys_log.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.sys_log.insert("1.0", f"app dir: {APP_DIR}\n")

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------

    def _log(self, msg: str):
        self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _drain_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.sys_log.insert("end", msg + "\n")
                self.sys_log.see("end")
        except queue.Empty:
            pass
        self.after(180, self._drain_log)

    def _append_chat(self, who: str, text: str, tag: str):
        self.chat_out.insert("end", f"\n{who}\n", "header")
        self.chat_out.insert("end", text.strip() + "\n", tag)
        self.chat_out.see("end")

    def _send_chat(self):
        text = self.chat_in.get("1.0", "end").strip()
        if not text or not self.memory:
            return
        self.chat_in.delete("1.0", "end")
        self._append_chat("you", text, "user")
        self.status_var.set("thinking with memory\u2026")

        def worker():
            try:
                url = self.base_url_var.get().strip()
                model = self.chat_model_var.get().strip()
                chat_fn = lambda prompt: ollama_chat(url, model, prompt)
                answer, scene = self.memory.answer(text, chat_fn)
                self.current_scene = scene
                self.after(0, lambda: self._append_chat("assistant", answer, "assistant"))
                self.after(0, lambda: self._update_scene(scene))
                self.after(0, self.refresh_all)
                self.after(0, lambda: self.status_var.set("ready"))
            except Exception as e:
                self._log(f"error: {e}")
                self.after(0, lambda: self.status_var.set("error"))
        threading.Thread(target=worker, daemon=True).start()

    def _ingest_only(self):
        text = self.chat_in.get("1.0", "end").strip()
        if not text or not self.memory:
            return
        self.chat_in.delete("1.0", "end")
        self._append_chat("you", text, "user")
        def worker():
            try:
                self.memory.ingest(text)
                self.after(0, self.refresh_all)
                self.after(0, lambda: self.status_var.set("ingested"))
            except Exception as e:
                self._log(str(e))
        threading.Thread(target=worker, daemon=True).start()

    def _update_scene(self, scene):
        self.scene_text.delete("1.0", "end")
        self.scene_text.insert("1.0", json.dumps(scene, indent=2, ensure_ascii=False, default=str))

    def _consolidate(self):
        if not self.memory:
            return
        def worker():
            log = self.memory.consolidate()
            self._log(f"consolidation: {json.dumps(log, default=str)}")
            self.after(0, self.refresh_all)
        threading.Thread(target=worker, daemon=True).start()

    # -----------------------------------------------------------------
    # Refresh
    # -----------------------------------------------------------------

    def refresh_all(self):
        self._refresh_crystals()
        self._refresh_stats()
        self._refresh_graph()

    def _refresh_crystals(self):
        for i in self.crystal_tree.get_children():
            self.crystal_tree.delete(i)
        if not self.memory:
            return
        for c in self.memory._storage.recent_crystals(limit=200):
            self.crystal_tree.insert("", "end", iid=str(c.id), values=(
                c.id, c.title, c.self_state, c.memory_type, c.access_count, c.version,
            ))

    def _show_crystal(self, event=None):
        sel = self.crystal_tree.selection()
        if not sel or not self.memory:
            return
        c = self.memory._storage.get_crystal(int(sel[0]))
        if not c:
            return
        d = c.__dict__.copy()
        d["salience"] = c.salience.to_dict()
        self.crystal_detail.delete("1.0", "end")
        self.crystal_detail.insert("1.0", json.dumps(d, indent=2, ensure_ascii=False, default=str))

    def _refresh_stats(self):
        if not self.memory:
            return
        self.stats_text.delete("1.0", "end")
        try:
            s = self.memory.stats()
            self.stats_text.insert("1.0", json.dumps(s, indent=2, ensure_ascii=False, default=str))
        except Exception as e:
            self.stats_text.insert("1.0", f"error: {e}")

    def _refresh_graph(self):
        c = self.graph_canvas
        c.delete("all")
        if not self.memory:
            return
        w = max(c.winfo_width(), 900)
        h = max(c.winfo_height(), 700)

        crystals = self.memory._storage.recent_crystals(limit=20)
        entities = self.memory._storage.top_entities(limit=12)
        relations = self.memory._storage.all_relations()[:300]
        scene_cids = {x["id"] for x in self.current_scene.get("crystals", [])}

        nodes: List[Dict] = []
        node_meta: Dict[int, Dict] = {}
        nid = 0
        crystal_id_map: Dict[int, int] = {}
        for cr in crystals:
            nid += 1
            crystal_id_map[cr.id] = nid
            size = 14 + int(12 * cr.salience.combined())
            nodes.append({"id": nid, "size": size})
            node_meta[nid] = {"type": "crystal", "label": cr.title, "active": cr.id in scene_cids, "size": size}

        entity_id_map: Dict[int, int] = {}
        for ent in entities:
            nid += 1
            entity_id_map[ent.id] = nid
            nodes.append({"id": nid, "size": 8 + int(6 * ent.salience)})
            node_meta[nid] = {"type": "entity", "label": ent.name, "active": False, "size": 8 + int(6 * ent.salience)}

        edges: List[Tuple[int, int, float]] = []
        for r in relations:
            src_map = crystal_id_map if r.source_type == "crystal" else entity_id_map
            tgt_map = crystal_id_map if r.target_type == "crystal" else entity_id_map
            s = src_map.get(r.source_id)
            t = tgt_map.get(r.target_id)
            if s and t:
                edges.append((s, t, r.weight))

        if not nodes:
            return
        positions = force_directed_layout(nodes, edges, w, h, iterations=60)

        # Draw edges
        for src, tgt, weight in edges:
            if src in positions and tgt in positions:
                x1, y1 = positions[src]
                x2, y2 = positions[tgt]
                c.create_line(x1, y1, x2, y2, fill="#2c3440", width=1 + 2 * weight)

        # Draw nodes
        for n in nodes:
            if n["id"] not in positions:
                continue
            x, y = positions[n["id"]]
            meta = node_meta[n["id"]]
            r = meta["size"]
            is_crystal = meta["type"] == "crystal"
            fill = "#18202b" if is_crystal else "#11261a"
            outline = "#96b3ff" if meta["active"] else ("#5080c0" if is_crystal else "#40856a")
            if meta["active"]:
                c.create_oval(x - r - 6, y - r - 6, x + r + 6, y + r + 6, outline="#2d3d58", width=2)
            c.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)
            label_char = "C" if is_crystal else "E"
            c.create_text(x, y, text=label_char, fill="#fff", font=("Helvetica", max(8, r // 2), "bold"))
            c.create_text(x, y + r + 12, text=summarize_text(meta["label"], 18), fill="#cfd4dd", font=("Helvetica", 9))

    # -----------------------------------------------------------------
    # System actions
    # -----------------------------------------------------------------

    def _test_ollama(self):
        def worker():
            try:
                url = self.base_url_var.get().strip().rstrip("/")
                req = urllib.request.Request(f"{url}/tags", method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                models = [m.get("name") for m in data.get("models", [])][:20]
                self._log("ollama ok: " + ", ".join(models) if models else "ollama ok (no models)")
                self.after(0, lambda: self.status_var.set("ollama ok"))
            except Exception as e:
                self._log(f"ollama fail: {e}")
                self.after(0, lambda: self.status_var.set("ollama fail"))
        threading.Thread(target=worker, daemon=True).start()

    def _pull_model(self, model_name):
        def worker():
            try:
                if not shutil.which("ollama"):
                    self._log("ollama cli not found")
                    return
                self._log(f"pulling {model_name}\u2026")
                proc = subprocess.Popen(["ollama", "pull", model_name],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:  # type: ignore
                    self._log(line.rstrip())
                proc.wait()
                self._log(f"ready: {model_name}")
            except Exception as e:
                self._log(f"pull error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def _pull_chat(self):
        self._pull_model(self.chat_model_var.get().strip())

    def _pull_embed(self):
        self._pull_model(self.embed_model_var.get().strip())

    def _export(self):
        if not self.memory:
            return
        filepath = filedialog.asksaveasfilename(
            title="export memory json",
            defaultextension=".json",
            initialdir=EXPORT_DIR,
            initialfile=f"memory_{time.strftime('%Y%m%d_%H%M%S')}.json",
            filetypes=[("json", "*.json")],
        )
        if not filepath:
            return
        try:
            data = self.memory.export()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            self._log(f"exported: {filepath}")
        except Exception as e:
            self._log(f"export error: {e}")


def main():
    app = MemoryLabApp()
    app.mainloop()


if __name__ == "__main__":
    main()
