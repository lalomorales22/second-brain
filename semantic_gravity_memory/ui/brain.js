/** A graph of stored records. Every edge comes from the API; no decorative nodes. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const viewport = document.getElementById("viewport");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const colors = { crystal: 0xa49aff, entity: 0x76cdb6, source: 0xe5bc81 };
const edgeColors = {
  mentions: 0x526c79,
  co_occurred: 0x5b887a,
  source_memory: 0xb89359,
  semantic_similarity: 0xc4d992,
  contradicts: 0xdd687a,
  temporal_cluster: 0x8172a8,
};
let renderer,
  camera,
  scene,
  controls,
  composer,
  graph,
  nodes = new Map(),
  edges = [];
let graphData = null,
  showSimilarity = false;
let step = 0,
  hover = null,
  initialized = false,
  selectedKey = null,
  firstGraph = true;
const raycaster = new THREE.Raycaster(),
  pointer = new THREE.Vector2();
const clock = new THREE.Clock();

function initialPosition(key) {
  let seed = 2166136261;
  for (const c of key) seed = Math.imul(seed ^ c.charCodeAt(0), 16777619) >>> 0;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const angle = random() * Math.PI * 2,
    z = random() * 2 - 1,
    radius = 12 + random() * 12;
  return new THREE.Vector3(
    Math.sqrt(1 - z * z) * Math.cos(angle),
    z,
    Math.sqrt(1 - z * z) * Math.sin(angle),
  ).multiplyScalar(radius);
}
function label(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 80;
  const ctx = canvas.getContext("2d");
  ctx.font = "24px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  ctx.fillStyle = "#a6acb9";
  ctx.fillText(text.length > 32 ? text.slice(0, 30) + "…" : text, 320, 38);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0.85,
      depthTest: false,
    }),
  );
  sprite.scale.set(18, 2.25, 1);
  sprite.position.y = -2.4;
  sprite.userData.isLabel = true;
  return sprite;
}
function dispose(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      materials.forEach((material) => {
        material.map?.dispose();
        material.dispose();
      });
    }
  });
}
function build(data) {
  if (!initialized) return;
  graphData = data;
  const prior = new Map([...nodes].map(([key, n]) => [key, n.pos.clone()]));
  while (graph.children.length) {
    const child = graph.children[0];
    graph.remove(child);
    dispose(child);
  }
  nodes = new Map();
  edges = [];
  const records = [
    ...(data.crystals || []).map((d) => ({ type: "crystal", data: d })),
    ...(data.entities || []).map((d) => ({ type: "entity", data: d })),
    ...(data.sources || []).map((d) => ({ type: "source", data: d })),
  ];
  const important = new Set(
    (data.crystals || [])
      .slice()
      .sort(
        (a, b) =>
          (b.grav_mass || b.salience_combined) -
          (a.grav_mass || a.salience_combined),
      )
      .slice(0, 5)
      .map((c) => `crystal_${c.id}`),
  );
  for (const { type, data: record } of records) {
    const key = `${type}_${record.id}`;
    const weight =
      type === "crystal"
        ? Math.min(
            1,
            Math.max(record.grav_mass || record.salience_combined || 0.1, 0.1),
          )
        : type === "entity"
          ? Math.min(1, (record.mention_count || 1) / 15)
          : Math.min(1, Math.log2((record.chunks || 1) + 1) / 6);
    const radius =
      type === "crystal"
        ? 0.5 + weight * 0.9
        : type === "entity"
          ? 0.25 + weight * 0.35
          : 0.85;
    const geometry =
      type === "crystal"
        ? new THREE.SphereGeometry(radius, 24, 18)
        : type === "entity"
          ? new THREE.IcosahedronGeometry(radius, 0)
          : new THREE.OctahedronGeometry(radius, 0);
    const material = new THREE.MeshStandardMaterial({
      color: colors[type],
      emissive: colors[type],
      emissiveIntensity: 0.4 + weight * 0.4,
      roughness: 0.5,
      metalness: 0.15,
      transparent: true,
      opacity: 0.95,
    });
    const mesh = new THREE.Mesh(geometry, material);
    const pos = prior.get(key) || initialPosition(key);
    mesh.position.copy(pos);
    mesh.userData.key = key;
    if (type === "source" || important.has(key))
      mesh.add(label(record.name || record.title));
    if (type === "crystal" && record.memory_type === "semantic") {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius + 0.2, 0.025, 6, 40),
        new THREE.MeshBasicMaterial({
          color: colors[type],
          transparent: true,
          opacity: 0.4,
        }),
      );
      mesh.add(ring);
    }
    graph.add(mesh);
    nodes.set(key, {
      key,
      type,
      data: record,
      pos,
      vel: new THREE.Vector3(),
      mesh,
      radius,
      weight,
      baseGlow: 0.4 + weight * 0.4,
    });
  }
  const seen = new Set();
  for (const relation of data.relations || []) {
    if (relation.relation === "semantic_similarity" && !showSimilarity)
      continue;
    const a = nodes.get(`${relation.source_type}_${relation.source_id}`),
      b = nodes.get(`${relation.target_type}_${relation.target_id}`);
    if (!a || !b || a === b) continue;
    const key = [a.key, b.key, relation.relation].join(":");
    if (seen.has(key)) continue;
    seen.add(key);
    const geometry = new THREE.BufferGeometry().setFromPoints([a.pos, b.pos]);
    const material =
      relation.relation === "semantic_similarity"
        ? new THREE.LineDashedMaterial({
            color: edgeColors.semantic_similarity,
            dashSize: 0.5,
            gapSize: 0.3,
            transparent: true,
            opacity: 0.45,
          })
        : new THREE.LineBasicMaterial({
            color: edgeColors[relation.relation] || 0x536782,
            transparent: true,
            opacity: 0.18 + relation.weight * 0.28,
          });
    const line = new THREE.Line(geometry, material);
    if (material.isLineDashedMaterial) line.computeLineDistances();
    graph.add(line);
    edges.push({ a, b, line, data: relation });
  }
  step = 0;
  if (firstGraph && nodes.size) {
    firstGraph = false;
    fit();
  }
  document.getElementById("node-count").textContent =
    `${nodes.size} nodes · ${edges.length} links in view`;
  if (selectedKey && nodes.has(selectedKey)) focus(selectedKey, false);
}
function fit() {
  if (!nodes.size) return;
  const box = new THREE.Box3();
  for (const n of nodes.values()) box.expandByPoint(n.pos);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const half =
    Math.max(size.x / Math.max(camera.aspect, 0.35), size.y, size.z, 25) / 2;
  const distance =
    (half / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.4;
  const direction = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(center);
  camera.position
    .copy(center)
    .add(direction.multiplyScalar(Math.max(45, distance)));
  controls.update();
}
function layout() {
  if (step > 280 || !nodes.size) return;
  const list = [...nodes.values()];
  for (let i = 0; i < list.length; i++)
    for (let j = i + 1; j < list.length; j++) {
      const a = list[i],
        b = list[j],
        delta = a.pos.clone().sub(b.pos);
      const distance = Math.max(delta.length(), 1.5);
      delta.normalize().multiplyScalar(18 / (distance * distance));
      a.vel.add(delta);
      b.vel.sub(delta);
    }
  for (const e of edges) {
    const delta = e.b.pos.clone().sub(e.a.pos);
    const length = delta.length();
    const rest = e.data.relation === "source_memory" ? 4 : 7;
    delta
      .normalize()
      .multiplyScalar((length - rest) * 0.007 * Math.max(0.1, e.data.weight));
    e.a.vel.add(delta);
    e.b.vel.sub(delta);
  }
  let energy = 0;
  for (const n of nodes.values()) {
    n.vel.addScaledVector(n.pos, -0.0007);
    n.vel.multiplyScalar(0.8);
    n.vel.clampLength(0, 1.4);
    n.pos.add(n.vel);
    n.mesh.position.copy(n.pos);
    energy += n.vel.length();
  }
  for (const e of edges) {
    const positions = e.line.geometry.attributes.position;
    positions.setXYZ(0, e.a.pos.x, e.a.pos.y, e.a.pos.z);
    positions.setXYZ(1, e.b.pos.x, e.b.pos.y, e.b.pos.z);
    positions.needsUpdate = true;
    if (e.line.material.isLineDashedMaterial) e.line.computeLineDistances();
  }
  step++;
  if (step === 180) fit();
  if (step > 80 && energy / list.length < 0.002) step = 281;
}
function illuminate(keys) {
  for (const n of nodes.values()) {
    const active = keys.has(n.key);
    n.mesh.material.emissiveIntensity = active
      ? 1.3
      : n.baseGlow * (keys.size ? 0.32 : 1);
    n.mesh.material.opacity = keys.size && !active ? 0.28 : 0.95;
    for (const child of n.mesh.children)
      if (child.userData.isLabel)
        child.material.opacity = keys.size && !active ? 0.15 : 0.8;
  }
  for (const e of edges)
    e.line.material.opacity = keys.size
      ? keys.has(e.a.key) && keys.has(e.b.key)
        ? 0.75
        : 0.055
      : 0.18 + e.data.weight * 0.28;
}
function focus(key, move = true) {
  const node = nodes.get(key);
  if (!node) return;
  selectedKey = key;
  const keys = new Set([key]),
    connections = [];
  for (const e of edges) {
    if (e.a.key === key || e.b.key === key) {
      const other = e.a.key === key ? e.b : e.a;
      keys.add(other.key);
      connections.push({
        title: other.data.name || other.data.title,
        relation: e.data.relation,
        weight: e.data.weight,
      });
    }
  }
  illuminate(keys);
  if (move) controls.target.copy(node.pos);
  window.dispatchEvent(
    new CustomEvent("brain:select", {
      detail: { node: { type: node.type, data: node.data }, connections },
    }),
  );
}
function highlight(data) {
  const keys = new Set([
    ...(data.crystals || []).map((c) => `crystal_${c.id}`),
    ...(data.entities || []).map((e) => `entity_${e.id}`),
    ...(data.sources || []).map((s) => `source_${s.source_id}`),
  ]);
  illuminate(keys);
}
function resize() {
  if (!initialized) return;
  const width = viewport.clientWidth,
    height = viewport.clientHeight;
  if (!width || !height) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  composer.setSize(width, height);
}
function init() {
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: document.getElementById("brain-canvas"),
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x0a0c10, 0);
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(48, 1, 0.1, 1000);
    camera.position.set(8, 14, 75);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.065;
    controls.autoRotate = !reducedMotion;
    controls.autoRotateSpeed = 0.18;
    controls.minDistance = 6;
    controls.maxDistance = 400;
    scene.add(new THREE.AmbientLight(0xc6cbef, 2));
    const light = new THREE.PointLight(0xe0e4ff, 15, 200);
    light.position.set(10, 30, 30);
    scene.add(light);
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(
      new UnrealBloomPass(new THREE.Vector2(1, 1), 0.35, 0.4, 0.65),
    );
    composer.addPass(new OutputPass());
    graph = new THREE.Group();
    scene.add(graph);
    initialized = true;
    resize();
    new ResizeObserver(resize).observe(viewport);
    renderer.domElement.addEventListener("pointermove", (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(
        [...nodes.values()].map((n) => n.mesh),
        false,
      )[0];
      hover = hit ? nodes.get(hit.object.userData.key) : null;
      const tip = document.getElementById("tooltip");
      tip.classList.toggle("hidden", !hover);
      renderer.domElement.style.cursor = hover ? "pointer" : "grab";
      if (hover) {
        tip.replaceChildren();
        const title = document.createElement("div");
        title.className = "tt-title";
        title.textContent = hover.data.name || hover.data.title;
        const meta = document.createElement("div");
        meta.className = "tt-meta";
        meta.textContent = `${hover.type === "crystal" ? "memory" : hover.type} · click to see its connections`;
        tip.append(title, meta);
        tip.style.left =
          Math.min(
            event.clientX - rect.left + 15,
            Math.max(5, rect.width - 270),
          ) + "px";
        tip.style.top =
          Math.min(event.clientY - rect.top + 15, rect.height - 80) + "px";
      }
    });
    let down;
    renderer.domElement.addEventListener(
      "pointerdown",
      (e) => (down = [e.clientX, e.clientY]),
    );
    renderer.domElement.addEventListener("click", (e) => {
      if (
        hover &&
        down &&
        Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5
      )
        focus(hover.key);
    });
    renderer.domElement.addEventListener("pointerleave", () => {
      hover = null;
      document.getElementById("tooltip").classList.add("hidden");
    });
    document.getElementById("btn-fit").addEventListener("click", fit);
    document.getElementById("btn-similarity").addEventListener("click", () => {
      showSimilarity = !showSimilarity;
      document
        .getElementById("btn-similarity")
        .setAttribute("aria-pressed", String(showSimilarity));
      if (graphData) build(graphData);
    });
    document
      .getElementById("btn-rotate")
      .setAttribute("aria-pressed", String(controls.autoRotate));
    document.getElementById("btn-rotate").addEventListener("click", () => {
      controls.autoRotate = !controls.autoRotate;
      document
        .getElementById("btn-rotate")
        .setAttribute("aria-pressed", String(controls.autoRotate));
    });
    document.getElementById("btn-clear-focus").addEventListener("click", () => {
      selectedKey = null;
      illuminate(new Set());
      document.getElementById("node-inspector").classList.add("hidden");
      fit();
    });
    window.addEventListener("brain:data", (event) => build(event.detail));
    window.addEventListener("brain:focus", (event) => focus(event.detail));
    window.addEventListener("brain:highlight", (event) =>
      highlight(event.detail),
    );
    fetch("/api/graph")
      .then((r) => r.json())
      .then(build)
      .catch(() => {});
    function animate() {
      requestAnimationFrame(animate);
      if (!viewport.clientWidth || document.hidden) return;
      layout();
      controls.update();
      // Keep real data nodes legible when the graph is fitted into a narrow
      // panel. Importance still controls relative size; zoom never erases nodes.
      for (const node of nodes.values()) {
        const distance = camera.position.distanceTo(node.pos);
        const worldPerPixel =
          (2 * distance * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) /
          viewport.clientHeight;
        const pixels =
          node.type === "source"
            ? 5
            : node.type === "crystal"
              ? 3.5 + node.weight * 3
              : 2 + node.weight * 2;
        const scale = Math.max(1, (pixels * worldPerPixel) / node.radius);
        node.mesh.scale.setScalar(scale);
        for (const child of node.mesh.children) {
          if (!child.userData.isLabel) continue;
          child.visible =
            viewport.clientWidth > 520 ||
            hover?.key === node.key ||
            selectedKey === node.key;
          child.scale.set(
            (240 * worldPerPixel) / scale,
            (30 * worldPerPixel) / scale,
            1,
          );
          child.position.y = -node.radius - (14 * worldPerPixel) / scale;
        }
      }
      composer.render();
    }
    animate();
  } catch (error) {
    document.getElementById("graph-fallback").classList.remove("hidden");
    console.warn("Graph unavailable:", error.message);
  }
}
init();
