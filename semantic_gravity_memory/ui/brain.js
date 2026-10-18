/**
 * A graph of stored records. Every node and edge comes from the API; nothing is decorative.
 *
 * Nodes are shader-lit orbs (flowing plasma, a bright rim, a slow spin) with a soft halo.
 * Edges are curved bundles of thin strands that fade into the orbs, with light pulses and
 * sparks travelling along them. Halos, strands, and sparks are each one batched draw call,
 * so the scene stays light on integrated GPUs.
 */
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
const SEGMENTS = 24; // line pieces per curved strand
const HALO = { crystal: 5.2, entity: 4.4, source: 4.8 }; // halo diameter, in orb radii

let renderer,
  camera,
  scene,
  controls,
  composer,
  graph,
  nodes = new Map(),
  edges = [],
  pickables = [];
let graphData = null,
  showSimilarity = false;
let step = 0,
  hover = null,
  initialized = false,
  selectedKey = null,
  firstGraph = true,
  edgesDirty = true;
let halo = null,
  strands = null,
  sparks = null;
const raycaster = new THREE.Raycaster(),
  pointer = new THREE.Vector2();
const clock = new THREE.Clock();
// Uniform objects shared by every material, so one assignment animates the whole scene.
const time = { value: 0 };
const motion = { value: reducedMotion ? 0 : 1 };
const pixelScale = { value: 800 }; // device pixels per world unit at distance 1
const shared = {};

/* ------------------------------------------------------------------ shaders */

const ORB_VERTEX = /* glsl */ `
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  varying vec3 vLocal;
  void main() {
    vLocal = position;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vViewDir = normalize(-mv.xyz);
    vNormalView = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * mv;
  }`;

// Domain-warped value noise gives slow, flowing plasma. It is sampled in the orb's own
// space, so the pattern turns with the orb's spin.
const ORB_FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform float uTime;
  uniform float uIntensity;
  uniform float uOpacity;
  uniform float uSeed;
  uniform float uShell;
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  varying vec3 vLocal;

  float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }
  float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
          mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
      mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
          mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
      f.z);
  }
  float fbm(vec3 p) {
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < 4; i++) {
      sum += amp * noise(p);
      p = p * 2.03 + vec3(1.7, 9.2, 3.1);
      amp *= 0.5;
    }
    return sum;
  }

  void main() {
    float facing = clamp(dot(normalize(vNormalView), normalize(vViewDir)), 0.0, 1.0);
    float rim = pow(1.0 - facing, 2.3);
    vec3 p = vLocal * 2.1 + uSeed;
    float t = uTime * 0.2;
    float warp = fbm(p + vec3(t, -0.7 * t, 0.4 * t));
    float flow = fbm(p * 1.7 + 2.3 * warp - vec3(0.0, 1.2 * t, 0.0));
    float veins = smoothstep(0.38, 0.8, flow);

    vec3 color = mix(uColor * 0.2, uColor * 1.1, veins);
    color += pow(veins, 3.0) * 0.45;          // hot filaments
    color += uColor * rim * 1.9;              // glowing atmosphere at the edge
    color *= uIntensity;

    // Sources are glassy shells: see-through in the middle, bright at the rim.
    float body = mix(0.6 + 0.4 * veins, 0.18 + 0.35 * veins, uShell);
    float alpha = uOpacity * clamp(body + rim, 0.0, 1.0);
    gl_FragColor = vec4(color, alpha);
  }`;

const HALO_VERTEX = /* glsl */ `
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aPhase;
  attribute float aGlow;
  uniform float uTime;
  uniform float uMotion;
  uniform float uPixel;
  varying vec3 vColor;
  varying float vGlow;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float breathe = sin(uTime * 1.5 + aPhase) * uMotion;
    gl_PointSize = aSize * (1.0 + 0.1 * breathe) * uPixel / -mv.z;
    vColor = aColor;
    vGlow = aGlow * (0.88 + 0.12 * breathe);
    gl_Position = projectionMatrix * mv;
  }`;

const HALO_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vGlow;
  void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    float core = exp(-r * r * 6.0);
    float ring = (r - 0.5) * 6.0;
    float corona = exp(-ring * ring) * 0.18;
    float edge = 1.0 - smoothstep(0.75, 1.0, r);
    gl_FragColor = vec4(vColor, (core * 0.55 + corona) * edge * vGlow);
  }`;

const STRAND_VERTEX = /* glsl */ `
  attribute float aT;
  attribute vec3 aColor;
  attribute vec3 aInfo;   // x: opacity, y: pulse phase, z: dashed
  attribute float aLen;
  attribute float aFocus;
  varying float vT;
  varying vec3 vColor;
  varying vec3 vInfo;
  varying float vLen;
  varying float vFocus;
  void main() {
    vT = aT;
    vColor = aColor;
    vInfo = aInfo;
    vLen = aLen;
    vFocus = aFocus;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const STRAND_FRAGMENT = /* glsl */ `
  uniform float uTime;
  uniform float uMotion;
  varying float vT;
  varying vec3 vColor;
  varying vec3 vInfo;
  varying float vLen;
  varying float vFocus;
  void main() {
    float taper = smoothstep(0.0, 0.16, vT) * smoothstep(1.0, 0.84, vT);
    float head = fract(uTime * 0.14 + vInfo.y);
    float d = abs(vT - head);
    d = min(d, 1.0 - d);
    float pulse = exp(-d * d * 650.0) * uMotion;
    float dash = mix(1.0, step(0.5, fract(vT * vLen * 0.5)), vInfo.z);
    float alpha = vInfo.x * taper * dash * vFocus * (1.0 + 2.6 * pulse);
    gl_FragColor = vec4(vColor * (1.0 + 1.4 * pulse), clamp(alpha, 0.0, 1.0));
  }`;

const SPARK_VERTEX = /* glsl */ `
  attribute vec3 aEnd;
  attribute vec3 aControl;
  attribute vec3 aColor;
  attribute float aPhase;
  attribute float aSpeed;
  attribute float aFocus;
  uniform float uTime;
  uniform float uMotion;
  uniform float uPixel;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float t = fract(uTime * aSpeed + aPhase);
    float s = 1.0 - t;
    vec3 p = s * s * position + 2.0 * s * t * aControl + t * t * aEnd;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vAlpha = smoothstep(0.0, 0.18, t) * smoothstep(1.0, 0.82, t) * aFocus * uMotion;
    vColor = aColor;
    gl_PointSize = max(2.0, 0.5 * uPixel / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const SPARK_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    gl_FragColor = vec4(vColor, exp(-r * r * 5.0) * vAlpha);
  }`;

/* ------------------------------------------------------------------ helpers */

function seeded(key) {
  let seed = 2166136261;
  for (const c of key) seed = Math.imul(seed ^ c.charCodeAt(0), 16777619) >>> 0;
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}
function initialPosition(key) {
  const random = seeded(key);
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
  ctx.shadowColor = "rgba(5, 7, 11, 0.9)";
  ctx.shadowBlur = 8;
  ctx.fillStyle = "#b3b9c6";
  ctx.fillText(text.length > 32 ? text.slice(0, 30) + "…" : text, 320, 38);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0.85,
      depthTest: false,
    }),
  );
  sprite.renderOrder = 10; // text sits above the glow layers
  sprite.userData.isLabel = true;
  return sprite;
}
function vignette() {
  // The CSS backdrop, painted into the scene so the glow pass works on an opaque frame.
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 512;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, 512, 512);
  const gradient = ctx.createRadialGradient(307, 230, 0, 307, 230, 272);
  gradient.addColorStop(0, "rgba(22, 24, 38, 0.44)");
  gradient.addColorStop(1, "rgba(22, 24, 38, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 512, 512);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
function dispose(object) {
  object.traverse((child) => {
    if (!child.geometry?.userData.shared) child.geometry?.dispose();
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
function sharedGeometry(name, make) {
  if (!shared[name]) {
    shared[name] = make();
    shared[name].userData.shared = true;
  }
  return shared[name];
}
function orbMaterial(type, seed) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(colors[type]) },
      uTime: time,
      uIntensity: { value: 1 },
      uOpacity: { value: 0.95 },
      uSeed: { value: seed * 37 },
      uShell: { value: type === "source" ? 1 : 0 },
    },
    vertexShader: ORB_VERTEX,
    fragmentShader: ORB_FRAGMENT,
    transparent: true,
  });
}
function attribute(geometry, name, size, count) {
  const value = new THREE.BufferAttribute(new Float32Array(count * size), size);
  geometry.setAttribute(name, value);
  return value.array;
}
function points(geometry, vertexShader, fragmentShader, renderOrder) {
  const object = new THREE.Points(
    geometry,
    new THREE.ShaderMaterial({
      uniforms: { uTime: time, uMotion: motion, uPixel: pixelScale },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  object.frustumCulled = false; // positions change every frame; bounds would go stale
  object.renderOrder = renderOrder;
  return object;
}

/* ------------------------------------------------------------------ building */

function makeNode(type, record, important, prior) {
  const key = `${type}_${record.id}`;
  const random = seeded(key + ":look");
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
  const orb = sharedGeometry("orb", () => new THREE.SphereGeometry(1, 48, 32));
  const mesh = new THREE.Mesh(orb, orbMaterial(type, random()));
  const pos = prior.get(key) || initialPosition(key);
  mesh.position.copy(pos);
  mesh.rotation.x = (random() - 0.5) * 0.9;
  mesh.userData.key = key;
  const node = {
    key,
    type,
    data: record,
    pos,
    vel: new THREE.Vector3(),
    mesh,
    radius,
    weight,
    level: 1,
    boost: 1,
    alpha: 0.95,
    baseLevel: type === "crystal" ? 0.85 + weight * 0.4 : 0.95,
    glow: 0.55 + weight * 0.6,
    spin: (0.12 + random() * 0.25) * (random() < 0.5 ? -1 : 1),
    phase: random() * Math.PI * 2,
  };
  node.level = node.baseLevel;
  if (type === "source" || important.has(key))
    mesh.add(label(record.name || record.title));
  if (type === "crystal" && record.memory_type === "semantic") {
    // Semantic memories keep their ring, now a thin orbit that turns on its own axis.
    node.ring = new THREE.Mesh(
      sharedGeometry("ring", () => new THREE.TorusGeometry(1.45, 0.025, 8, 96)),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(colors.crystal).multiplyScalar(1.3),
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    node.ring.rotation.x = Math.PI / 2 + (random() - 0.5) * 0.8;
    node.ring.userData.baseOpacity = 0.5;
    mesh.add(node.ring);
  }
  if (type === "source") {
    // Sources keep their diamond, now a bright core turning inside a glass shell.
    node.core = new THREE.Mesh(
      sharedGeometry("diamond", () => new THREE.OctahedronGeometry(0.56, 0)),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(colors.source).multiplyScalar(1.35),
      }),
    );
    node.core.userData.baseColor = node.core.material.color.clone();
    mesh.add(node.core);
  }
  return node;
}
function makeHalo(list) {
  const geometry = new THREE.BufferGeometry();
  attribute(geometry, "position", 3, list.length);
  const color = attribute(geometry, "aColor", 3, list.length);
  attribute(geometry, "aSize", 1, list.length);
  const phase = attribute(geometry, "aPhase", 1, list.length);
  const glow = attribute(geometry, "aGlow", 1, list.length);
  list.forEach((node, i) => {
    node.index = i;
    new THREE.Color(colors[node.type]).toArray(color, i * 3);
    phase[i] = node.phase;
    glow[i] = node.glow;
  });
  return points(geometry, HALO_VERTEX, HALO_FRAGMENT, 2);
}
function makeEdgeLayers() {
  let vertices = 0,
    sparkCount = 0;
  for (const e of edges) {
    e.vStart = vertices;
    e.vCount = e.strandCount * SEGMENTS * 2;
    vertices += e.vCount;
    e.spark = e.similarity ? -1 : sparkCount++;
  }
  const lineGeometry = new THREE.BufferGeometry();
  attribute(lineGeometry, "position", 3, vertices);
  const tAlong = attribute(lineGeometry, "aT", 1, vertices);
  const color = attribute(lineGeometry, "aColor", 3, vertices);
  const info = attribute(lineGeometry, "aInfo", 3, vertices);
  attribute(lineGeometry, "aLen", 1, vertices);
  const focus = attribute(lineGeometry, "aFocus", 1, vertices);
  const sparkGeometry = new THREE.BufferGeometry();
  attribute(sparkGeometry, "position", 3, sparkCount);
  attribute(sparkGeometry, "aEnd", 3, sparkCount);
  attribute(sparkGeometry, "aControl", 3, sparkCount);
  const sparkColor = attribute(sparkGeometry, "aColor", 3, sparkCount);
  const sparkPhase = attribute(sparkGeometry, "aPhase", 1, sparkCount);
  const sparkSpeed = attribute(sparkGeometry, "aSpeed", 1, sparkCount);
  attribute(sparkGeometry, "aFocus", 1, sparkCount).fill(1);

  const from = new THREE.Color(),
    to = new THREE.Color(),
    tint = new THREE.Color(),
    mixed = new THREE.Color();
  for (const e of edges) {
    from.set(colors[e.a.type]);
    to.set(colors[e.b.type]);
    tint.set(edgeColors[e.data.relation] || 0x536782);
    const tintAmount = e.data.relation === "contradicts" ? 0.8 : 0.45;
    let v = e.vStart;
    for (let s = 0; s < e.strandCount; s++) {
      const opacity = e.similarity
        ? 0.42
        : (0.14 + 0.2 * Math.max(0.1, e.data.weight)) * (s === 0 ? 1 : 0.6);
      for (let k = 0; k < SEGMENTS; k++)
        for (const t of [k / SEGMENTS, (k + 1) / SEGMENTS]) {
          tAlong[v] = t;
          mixed.copy(from).lerp(to, t).lerp(tint, tintAmount).toArray(color, v * 3);
          info[v * 3] = opacity;
          info[v * 3 + 1] = e.phase + s * 0.03;
          info[v * 3 + 2] = e.similarity ? 1 : 0;
          focus[v] = 1;
          v++;
        }
    }
    if (e.spark >= 0) {
      mixed.copy(from).lerp(to, 0.5).lerp(tint, 0.3).lerp(new THREE.Color(1, 1, 1), 0.3);
      mixed.toArray(sparkColor, e.spark * 3);
      sparkPhase[e.spark] = e.phase;
      sparkSpeed[e.spark] = 0.07 + seeded(e.id + ":speed")() * 0.08;
    }
  }
  strands = new THREE.LineSegments(
    lineGeometry,
    new THREE.ShaderMaterial({
      uniforms: { uTime: time, uMotion: motion },
      vertexShader: STRAND_VERTEX,
      fragmentShader: STRAND_FRAGMENT,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  strands.frustumCulled = false;
  strands.renderOrder = 1;
  sparks = points(sparkGeometry, SPARK_VERTEX, SPARK_FRAGMENT, 3);
  graph.add(strands, sparks);
  edgesDirty = true;
}

const _dir = new THREE.Vector3(),
  _helper = new THREE.Vector3(),
  _u = new THREE.Vector3(),
  _v = new THREE.Vector3(),
  _mid = new THREE.Vector3(),
  _control = new THREE.Vector3();
// Recompute every curve from the current node positions (after layout moves nodes).
function writeEdges() {
  if (!strands) return;
  const position = strands.geometry.attributes.position.array;
  const lengths = strands.geometry.attributes.aLen.array;
  const start = sparks.geometry.attributes.position.array;
  const end = sparks.geometry.attributes.aEnd.array;
  const control = sparks.geometry.attributes.aControl.array;
  for (const e of edges) {
    const a = e.a.pos,
      b = e.b.pos;
    _dir.subVectors(b, a);
    const length = _dir.length() || 1;
    _dir.divideScalar(length);
    _helper.set(0, 1, 0);
    if (Math.abs(_dir.y) > 0.9) _helper.set(1, 0, 0);
    _u.crossVectors(_dir, _helper).normalize();
    _v.crossVectors(_dir, _u);
    _mid.addVectors(a, b).multiplyScalar(0.5);
    let i = e.vStart * 3;
    for (let s = 0; s < e.strandCount; s++) {
      // Strands meet at both orbs and fan out around the axis in between.
      const angle = e.angle + (s - (e.strandCount - 1) / 2) * 0.55;
      const bend = length * e.bend * (1 + 0.18 * s);
      _control
        .copy(_mid)
        .addScaledVector(_u, Math.cos(angle) * bend)
        .addScaledVector(_v, Math.sin(angle) * bend);
      if (s === 0 && e.spark >= 0) {
        a.toArray(start, e.spark * 3);
        b.toArray(end, e.spark * 3);
        _control.toArray(control, e.spark * 3);
      }
      for (let k = 0; k < SEGMENTS; k++)
        for (const t of [k / SEGMENTS, (k + 1) / SEGMENTS]) {
          const q = 1 - t;
          position[i++] = q * q * a.x + 2 * q * t * _control.x + t * t * b.x;
          position[i++] = q * q * a.y + 2 * q * t * _control.y + t * t * b.y;
          position[i++] = q * q * a.z + 2 * q * t * _control.z + t * t * b.z;
        }
    }
    lengths.fill(length, e.vStart, e.vStart + e.vCount);
  }
  strands.geometry.attributes.position.needsUpdate = true;
  strands.geometry.attributes.aLen.needsUpdate = true;
  for (const name of ["position", "aEnd", "aControl"])
    sparks.geometry.attributes[name].needsUpdate = true;
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
  halo = strands = sparks = null;
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
    const node = makeNode(type, record, important, prior);
    graph.add(node.mesh);
    nodes.set(node.key, node);
  }
  pickables = [...nodes.values()].map((n) => n.mesh);
  halo = makeHalo([...nodes.values()]);
  graph.add(halo);

  const seen = new Set();
  for (const relation of data.relations || []) {
    const similarity = relation.relation === "semantic_similarity";
    if (similarity && !showSimilarity) continue;
    const a = nodes.get(`${relation.source_type}_${relation.source_id}`),
      b = nodes.get(`${relation.target_type}_${relation.target_id}`);
    if (!a || !b || a === b) continue;
    const id = [a.key, b.key, relation.relation].join(":");
    if (seen.has(id)) continue;
    seen.add(id);
    const random = seeded(id);
    edges.push({
      id,
      a,
      b,
      data: relation,
      similarity,
      strandCount: similarity
        ? 1
        : relation.relation === "contradicts" || relation.weight < 0.6
          ? 2
          : 3,
      angle: random() * Math.PI * 2,
      bend: similarity ? 0.05 : 0.09 + random() * 0.11,
      phase: random(),
    });
  }
  makeEdgeLayers();
  step = 0;
  if (firstGraph && nodes.size) {
    firstGraph = false;
    fit();
  }
  document.getElementById("node-count").textContent =
    `${nodes.size} nodes · ${edges.length} links in view`;
  if (selectedKey && nodes.has(selectedKey)) focus(selectedKey, false);
}

/* ------------------------------------------------------------------ behaviour */

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
  edgesDirty = true;
  step++;
  if (step === 180) fit();
  if (step > 80 && energy / list.length < 0.002) step = 281;
}
function illuminate(keys) {
  const any = keys.size > 0;
  const glow = halo?.geometry.attributes.aGlow;
  for (const n of nodes.values()) {
    const active = keys.has(n.key);
    n.level = active ? n.baseLevel * 1.45 : n.baseLevel * (any ? 0.35 : 1);
    n.alpha = any && !active ? 0.28 : 0.95;
    n.mesh.material.uniforms.uOpacity.value = n.alpha;
    if (glow) glow.array[n.index] = active ? n.glow * 1.5 : any ? n.glow * 0.15 : n.glow;
    if (n.ring)
      n.ring.material.opacity = active ? 0.85 : any ? 0.06 : n.ring.userData.baseOpacity;
    if (n.core)
      n.core.material.color
        .copy(n.core.userData.baseColor)
        .multiplyScalar(any && !active ? 0.25 : 1);
    for (const child of n.mesh.children)
      if (child.userData.isLabel)
        child.material.opacity = any && !active ? 0.15 : 0.85;
  }
  if (glow) glow.needsUpdate = true;
  if (!strands) return;
  const focus = strands.geometry.attributes.aFocus;
  const sparkFocus = sparks.geometry.attributes.aFocus;
  for (const e of edges) {
    const level = any ? (keys.has(e.a.key) && keys.has(e.b.key) ? 2.4 : 0.08) : 1;
    focus.array.fill(level, e.vStart, e.vStart + e.vCount);
    if (e.spark >= 0) sparkFocus.array[e.spark] = level;
  }
  focus.needsUpdate = true;
  sparkFocus.needsUpdate = true;
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
function clearFocus() {
  selectedKey = null;
  illuminate(new Set());
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
  pixelScale.value =
    renderer.getDrawingBufferSize(new THREE.Vector2()).y /
    (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
}

/* ------------------------------------------------------------------ frame loop */

function animateNodes(dt) {
  const worldPerPixelAtOne =
    (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / viewport.clientHeight;
  const haloPosition = halo?.geometry.attributes.position;
  const haloSize = halo?.geometry.attributes.aSize;
  const spin = dt * motion.value;
  for (const node of nodes.values()) {
    // Keep real data nodes legible when the graph is fitted into a narrow panel.
    // Importance still controls relative size; zoom never erases nodes.
    const worldPerPixel = camera.position.distanceTo(node.pos) * worldPerPixelAtOne;
    const pixels =
      node.type === "source"
        ? 5
        : node.type === "crystal"
          ? 3.5 + node.weight * 3
          : 2 + node.weight * 2;
    const scale = Math.max(1, (pixels * worldPerPixel) / node.radius);
    const size = node.radius * scale;
    node.mesh.scale.setScalar(size);
    node.mesh.rotation.y += node.spin * spin;
    if (node.ring) node.ring.rotation.z += 0.5 * spin;
    if (node.core) {
      node.core.rotation.y -= 0.8 * spin;
      node.core.rotation.x += 0.3 * spin;
    }
    node.boost += ((hover === node ? 1.35 : 1) - node.boost) * Math.min(1, dt * 10);
    node.mesh.material.uniforms.uIntensity.value = node.level * node.boost;
    if (haloPosition) {
      node.pos.toArray(haloPosition.array, node.index * 3);
      haloSize.array[node.index] = size * HALO[node.type] * node.boost;
    }
    for (const child of node.mesh.children) {
      if (!child.userData.isLabel) continue;
      child.visible =
        viewport.clientWidth > 520 ||
        hover?.key === node.key ||
        selectedKey === node.key;
      child.scale.set((240 * worldPerPixel) / size, (30 * worldPerPixel) / size, 1);
      child.position.y = -1.4 - (14 * worldPerPixel) / size; // clear of the orb's glow
    }
  }
  if (haloPosition) {
    haloPosition.needsUpdate = true;
    haloSize.needsUpdate = true;
  }
}

function init() {
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: document.getElementById("brain-canvas"),
      antialias: true,
    });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    scene = new THREE.Scene();
    scene.background = vignette();
    camera = new THREE.PerspectiveCamera(48, 1, 0.1, 1000);
    camera.position.set(8, 14, 75);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.065;
    controls.autoRotate = !reducedMotion;
    controls.autoRotateSpeed = 0.18;
    controls.minDistance = 6;
    controls.maxDistance = 400;
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(
      new UnrealBloomPass(new THREE.Vector2(1, 1), 0.85, 0.55, 0.5),
    );
    composer.addPass(new OutputPass());
    graph = new THREE.Group();
    scene.add(graph);
    initialized = true;
    resize();
    new ResizeObserver(resize).observe(viewport);
    // The node under a screen point, if any.
    const pick = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(pickables, false)[0];
      return hit ? nodes.get(hit.object.userData.key) : null;
    };
    renderer.domElement.addEventListener("pointermove", (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      hover = pick(event);
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
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) >= 5) return;
      // Hit-test the click itself: taps on touch screens have no hover beforehand.
      // A node shows its connections; empty space clears the focus (and the details panel).
      const node = pick(e);
      if (node) focus(node.key);
      else clearFocus();
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
      clearFocus();
      document.getElementById("node-inspector").classList.add("hidden");
      fit();
    });
    window.addEventListener("brain:data", (event) => build(event.detail));
    window.addEventListener("brain:focus", (event) => focus(event.detail));
    window.addEventListener("brain:clear", clearFocus);
    window.addEventListener("brain:highlight", (event) =>
      highlight(event.detail),
    );
    fetch("/api/graph")
      .then((r) => r.json())
      .then(build)
      .catch(() => {});
    function animate() {
      requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      if (!viewport.clientWidth || document.hidden) return;
      time.value += dt * motion.value;
      layout();
      if (edgesDirty) {
        writeEdges();
        edgesDirty = false;
      }
      controls.update();
      animateNodes(dt);
      composer.render();
    }
    animate();
  } catch (error) {
    document.getElementById("graph-fallback").classList.remove("hidden");
    console.warn("Graph unavailable:", error.message);
  }
}
init();
