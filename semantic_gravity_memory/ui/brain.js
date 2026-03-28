/**
 * Second Brain — 3D Visualization
 *
 * Three.js scene: force-directed graph of memory crystals & entities,
 * bloom postprocessing, organic motion, activation animations.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// =========================================================================
// Constants
// =========================================================================

const SELF_STATE_COLORS = {
  builder:      0x4488ff,
  professional: 0x44ddff,
  founder:      0xffaa00,
  student:      0x44ff88,
  creative:     0xaa44ff,
  family:       0xff8844,
  personal:     0xff44aa,
  general:      0x8888aa,
};

const RELATION_COLORS = {
  mentions:         0x4488ff,
  co_occurred:      0x44cc66,
  temporal_cluster: 0xddaa33,
  contradicts:      0xff4466,
  crystallized_into: 0x6666aa,
  caused:           0x44aacc,
};

const API = '';  // Same origin

// =========================================================================
// State
// =========================================================================

let scene, camera, renderer, composer, controls, bloomPass;
let graphGroup;   // THREE.Group holding all graph objects
let clock = new THREE.Clock();

// Graph data
let nodes = [];        // { id, type, data, mesh, pos, vel, label }
let edges = [];        // { source, target, data, line }
let nodeMap = {};      // `crystal_3` or `entity_7` → node
let activeNodeIds = new Set();  // Nodes activated by last recall

// Raycasting
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();
let hoveredNode = null;
let selectedNode = null;

// Animation
let activationWaves = [];  // { center, startTime, maxRadius }
let isLayoutSettled = false;
let layoutIterations = 0;
const MAX_LAYOUT_ITERS = 200;
const IDLE_DRIFT_SPEED = 0.03;

// =========================================================================
// Init
// =========================================================================

let webglAvailable = false;

function init() {
  // UI and data loading work without WebGL — always set these up
  setupUI();
  loadModels();

  // Try to init 3D — if WebGL isn't available, the chat/API still works
  try {
    init3D();
    webglAvailable = true;
    loadGraph();
    setStatus('ready');
    animate();
  } catch (e) {
    console.warn('WebGL not available:', e.message);
    webglAvailable = false;
    setStatus('chat ready (3D disabled — no WebGL)');

    // Show a message in the viewport
    const vp = document.getElementById('viewport');
    const msg = document.createElement('div');
    msg.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#8888aa;text-align:center;font-size:14px;max-width:400px;line-height:1.6;';
    msg.innerHTML = `
      <div style="font-size:20px;margin-bottom:12px;">3D brain unavailable</div>
      <div>WebGL could not initialize on this device.</div>
      <div style="margin-top:8px;">Chat and memory still work — use the panel on the right.</div>
      <div style="margin-top:12px;color:#6666aa;font-size:12px;">
        To see the 3D brain, try:<br>
        &bull; Chrome with --ignore-gpu-blocklist flag<br>
        &bull; Firefox (often better WebGL on Linux)<br>
        &bull; Open from another device on your network
      </div>
    `;
    vp.style.position = 'relative';
    vp.appendChild(msg);
  }
}

function init3D() {
  const canvas = document.getElementById('brain-canvas');
  const vp = document.getElementById('viewport');

  // Renderer — this throws if WebGL is not available
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setSize(vp.clientWidth, vp.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ReinhardToneMapping;
  renderer.toneMappingExposure = 2.2;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x06060c);
  scene.fog = new THREE.FogExp2(0x06060c, 0.003);

  // Camera
  camera = new THREE.PerspectiveCamera(60, vp.clientWidth / vp.clientHeight, 0.1, 500);
  camera.position.set(0, 10, 50);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 5;
  controls.maxDistance = 400;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.3;

  // Lights
  scene.add(new THREE.AmbientLight(0x334466, 1.6));
  const point = new THREE.PointLight(0x4488ff, 2.5, 250);
  point.position.set(0, 30, 0);
  scene.add(point);
  const point2 = new THREE.PointLight(0xff8844, 1.2, 150);
  point2.position.set(-20, -10, 20);
  scene.add(point2);

  // Postprocessing
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(vp.clientWidth, vp.clientHeight),
    1.6,   // strength
    0.6,   // radius
    0.1    // threshold — lower so dim things still bloom
  );
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  // Graph container
  graphGroup = new THREE.Group();
  scene.add(graphGroup);

  // Background particles
  createStarfield();

  // Events
  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onClick);
}

// =========================================================================
// Starfield background
// =========================================================================

function createStarfield() {
  const geo = new THREE.BufferGeometry();
  const count = 3000;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i++) {
    positions[i] = (Math.random() - 0.5) * 500;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({
    size: 0.3,
    color: 0x5577aa,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true,
  });
  scene.add(new THREE.Points(geo, mat));
}

// =========================================================================
// Graph loading
// =========================================================================

async function loadGraph() {
  if (!webglAvailable) return;
  setStatus('loading graph...');
  try {
    const resp = await fetch(`${API}/api/graph?crystal_limit=100&entity_limit=50`);
    const data = await resp.json();
    buildGraph(data);
    setStatus(`${nodes.length} nodes, ${edges.length} edges`);
    document.getElementById('node-count').textContent =
      `${data.crystals.length} crystals \u00b7 ${data.entities.length} entities`;

    // If there's a last scene, highlight it
    if (data.last_scene && data.last_scene.crystals) {
      highlightScene(data.last_scene);
    }
  } catch (e) {
    setStatus('no data yet \u2014 ingest some memories');
    console.warn('graph load:', e);
  }
}

function buildGraph(data) {
  // Clear old
  clearGraph();

  const { crystals, entities, relations } = data;

  // Create crystal nodes
  for (const c of crystals) {
    const key = `crystal_${c.id}`;
    // Use REAL gravitational mass for visual weight — what you see is what the memory weighs
    const mass = Math.max(0.05, Math.min(1.0, c.grav_mass || 0));
    const strength = Math.max(0.2, Math.min(1.0, mass + (c.access_count || 0) * 0.02));
    const radius = 0.5 + strength * 1.5;
    const color = SELF_STATE_COLORS[c.self_state] || SELF_STATE_COLORS.general;

    // Emissive intensity is driven by gravitational mass — heavy = bright, light = dim
    const baseEmissive = 0.3 + mass * 0.9;

    const geo = new THREE.SphereGeometry(radius, 24, 16);
    const mat = new THREE.MeshStandardMaterial({
      color: color,
      emissive: color,
      emissiveIntensity: baseEmissive,
      roughness: 0.2,
      metalness: 0.3,
      transparent: true,
      opacity: 0.7 + mass * 0.3,
    });
    const mesh = new THREE.Mesh(geo, mat);

    // Ring for memory type
    if (c.memory_type === 'semantic') {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius + 0.15, 0.04, 8, 32),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 })
      );
      mesh.add(ring);
    } else {
      // Dashed ring for episodic — approximate with a thin torus
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius + 0.15, 0.02, 8, 16),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 })
      );
      mesh.add(ring);
    }

    // Contradiction halo
    if (c.contradiction_state === 'tension') {
      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(radius + 0.3, 16, 8),
        new THREE.MeshBasicMaterial({
          color: 0xff4466,
          transparent: true,
          opacity: 0.15,
          side: THREE.BackSide,
        })
      );
      halo.userData.isHalo = true;
      mesh.add(halo);
    }

    const pos = randomSpherePoint(20);
    mesh.position.copy(pos);
    mesh.userData = { key, type: 'crystal', data: c, baseEmissive: baseEmissive };
    graphGroup.add(mesh);

    const node = {
      id: c.id, type: 'crystal', key, data: c, mesh,
      pos: pos.clone(), vel: new THREE.Vector3(), radius, strength,
      gravMass: mass,
    };
    nodes.push(node);
    nodeMap[key] = node;
  }

  // Create entity nodes — brightness = real mention count + salience
  for (const e of entities) {
    const key = `entity_${e.id}`;
    const brightness = Math.max(0.15, Math.min(1.0, (e.mention_count || 0) * 0.06 + e.salience * 0.4));
    const radius = 0.25 + brightness * 0.45;
    const baseEmissive = 0.3 + brightness * 0.6;

    const geo = new THREE.IcosahedronGeometry(radius, 1);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xaaccee,
      emissive: 0x88bbdd,
      emissiveIntensity: baseEmissive,
      roughness: 0.3,
      metalness: 0.2,
      transparent: true,
      opacity: 0.6 + brightness * 0.4,
    });
    const mesh = new THREE.Mesh(geo, mat);

    const pos = randomSpherePoint(18);
    mesh.position.copy(pos);
    mesh.userData = { key, type: 'entity', data: e, baseEmissive: baseEmissive };
    graphGroup.add(mesh);

    const node = {
      id: e.id, type: 'entity', key, data: e, mesh,
      pos: pos.clone(), vel: new THREE.Vector3(), radius, strength: brightness,
      gravMass: brightness,
    };
    nodes.push(node);
    nodeMap[key] = node;
  }

  // Create edges
  for (const r of relations) {
    const srcKey = `${r.source_type}_${r.source_id}`;
    const tgtKey = `${r.target_type}_${r.target_id}`;
    const srcNode = nodeMap[srcKey];
    const tgtNode = nodeMap[tgtKey];
    if (!srcNode || !tgtNode) continue;

    const color = RELATION_COLORS[r.relation] || 0x5577aa;
    const opacity = Math.max(0.35, Math.min(0.9, r.weight * 0.9 + 0.2));

    const geo = new THREE.BufferGeometry().setFromPoints([
      srcNode.pos.clone(), tgtNode.pos.clone()
    ]);
    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity,
      linewidth: 1,
    });
    const line = new THREE.Line(geo, mat);
    line.userData = { relation: r.relation };
    graphGroup.add(line);

    edges.push({ source: srcNode, target: tgtNode, data: r, line, weight: r.weight });
  }

  // Reset layout
  isLayoutSettled = false;
  layoutIterations = 0;
}

function clearGraph() {
  // Remove all children from graph group
  while (graphGroup.children.length > 0) {
    const child = graphGroup.children[0];
    graphGroup.remove(child);
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
  }
  nodes = [];
  edges = [];
  nodeMap = {};
  activeNodeIds.clear();
}

// =========================================================================
// Force-directed layout (3D)
// =========================================================================

function stepLayout() {
  if (isLayoutSettled || nodes.length === 0) return;

  const repulsion = 60.0;
  const attraction = 0.004;
  const gravity = 0.002;
  const damping = 0.88;
  const minDist = 0.5;

  // Repulsion (O(n^2) — fine for <200 nodes)
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = a.pos.x - b.pos.x;
      const dy = a.pos.y - b.pos.y;
      const dz = a.pos.z - b.pos.z;
      const dist = Math.max(minDist, Math.sqrt(dx * dx + dy * dy + dz * dz));
      const force = repulsion / (dist * dist);
      const fx = force * dx / dist;
      const fy = force * dy / dist;
      const fz = force * dz / dist;
      a.vel.x += fx; a.vel.y += fy; a.vel.z += fz;
      b.vel.x -= fx; b.vel.y -= fy; b.vel.z -= fz;
    }
  }

  // Attraction along edges
  for (const e of edges) {
    const a = e.source, b = e.target;
    const dx = b.pos.x - a.pos.x;
    const dy = b.pos.y - a.pos.y;
    const dz = b.pos.z - a.pos.z;
    const dist = Math.max(minDist, Math.sqrt(dx * dx + dy * dy + dz * dz));
    const force = attraction * dist * (e.weight || 0.5);
    const fx = force * dx / dist;
    const fy = force * dy / dist;
    const fz = force * dz / dist;
    a.vel.x += fx; a.vel.y += fy; a.vel.z += fz;
    b.vel.x -= fx; b.vel.y -= fy; b.vel.z -= fz;
  }

  // Gravity toward center
  let totalMovement = 0;
  for (const n of nodes) {
    n.vel.x -= n.pos.x * gravity;
    n.vel.y -= n.pos.y * gravity;
    n.vel.z -= n.pos.z * gravity;

    // Damping
    n.vel.multiplyScalar(damping);

    // Update position
    n.pos.add(n.vel);
    n.mesh.position.copy(n.pos);

    totalMovement += n.vel.length();
  }

  // Update edge lines
  for (const e of edges) {
    const positions = e.line.geometry.attributes.position;
    positions.setXYZ(0, e.source.pos.x, e.source.pos.y, e.source.pos.z);
    positions.setXYZ(1, e.target.pos.x, e.target.pos.y, e.target.pos.z);
    positions.needsUpdate = true;
  }

  layoutIterations++;
  if (layoutIterations > MAX_LAYOUT_ITERS || totalMovement / nodes.length < 0.01) {
    isLayoutSettled = true;
  }
}

// =========================================================================
// Idle organic drift — position only, no fake glow
// =========================================================================

function applyIdleDrift(time) {
  if (!isLayoutSettled) return;
  for (const n of nodes) {
    // Gentle positional breathing — heavier nodes drift less (they're anchored)
    const mass = n.gravMass || 0.1;
    const driftScale = IDLE_DRIFT_SPEED * 0.02 * (1.0 - mass * 0.5);
    const t = time + n.id * 1.37;
    const dx = Math.sin(t * 0.3 + n.id) * driftScale;
    const dy = Math.cos(t * 0.2 + n.id * 0.7) * driftScale;
    const dz = Math.sin(t * 0.25 + n.id * 1.1) * driftScale;
    n.mesh.position.x = n.pos.x + dx;
    n.mesh.position.y = n.pos.y + dy;
    n.mesh.position.z = n.pos.z + dz;
  }
}

// =========================================================================
// Activation highlighting
// =========================================================================

function highlightScene(sceneData) {
  if (!webglAvailable) return;
  // Reset all nodes
  activeNodeIds.clear();
  for (const n of nodes) {
    const mat = n.mesh.material;
    mat.emissiveIntensity = n.mesh.userData.baseEmissive;
    mat.opacity = 0.9;
  }

  // Dim non-active, brighten active
  if (!sceneData.crystals || sceneData.crystals.length === 0) return;

  const activeCrystalIds = new Set(sceneData.crystals.map(c => c.id));
  const activeEntityIds = new Set((sceneData.entities || []).map(e => e.id));

  for (const n of nodes) {
    const isActive =
      (n.type === 'crystal' && activeCrystalIds.has(n.id)) ||
      (n.type === 'entity' && activeEntityIds.has(n.id));

    if (isActive) {
      activeNodeIds.add(n.key);
      n.mesh.material.emissiveIntensity = 1.8;
      n.mesh.material.opacity = 1.0;
    } else {
      n.mesh.material.emissiveIntensity = n.mesh.userData.baseEmissive * 0.5;
      n.mesh.material.opacity = 0.55;
    }
  }

  // Dim non-active edges
  for (const e of edges) {
    const srcActive = activeNodeIds.has(e.source.key);
    const tgtActive = activeNodeIds.has(e.target.key);
    if (srcActive && tgtActive) {
      e.line.material.opacity = Math.max(0.6, e.weight * 0.9);
    } else {
      e.line.material.opacity = 0.12;
    }
  }

  // Trigger activation wave from center of active cluster
  if (activeCrystalIds.size > 0) {
    const center = new THREE.Vector3();
    let count = 0;
    for (const n of nodes) {
      if (n.type === 'crystal' && activeCrystalIds.has(n.id)) {
        center.add(n.pos);
        count++;
      }
    }
    if (count > 0) {
      center.divideScalar(count);
      activationWaves.push({
        center: center.clone(),
        startTime: clock.getElapsedTime(),
        maxRadius: 40,
        duration: 2.5,
      });
    }
  }
}

function resetHighlight() {
  activeNodeIds.clear();
  for (const n of nodes) {
    n.mesh.material.emissiveIntensity = n.mesh.userData.baseEmissive;
    n.mesh.material.opacity = 0.9;
  }
  for (const e of edges) {
    e.line.material.opacity = Math.max(0.35, e.weight * 0.9 + 0.2);
  }
}

// =========================================================================
// Activation wave animation
// =========================================================================

function updateActivationWaves(time) {
  for (let i = activationWaves.length - 1; i >= 0; i--) {
    const wave = activationWaves[i];
    const elapsed = time - wave.startTime;
    const progress = elapsed / wave.duration;

    if (progress >= 1) {
      activationWaves.splice(i, 1);
      continue;
    }

    const radius = progress * wave.maxRadius;
    const fade = 1 - progress;

    for (const n of nodes) {
      const dist = n.pos.distanceTo(wave.center);
      if (Math.abs(dist - radius) < 2.0) {
        // Node is in the wavefront
        n.mesh.material.emissiveIntensity =
          n.mesh.userData.baseEmissive + fade * 0.8;
      }
    }
  }
}

// =========================================================================
// Pulse animation — ONLY for genuinely activated nodes
// =========================================================================

function pulseActiveNodes(time) {
  // When no recall is active, all nodes sit at their true gravitational glow.
  // No fake pulsing. What you see = what the memory actually weighs.

  if (activeNodeIds.size === 0) {
    // Idle state: every node glows at its real gravitational mass.
    // This IS the brain at rest — heavy memories burn bright, light ones are dim.
    // No animation, no faking. Pure data.
    return;
  }

  // Active state: only recalled nodes pulse. This is REAL — these are
  // the nodes the retrieval engine actually activated.
  for (const n of nodes) {
    if (!activeNodeIds.has(n.key)) continue;
    const pulse = Math.sin(time * 3 + n.id) * 0.3 + 0.3;
    n.mesh.material.emissiveIntensity = 1.5 + pulse;
  }

  // Contradiction halos only flicker when the crystal is actively recalled
  // and in tension — real conflict, not decoration
  for (const n of nodes) {
    if (n.type !== 'crystal' || n.data.contradiction_state !== 'tension') continue;
    if (!activeNodeIds.has(n.key)) continue;
    n.mesh.children.forEach(child => {
      if (child.userData && child.userData.isHalo) {
        child.material.opacity = 0.15 + Math.sin(time * 5) * 0.1;
      }
    });
  }
}

// =========================================================================
// Raycasting / hover / click
// =========================================================================

function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const meshes = nodes.map(n => n.mesh);
  const intersects = raycaster.intersectObjects(meshes, false);

  const tooltip = document.getElementById('tooltip');

  if (intersects.length > 0) {
    const mesh = intersects[0].object;
    const ud = mesh.userData;
    hoveredNode = nodeMap[ud.key];

    // Show tooltip
    tooltip.classList.remove('hidden');
    tooltip.style.left = (event.clientX - renderer.domElement.getBoundingClientRect().left + 16) + 'px';
    tooltip.style.top = (event.clientY - renderer.domElement.getBoundingClientRect().top - 10) + 'px';

    if (ud.type === 'crystal') {
      const c = ud.data;
      tooltip.innerHTML = `
        <div class="tt-title">${esc(c.title)}</div>
        <div class="tt-meta">
          ${c.self_state} &middot; ${c.memory_type} &middot; v${c.version}<br>
          access: ${c.access_count} &middot; confidence: ${c.confidence.toFixed(2)}<br>
          ${c.contradiction_state !== 'clean' ? '<span style="color:#ff4466">contradiction: ' + c.contradiction_state + '</span><br>' : ''}
          ${c.summary ? esc(c.summary.slice(0, 120)) + (c.summary.length > 120 ? '...' : '') : ''}
        </div>`;
    } else {
      const e = ud.data;
      tooltip.innerHTML = `
        <div class="tt-title">${esc(e.name)}</div>
        <div class="tt-meta">
          ${e.kind} &middot; mentions: ${e.mention_count} &middot; salience: ${e.salience.toFixed(2)}
        </div>`;
    }

    renderer.domElement.style.cursor = 'pointer';
  } else {
    hoveredNode = null;
    tooltip.classList.add('hidden');
    renderer.domElement.style.cursor = 'default';
  }
}

function onClick(event) {
  if (!hoveredNode) return;
  selectedNode = hoveredNode;
  const detail = document.getElementById('detail-json');
  detail.textContent = JSON.stringify(hoveredNode.data, null, 2);

  // Switch to details tab
  switchTab('details');
}

// =========================================================================
// Render loop
// =========================================================================

function animate() {
  requestAnimationFrame(animate);

  const time = clock.getElapsedTime();

  // Layout
  if (!isLayoutSettled) {
    // Run multiple iterations per frame for faster settling
    for (let i = 0; i < 3; i++) stepLayout();
  }

  // Animations
  applyIdleDrift(time);
  pulseActiveNodes(time);
  updateActivationWaves(time);

  // Controls
  controls.update();

  // Render
  composer.render();
}

// =========================================================================
// UI wiring
// =========================================================================

function setupUI() {
  // Tabs
  document.querySelectorAll('#panel-tabs .tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Chat
  document.getElementById('btn-send').addEventListener('click', sendChat);
  document.getElementById('btn-ingest').addEventListener('click', ingestOnly);
  document.getElementById('chat-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  // Top bar
  document.getElementById('btn-refresh-models').addEventListener('click', loadModels);
  document.getElementById('btn-consolidate').addEventListener('click', consolidate);
  document.getElementById('btn-stats').addEventListener('click', showStats);

  // Stats modal close
  document.querySelector('.modal-close').addEventListener('click', () => {
    document.getElementById('stats-modal').classList.add('hidden');
  });
  document.getElementById('stats-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) e.currentTarget.classList.add('hidden');
  });
}

function switchTab(name) {
  document.querySelectorAll('#panel-tabs .tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-content').forEach(c =>
    c.classList.toggle('active', c.id === `tab-${name}`));
}

// =========================================================================
// Chat
// =========================================================================

async function sendChat() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  appendChat('you', text, 'user');
  setStatus('recalling memories...');

  try {
    const chatModel = document.getElementById('chat-model').value;
    const resp = await fetch(`${API}/api/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, model: chatModel }),
    });

    // Handle streaming SSE response
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let answerText = '';
    let thinkingText = '';
    let isThinking = false;
    let lastScene = null;

    // Create the streaming message containers
    const history = document.getElementById('chat-history');

    // Thinking container (collapsible)
    const thinkDiv = document.createElement('div');
    thinkDiv.className = 'chat-msg thinking hidden';
    thinkDiv.innerHTML = `<div class="who">thinking</div><div class="think-toggle" onclick="this.parentElement.classList.toggle('collapsed')">reasoning...</div><div class="body think-body"></div>`;
    history.appendChild(thinkDiv);
    const thinkBody = thinkDiv.querySelector('.think-body');

    // Answer container
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg assistant';
    msgDiv.innerHTML = `<div class="who">assistant</div><div class="body"></div>`;
    history.appendChild(msgDiv);
    const msgBody = msgDiv.querySelector('.body');

    setStatus('streaming response...');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch { continue; }

        if (data.type === 'scene') {
          lastScene = data.scene;
          document.getElementById('scene-json').textContent =
            JSON.stringify(data.scene, null, 2);
          highlightScene(data.scene);
          setStatus('generating response...');

        } else if (data.type === 'thinking_start') {
          isThinking = true;
          thinkDiv.classList.remove('hidden');
          setStatus('model is thinking...');

        } else if (data.type === 'thinking') {
          thinkingText += data.content;
          thinkBody.textContent = thinkingText;
          history.scrollTop = history.scrollHeight;

        } else if (data.type === 'thinking_end') {
          isThinking = false;
          thinkDiv.classList.add('collapsed');
          setStatus('responding...');

        } else if (data.type === 'token') {
          answerText += data.content;
          msgBody.textContent = answerText;
          history.scrollTop = history.scrollHeight;

        } else if (data.type === 'error') {
          appendChat('system', `error: ${data.message}`, 'system');
          setStatus('error');
          return;

        } else if (data.type === 'done') {
          lastScene = data.scene || lastScene;
        }
      }
    }

    // Remove empty containers
    if (!thinkingText) thinkDiv.remove();
    if (!answerText) msgDiv.remove();

    // Reload graph and re-highlight
    if (webglAvailable) {
      await loadGraph();
      if (lastScene) highlightScene(lastScene);
    }

    setStatus('ready');
  } catch (e) {
    appendChat('system', `error: ${e.message}`, 'system');
    setStatus('error');
  }
}

async function ingestOnly() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  appendChat('you', text, 'user');
  setStatus('ingesting...');

  try {
    const resp = await fetch(`${API}/api/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json();
    appendChat('system', `ingested \u2192 crystal #${data.crystal_id}`, 'system');
    if (webglAvailable) await loadGraph();
    setStatus('ready');
  } catch (e) {
    appendChat('system', `error: ${e.message}`, 'system');
    setStatus('error');
  }
}

async function consolidate() {
  setStatus('consolidating...');
  try {
    const resp = await fetch(`${API}/api/consolidate`, { method: 'POST' });
    const data = await resp.json();
    appendChat('system', `consolidation done: ${JSON.stringify(data)}`, 'system');
    if (webglAvailable) await loadGraph();
    setStatus('ready');
  } catch (e) {
    setStatus('error');
  }
}

async function showStats() {
  try {
    const resp = await fetch(`${API}/api/stats`);
    const data = await resp.json();
    document.getElementById('stats-json').textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    document.getElementById('stats-json').textContent = 'could not load stats';
  }
  document.getElementById('stats-modal').classList.remove('hidden');
}

// =========================================================================
// Models
// =========================================================================

async function loadModels() {
  try {
    const resp = await fetch(`${API}/api/models`);
    const data = await resp.json();
    const chatSel = document.getElementById('chat-model');
    const embedSel = document.getElementById('embed-model');

    populateSelect(chatSel, data.models, guessDefault(data.models, ['gpt-oss', 'llama', 'mistral', 'gemma']));
    populateSelect(embedSel, data.models, guessDefault(data.models, ['minilm', 'embed', 'nomic']));
  } catch (e) {
    console.warn('could not fetch models:', e);
  }
}

function populateSelect(sel, models, defaultVal) {
  const current = sel.value;
  sel.innerHTML = '';
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  }
  if (current && models.includes(current)) {
    sel.value = current;
  } else if (defaultVal) {
    sel.value = defaultVal;
  }
}

function guessDefault(models, hints) {
  for (const hint of hints) {
    const match = models.find(m => m.toLowerCase().includes(hint));
    if (match) return match;
  }
  return models[0] || '';
}

// =========================================================================
// Helpers
// =========================================================================

function appendChat(who, text, cls) {
  const history = document.getElementById('chat-history');
  const div = document.createElement('div');
  div.className = `chat-msg ${cls}`;
  div.innerHTML = `<div class="who">${esc(who)}</div><div class="body">${esc(text)}</div>`;
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
}

function setStatus(text) {
  document.getElementById('status-text').textContent = text;
}

function esc(s) {
  if (!s) return '';
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function randomSpherePoint(radius) {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const r = radius * Math.cbrt(Math.random());
  return new THREE.Vector3(
    r * Math.sin(phi) * Math.cos(theta),
    r * Math.sin(phi) * Math.sin(theta),
    r * Math.cos(phi),
  );
}

function onResize() {
  if (!webglAvailable) return;
  const vp = document.getElementById('viewport');
  camera.aspect = vp.clientWidth / vp.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(vp.clientWidth, vp.clientHeight);
  composer.setSize(vp.clientWidth, vp.clientHeight);
}

// =========================================================================
// Launch
// =========================================================================

init();
