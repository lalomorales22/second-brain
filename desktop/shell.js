'use strict';
// Second Brain — Electron shell (imported by main.js).
// Starts the Python workspace server from the repo's virtual environment and shows it in a window.
// This is a dev-machine wrapper: it does not bundle Python or the models.

const { app, BrowserWindow, dialog, shell } = require('electron');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');

const HOST = '127.0.0.1';
const PREFERRED_PORT = 8487;
const REPO = path.resolve(__dirname, '..');
const PYTHON = process.platform === 'win32'
  ? path.join(REPO, '.venv', 'Scripts', 'python.exe')
  : path.join(REPO, '.venv', 'bin', 'python');
// SECOND_BRAIN_DB overrides the database (assets are stored beside it), e.g. for a scratch copy.
const DB = process.env.SECOND_BRAIN_DB || path.join(REPO, '.data', 'memory.db');
const OLLAMA_TAGS = 'http://127.0.0.1:11434/api/tags';

let server = null;   // the Python server ChildProcess
let serverLog = [];  // last lines it printed, for the error dialog
let quitting = false;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function urlOk(url, timeoutMs = 2000) {
  return new Promise((resolve) => {
    const request = http.get(url, { timeout: timeoutMs }, (response) => {
      response.resume();
      resolve(response.statusCode < 500);
    });
    request.on('timeout', () => { request.destroy(); resolve(false); });
    request.on('error', () => resolve(false));
  });
}

async function waitFor(url, seconds, stillWorth = () => true) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (await urlOk(url)) return true;
    if (!stillWorth()) return false;
    await sleep(500);
  }
  return false;
}

// Start the Ollama desktop app if nothing answers on its port. Best effort.
async function ensureOllama() {
  if (await urlOk(OLLAMA_TAGS)) return true;
  try {
    if (process.platform === 'win32') {
      const exe = path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Ollama', 'ollama app.exe');
      if (!fs.existsSync(exe)) return false;
      spawn(exe, [], { detached: true, stdio: 'ignore' }).unref();
    } else if (process.platform === 'darwin') {
      if (!fs.existsSync('/Applications/Ollama.app')) return false;
      spawn('open', ['-a', 'Ollama'], { detached: true, stdio: 'ignore' }).unref();
    } else {
      return false;
    }
  } catch (error) {
    return false;
  }
  return waitFor(OLLAMA_TAGS, 20);
}

// The preferred port when it is free, otherwise any free port.
function pickPort(preferred) {
  const probe = (port) => new Promise((resolve) => {
    const socket = net.createServer();
    socket.once('error', () => resolve(null));
    socket.listen(port, HOST, () => {
      const chosen = socket.address().port;
      socket.close(() => resolve(chosen));
    });
  });
  return probe(preferred).then((port) => (port !== null ? port : probe(0)));
}

function spawnServer(port) {
  const args = ['-m', 'semantic_gravity_memory.api.server', '--db', DB, '--host', HOST, '--port', String(port)];
  server = spawn(PYTHON, args, { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  const remember = (chunk) => {
    const text = chunk.toString();
    process.stdout.write(text);
    serverLog = serverLog.concat(text.split(/\r?\n/)).filter(Boolean).slice(-40);
  };
  server.stdout.on('data', remember);
  server.stderr.on('data', remember);
  server.on('exit', (code, signal) => {
    if (quitting) return;
    quitting = true;
    dialog.showErrorBox('Second Brain stopped', `The local server exited (${signal || code}).\n\n${serverLog.join('\n')}`);
    app.quit();
  });
  return server;
}

function killServer() {
  if (!server || server.exitCode !== null) return;
  if (process.platform === 'win32') {
    // child.kill() alone can leave the Python process behind on Windows; take the whole tree down.
    spawnSync('taskkill', ['/pid', String(server.pid), '/t', '/f'], { stdio: 'ignore', windowsHide: true });
  } else {
    server.kill('SIGTERM');
  }
}

function createWindow(url) {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'Second Brain',
    backgroundColor: '#0b0b0f',
    autoHideMenuBar: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true },
  });
  win.on('page-title-updated', (event) => event.preventDefault());
  // Links that open a new window: the app's own pages (e.g. an original file) get an app
  // window; anything else (e.g. "Watch on YouTube") goes to the system browser.
  win.webContents.setWindowOpenHandler(({ url: target }) => {
    if (target.startsWith(url)) return { action: 'allow' };
    if (/^https?:\/\//i.test(target)) shell.openExternal(target);
    return { action: 'deny' };
  });
  win.loadURL(url);
  return win;
}

async function main() {
  if (!fs.existsSync(PYTHON)) {
    dialog.showErrorBox('Second Brain', `Python environment not found at\n${PYTHON}\n\nRun start.bat (or start.sh) once to create it.`);
    app.quit();
    return;
  }
  if (!(await ensureOllama())) {
    console.warn('Ollama is not running; embeddings and chat will be unavailable until it starts.');
  }
  const port = await pickPort(PREFERRED_PORT);
  spawnServer(port);
  const url = `http://${HOST}:${port}/`;
  const ready = await waitFor(url + 'api/config', 60, () => server.exitCode === null);
  if (!ready) {
    quitting = true;
    killServer();
    dialog.showErrorBox('Second Brain could not start',
      `The local server did not answer at ${url}\n\n${serverLog.join('\n') || '(no output)'}`);
    app.quit();
    return;
  }
  createWindow(url);
}

app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => { quitting = true; killServer(); });
process.on('exit', killServer);
process.on('uncaughtException', (error) => {
  console.error(error);
  quitting = true;
  killServer();
  app.exit(1);
});

// Entry point used by main.js. (Electron's require.main is its own bootstrap, so the
// usual "require.main === module" guard cannot tell the entry script from a library import.)
function run() {
  // One app at a time: launching again (e.g. typing second-brain twice) brings the open
  // window forward instead of starting a second server on the same database.
  if (!app.requestSingleInstanceLock()) {
    app.quit();
    return;
  }
  app.on('second-instance', () => {
    const win = BrowserWindow.getAllWindows()[0];
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
  });
  app.whenReady().then(main);
}

module.exports = {
  HOST, PREFERRED_PORT, PYTHON, DB,
  urlOk, waitFor, ensureOllama, pickPort, spawnServer, killServer, createWindow, main, run,
  getServer: () => server,
  setQuitting: (value) => { quitting = value; },
};
