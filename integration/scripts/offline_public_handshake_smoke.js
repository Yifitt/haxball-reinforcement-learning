import assert from "node:assert/strict";
import { once } from "node:events";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import WebSocket from "ws";

import { createBridgeServer } from "../bridge/websocket_server.js";
import { message } from "../bridge/protocol.js";

const send = (socket, type, fields = {}) => {
  socket.send(JSON.stringify(message(type, fields)));
};

const connectRole = async (port, role) => {
  console.log(
    `local_connection_attempt: component=offline_${role} transport=websocket ` +
    `host=127.0.0.1 port=${port} attempt=1`,
  );
  const socket = new WebSocket(`ws://127.0.0.1:${port}`);
  await once(socket, "open");
  const accepted = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`${role} hello timeout`)), 2_000);
    socket.on("message", (raw) => {
      const payload = JSON.parse(raw.toString());
      if (payload.type === "hello" && payload.accepted === true) {
        clearTimeout(timeout);
        resolve(payload);
      }
    });
  });
  send(socket, "hello", { role });
  await accepted;
  return socket;
};

const root = fileURLToPath(new URL("../../", import.meta.url));
const python = process.env.PYTHON ?? "python";
const bridge = createBridgeServer({ port: 0, startupMode: "public_human_queue" });
await once(bridge.server, "listening");
const port = bridge.server.address().port;

const pythonSource = `
import json, sys
from integration.controller.client import BridgeClient
client = BridgeClient(sys.argv[1], human_opponent=True, public_room=True)
client.connect()
snapshot = client.wait_until_ready(timeout=5.0)
print(json.dumps({
    "mode": snapshot["startup_mode"],
    "ready": snapshot["barrier_ready"],
    "public_room": snapshot["public_room"],
    "human_opponent": snapshot["human_opponent"],
    "opponent_control_required": snapshot["opponent_control_required"],
    "missing": snapshot["missing"],
}))
client.close()
`;
const controller = spawn(
  python,
  ["-c", pythonSource, `ws://127.0.0.1:${port}`],
  {
    cwd: root,
    env: {
      ...process.env,
      PYTHONPATH: root,
      WS_PROXY: "http://127.0.0.1:1",
      NO_PROXY: "",
      no_proxy: "",
    },
    stdio: ["ignore", "pipe", "pipe"],
  },
);
let stdout = "";
let stderr = "";
controller.stdout.on("data", (chunk) => { stdout += chunk; });
controller.stderr.on("data", (chunk) => { stderr += chunk; });

const controllerDeadline = Date.now() + 2_000;
while (!bridge.readiness.state.python_connected && Date.now() < controllerDeadline) {
  await new Promise((resolve) => setTimeout(resolve, 5));
}
assert.equal(bridge.readiness.state.python_connected, true);
assert.equal(bridge.readiness.snapshot().startup_mode, "public_human_queue");

const host = await connectRole(port, "host");
const browser = await connectRole(port, "browser");
send(host, "room_status", { lifecycle: "waiting_for_players", room_url: "offline-public" });
send(host, "room_status", {
  lifecycle: "player_joined", player: { team: 1, client: "controlled" },
});

const [exitCode] = await once(controller, "close");
if (exitCode !== 0) throw new Error(stderr || `Python controller exited ${exitCode}`);
const outputLines = stdout.trim().split("\n").filter(Boolean);
for (const line of outputLines.slice(0, -1)) console.log(line);
const report = JSON.parse(outputLines.at(-1));
assert.deepEqual(report, {
  mode: "public_human_queue",
  ready: true,
  public_room: true,
  human_opponent: true,
  opponent_control_required: false,
  missing: [],
});
assert.equal(bridge.readiness.state.opponent_joined, false);
assert.equal(bridge.readiness.state.game_running, false);

host.close();
browser.close();
const closed = once(bridge.server, "close");
bridge.shutdown();
await closed;
console.log(JSON.stringify({
  offline_public_handshake: true,
  python_connected_before_host: true,
  empty_lobby_ready: true,
  startup_mode: report.mode,
}));
