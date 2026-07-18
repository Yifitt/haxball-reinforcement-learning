import assert from "node:assert/strict";
import { once } from "node:events";
import test from "node:test";

import WebSocket from "ws";

import { createBridgeServer } from "../bridge/websocket_server.js";
import { message } from "../bridge/protocol.js";

const send = (socket, type, fields = {}) =>
  socket.send(JSON.stringify(message(type, fields)));

const makeInbox = (socket) => {
  const messages = [];
  const waiters = [];
  socket.on("message", (raw) => {
    const payload = JSON.parse(raw.toString());
    const index = waiters.findIndex(({ predicate }) => predicate(payload));
    if (index >= 0) {
      const [{ resolve, timeout }] = waiters.splice(index, 1);
      clearTimeout(timeout);
      resolve(payload);
    } else {
      messages.push(payload);
    }
  });
  return (predicate, timeoutMilliseconds = 2_000) => {
    const index = messages.findIndex(predicate);
    if (index >= 0) return Promise.resolve(messages.splice(index, 1)[0]);
    return new Promise((resolve, reject) => {
      const waiter = { predicate, resolve, timeout: null };
      waiter.timeout = setTimeout(() => {
        const waiterIndex = waiters.indexOf(waiter);
        if (waiterIndex >= 0) waiters.splice(waiterIndex, 1);
        reject(new Error("matching message timeout"));
      }, timeoutMilliseconds);
      waiters.push(waiter);
    });
  };
};

const connectRole = async (port, role) => {
  const socket = new WebSocket(`ws://127.0.0.1:${port}`);
  const receiveWhere = makeInbox(socket);
  await once(socket, "open");
  send(socket, "hello", { role });
  await receiveWhere((payload) => payload.type === "hello" && payload.accepted === true);
  return { socket, receiveWhere };
};

const beginRole = async (port, role) => {
  const socket = new WebSocket(`ws://127.0.0.1:${port}`);
  const receiveWhere = makeInbox(socket);
  await once(socket, "open");
  send(socket, "hello", { role });
  return { socket, receiveWhere };
};

const configurePublicLobby = async (port, bridge) => {
  const hostConnection = await connectRole(port, "host");
  const browserConnection = await connectRole(port, "browser");
  send(hostConnection.socket, "room_status", {
    lifecycle: "waiting_for_players", room_url: "public-room",
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { team: 1, client: "controlled" },
  });
  while (!bridge.readiness.state.controlled_red) {
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  return { hostConnection, browserConnection };
};

test("explicit public bridge completes an early Python hello once after host readiness", async (t) => {
  const bridge = createBridgeServer({ port: 0, startupMode: "public_human_queue" });
  await once(bridge.server, "listening");
  const port = bridge.server.address().port;
  const connections = [];
  t.after(async () => {
    for (const { socket } of connections) socket.close();
    const closed = once(bridge.server, "close");
    bridge.shutdown();
    await closed;
  });

  const initial = bridge.readiness.snapshot();
  assert.equal(initial.startup_mode, "public_human_queue");
  assert.equal(initial.public_room, true);
  assert.equal(initial.human_opponent, true);
  assert.equal(initial.opponent_control_required, false);

  const agentConnection = await beginRole(port, "agent");
  connections.push(agentConnection);
  while (!bridge.readiness.state.python_connected) {
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  assert.equal(bridge.readiness.state.python_protocol_ready, true);
  await assert.rejects(
    agentConnection.receiveWhere((payload) => payload.type === "hello", 30),
    /matching message timeout/,
  );

  const publicConnections = await configurePublicLobby(port, bridge);
  connections.push(publicConnections.hostConnection, publicConnections.browserConnection);
  const hello = await agentConnection.receiveWhere((payload) => payload.type === "hello");
  assert.equal(hello.accepted, true);
  const lobbyReady = await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.barrier_ready === true);
  assert.deepEqual(lobbyReady.missing, []);
  assert.ok(!lobbyReady.required.includes("opponent_player"));
  assert.ok(!lobbyReady.required.includes("active_game_state"));

  send(agentConnection.socket, "hello", { role: "agent" });
  await assert.rejects(
    agentConnection.receiveWhere((payload) => payload.type === "hello", 30),
    /matching message timeout/,
  );
  send(agentConnection.socket, "readiness", {
    startup_mode: "public_human_queue",
    python_protocol_ready: true,
    public_room: false,
    human_opponent: false,
    opponent_control_required: true,
  });
  const conflict = await agentConnection.receiveWhere((payload) =>
    payload.type === "error" && payload.code === "startup_config_conflict");
  assert.equal(conflict.fatal, true);
  assert.equal(bridge.readiness.snapshot().public_room, true);
  assert.equal(bridge.readiness.snapshot().startup_mode, "public_human_queue");

  send(publicConnections.hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { team: 2, client: "opponent" },
  });
  send(publicConnections.hostConnection.socket, "room_status", { lifecycle: "active_episode" });
  const gameplayBarrier = await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.game_running === true);
  assert.deepEqual(gameplayBarrier.missing, [
    "controlled_game_surface", "controlled_input", "active_game_state",
  ]);
});

test("explicit public bridge acknowledges immediately when host is ready first", async (t) => {
  const bridge = createBridgeServer({ port: 0, startupMode: "public_human_queue" });
  await once(bridge.server, "listening");
  const port = bridge.server.address().port;
  const publicConnections = await configurePublicLobby(port, bridge);
  t.after(async () => {
    publicConnections.hostConnection.socket.close();
    publicConnections.browserConnection.socket.close();
    const closed = once(bridge.server, "close");
    bridge.shutdown();
    await closed;
  });

  const agentConnection = await beginRole(port, "agent");
  const hello = await agentConnection.receiveWhere((payload) => payload.type === "hello");
  assert.equal(hello.accepted, true);
  const ready = await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.barrier_ready === true);
  assert.equal(ready.startup_mode, "public_human_queue");
  agentConnection.socket.close();
});

test("early state and action are nonfatal until the readiness barrier opens", async (t) => {
  const bridge = createBridgeServer({ port: 0 });
  await once(bridge.server, "listening");
  const port = bridge.server.address().port;
  const connections = await Promise.all(
    ["host", "agent", "browser"].map((role) => connectRole(port, role)),
  );
  const [hostConnection, agentConnection, browserConnection] = connections;
  const host = hostConnection.socket;
  const agent = agentConnection.socket;
  const browser = browserConnection.socket;
  t.after(async () => {
    const closed = once(bridge.server, "close");
    bridge.shutdown();
    await closed;
  });

  send(agent, "readiness", { python_protocol_ready: true, opponent_control_required: false });
  send(host, "room_status", { lifecycle: "waiting_for_players", room_url: "private" });
  send(host, "room_status", { lifecycle: "player_joined", player: { id: 1, team: 1 } });
  send(host, "room_status", { lifecycle: "player_joined", player: { id: 2, team: 2 } });
  send(host, "state", { tick_id: 1, game_active: true });
  await agentConnection.receiveWhere((payload) => payload.type === "state" && payload.tick_id === 1);

  send(agent, "action", { tick_id: 1, client: "controlled", action: 4 });
  const rejection = await agentConnection.receiveWhere((payload) => payload.type === "error");
  assert.equal(rejection.code, "not_ready");
  assert.equal(rejection.fatal, false);
  assert.equal(rejection.operation, "apply_action");
  assert.equal(rejection.component, "controlled_browser");
  assert.deepEqual(rejection.missing, [
    "controlled_game_surface", "opponent_game_surface", "controlled_input",
  ]);
  assert.equal(bridge.metrics.preReadyActions, 1);
  assert.ok(bridge.metrics.preReadyStates >= 1);

  send(browser, "readiness", {
    controlled_surface_ready: true,
    opponent_surface_ready: true,
    controlled_input_ready: true,
  });
  const ready = await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.barrier_ready === true);
  assert.deepEqual(ready.missing, []);
  const refresh = await hostConnection.receiveWhere((payload) =>
    payload.type === "state_request" && payload.reason === "readiness_barrier_open");
  assert.equal(refresh.lifecycle_id, 0);
  send(browser, "readiness", { controlled_input_ready: true });
  assert.equal(bridge.readiness.isReady(), true);

  send(host, "state", { tick_id: 2, game_active: true });
  await agentConnection.receiveWhere((payload) => payload.type === "state" && payload.tick_id === 2);
  send(agent, "action", { tick_id: 2, client: "controlled", action: 4 });
  const routed = await browserConnection.receiveWhere((payload) => payload.type === "action");
  assert.equal(routed.client, "controlled");
  assert.equal(routed.action, 4);

  send(agent, "readiness", { opponent_control_required: true });
  send(browser, "readiness", { opponent_input_ready: true });
  await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.barrier_ready === true &&
    payload.opponent_control_required === true);
  send(host, "state", { tick_id: 3, game_active: true });
  await agentConnection.receiveWhere((payload) => payload.type === "state" && payload.tick_id === 3);
  send(agent, "action", { tick_id: 3, client: "opponent", action: 3 });
  const opponentRouted = await browserConnection.receiveWhere((payload) =>
    payload.type === "action" && payload.client === "opponent");
  assert.equal(opponentRouted.action, 3);

  browser.close();
  await once(browser, "close");
  const fatal = await agentConnection.receiveWhere((payload) => payload.code === "component_disconnected");
  assert.equal(fatal.fatal, true);
  assert.equal(fatal.component, "controlled_browser");

});

test("public-room browser disconnect is recoverable for the checkpoint controller", async (t) => {
  const bridge = createBridgeServer({ port: 0 });
  await once(bridge.server, "listening");
  const port = bridge.server.address().port;
  const [hostConnection, agentConnection, browserConnection] = await Promise.all(
    ["host", "agent", "browser"].map((role) => connectRole(port, role)),
  );
  t.after(async () => {
    const closed = once(bridge.server, "close");
    bridge.shutdown();
    await closed;
  });

  send(agentConnection.socket, "readiness", {
    python_protocol_ready: true,
    opponent_control_required: false,
    human_opponent: true,
    public_room: true,
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "waiting_for_players", room_url: "public-room",
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { id: 1, team: 1, client: "controlled" },
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { id: 2, team: 2, client: "opponent" },
  });
  send(browserConnection.socket, "readiness", {
    controlled_surface_ready: true,
    controlled_input_ready: true,
  });
  send(hostConnection.socket, "state", { tick_id: 1, game_active: true });
  await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.barrier_ready === true);

  browserConnection.socket.close();
  await once(browserConnection.socket, "close");
  const recoverable = await agentConnection.receiveWhere((payload) =>
    payload.code === "component_disconnected");
  assert.equal(recoverable.component, "controlled_browser");
  assert.equal(recoverable.fatal, false);

  const replacement = await connectRole(port, "browser");
  send(replacement.socket, "readiness", {
    controlled_surface_ready: true,
    controlled_input_ready: true,
  });
  const reconnectRefresh = await hostConnection.receiveWhere((payload) =>
    payload.type === "state_request" && payload.reason === "readiness_barrier_open");
  assert.equal(reconnectRefresh.lifecycle_id, 0);
  replacement.socket.close();
});

test("public queue infrastructure handshake succeeds before a human joins", async (t) => {
  const bridge = createBridgeServer({ port: 0 });
  await once(bridge.server, "listening");
  const port = bridge.server.address().port;
  const [hostConnection, agentConnection, browserConnection] = await Promise.all(
    ["host", "agent", "browser"].map((role) => connectRole(port, role)),
  );
  t.after(async () => {
    const closed = once(bridge.server, "close");
    bridge.shutdown();
    await closed;
  });

  send(agentConnection.socket, "readiness", {
    python_protocol_ready: true,
    opponent_control_required: false,
    human_opponent: true,
    public_room: true,
    startup_mode: "public_human_queue",
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "waiting_for_players", room_url: "public-room",
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { id: 1, team: 1, client: "controlled" },
  });
  const lobbyReady = await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.startup_mode === "public_human_queue" &&
    payload.barrier_ready === true);
  assert.deepEqual(lobbyReady.missing, []);
  assert.ok(!lobbyReady.required.includes("private_room"));
  assert.ok(!lobbyReady.required.includes("opponent_player"));
  assert.ok(!lobbyReady.required.includes("active_game_state"));

  send(hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { id: 2, team: 2, client: "opponent" },
  });
  send(hostConnection.socket, "room_status", { lifecycle: "active_episode" });
  const waitingForActiveMatch = await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.startup_mode === "public_human_queue" &&
    payload.barrier_ready === false && payload.missing.includes("active_game_state"));
  assert.deepEqual(waitingForActiveMatch.missing, [
    "controlled_game_surface", "controlled_input", "active_game_state",
  ]);
  send(browserConnection.socket, "readiness", {
    controlled_surface_ready: true, controlled_input_ready: true,
  });
  send(hostConnection.socket, "state", { tick_id: 1, game_active: true });
  const matchReady = await agentConnection.receiveWhere((payload) =>
    payload.type === "readiness" && payload.startup_mode === "public_human_queue" &&
    payload.barrier_ready === true && payload.game_running === true);
  assert.deepEqual(matchReady.missing, []);
});

test("active Red watchdog requests one fresh snapshot after an action stall", async (t) => {
  const bridge = createBridgeServer({ port: 0 });
  await once(bridge.server, "listening");
  const port = bridge.server.address().port;
  const [hostConnection, agentConnection, browserConnection] = await Promise.all(
    ["host", "agent", "browser"].map((role) => connectRole(port, role)),
  );
  t.after(async () => {
    const closed = once(bridge.server, "close");
    bridge.shutdown();
    await closed;
  });
  send(agentConnection.socket, "readiness", {
    python_protocol_ready: true, opponent_control_required: false,
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "waiting_for_players", room_url: "offline-room",
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { team: 1, client: "controlled" },
  });
  send(hostConnection.socket, "room_status", {
    lifecycle: "player_joined", player: { team: 2, client: "opponent" },
  });
  send(browserConnection.socket, "readiness", {
    controlled_surface_ready: true, opponent_surface_ready: true,
    controlled_input_ready: true,
  });
  send(hostConnection.socket, "state", {
    tick_id: 11, lifecycle_id: 5, game_active: true,
    controlled: { team: 1 },
  });
  const request = await hostConnection.receiveWhere((payload) =>
    payload.type === "state_request" && payload.reason === "controller_watchdog", 2_000);
  assert.equal(request.lifecycle_id, 5);
  const reset = await agentConnection.receiveWhere((payload) =>
    payload.type === "reset" && payload.reason === "controller_watchdog");
  assert.equal(reset.lifecycle_id, 5);
});
