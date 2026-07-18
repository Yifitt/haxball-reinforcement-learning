import assert from "node:assert/strict";
import test from "node:test";

import { ActionGate, BridgeMetrics, message, parseMessage } from "../bridge/protocol.js";
import { ReadinessTracker } from "../bridge/readiness.js";
import { startupConfigForMode } from "../bridge/config.js";

test("protocol serialization and parsing", () => {
  const payload = message("action", { tick_id: 3, client: "opponent", action: 17 });
  assert.deepEqual(parseMessage(JSON.stringify(payload)), payload);
  assert.throws(() => parseMessage("bad"), /invalid JSON/);
  assert.throws(
    () => parseMessage('{"protocol_version":2,"type":"hello"}'),
    /unsupported protocol/,
  );
});

test("action gate rejects stale, duplicate, invalid, and not-ready actions", () => {
  const readiness = new ReadinessTracker();
  readiness.update({
    host_connected: true, room_created: true,
    controlled_joined: true, opponent_joined: true,
    opponent_surface_ready: true,
    python_connected: true, python_protocol_ready: true,
    active_game_state: true,
  });
  const gate = new ActionGate(readiness);
  gate.observeState({ tick_id: 10, game_active: true });
  const notReady = gate.accept({ tick_id: 10, client: "controlled", action: 1 });
  assert.equal(notReady.reason, "not_ready");
  assert.equal(notReady.diagnostic.operation, "apply_action");
  assert.equal(notReady.diagnostic.component, "controlled_browser");
  assert.ok(notReady.diagnostic.missing.includes("controlled_game_surface"));
  readiness.update({
    host_connected: true,
    room_created: true,
    controlled_joined: true,
    opponent_joined: true,
    controlled_surface_ready: true,
    opponent_surface_ready: true,
    controlled_input_ready: true,
    python_connected: true,
    python_protocol_ready: true,
    active_game_state: true,
  });
  assert.equal(gate.accept({ tick_id: 9, client: "controlled", action: 1 }).reason, "stale_tick");
  assert.equal(gate.accept({ tick_id: 10, client: "controlled", action: 18 }).reason, "invalid_action");
  assert.equal(gate.accept({ tick_id: 10, client: "controlled", action: 1 }).ok, true);
  assert.equal(gate.accept({ tick_id: 10, client: "controlled", action: 1 }).reason, "duplicate_action");
  assert.equal(gate.accept({ tick_id: 10, client: "wat", action: 1 }).reason, "unknown_client");
});

test("readiness tracks independent components and duplicate updates are idempotent", () => {
  const readiness = new ReadinessTracker();
  readiness.update({
    host_connected: true,
    room_created: true,
    controlled_joined: true,
    opponent_joined: true,
    python_connected: true,
    python_protocol_ready: true,
    active_game_state: true,
  });
  assert.equal(readiness.snapshot().barrier_ready, false);
  assert.deepEqual(
    readiness.snapshot().missing,
    ["controlled_game_surface", "opponent_game_surface", "controlled_input"],
  );
  assert.equal(readiness.update({ controlled_surface_ready: true }), true);
  assert.equal(readiness.update({ controlled_surface_ready: true }), false);
  readiness.update({ opponent_surface_ready: true });
  readiness.update({ controlled_input_ready: true });
  assert.equal(readiness.snapshot().barrier_ready, true);
  readiness.update({ opponent_control_required: true });
  assert.deepEqual(readiness.snapshot().missing, ["opponent_input"]);
  readiness.update({ opponent_input_ready: true });
  assert.equal(readiness.snapshot().barrier_ready, true);
});

test("human lobby defers game surface and input requirements until match start", () => {
  const readiness = new ReadinessTracker();
  readiness.update({
    human_opponent: true,
    host_connected: true, room_created: true,
    controlled_joined: true, opponent_joined: false,
    controlled_surface_ready: true, controlled_input_ready: true,
    opponent_surface_ready: false, opponent_input_ready: false,
    python_connected: true, python_protocol_ready: true,
    active_game_state: true,
  });
  assert.deepEqual(readiness.snapshot().missing, ["opponent_player"]);
  assert.ok(!readiness.snapshot().required.includes("controlled_game_surface"));
  assert.ok(!readiness.snapshot().required.includes("controlled_input"));
  assert.ok(!readiness.snapshot().required.includes("active_game_state"));
  readiness.update({ opponent_joined: true });
  assert.equal(readiness.snapshot().barrier_ready, true);
  readiness.update({
    game_running: true,
    controlled_surface_ready: false,
    controlled_input_ready: false,
    active_game_state: false,
  });
  assert.deepEqual(readiness.snapshot().missing, [
    "controlled_game_surface", "controlled_input", "active_game_state",
  ]);
  assert.ok(!readiness.snapshot().required.includes("opponent_game_surface"));
  assert.ok(!readiness.snapshot().required.includes("opponent_input"));
});

test("public human queue handshake is ready before a human joins", () => {
  const readiness = new ReadinessTracker();
  readiness.update({
    public_room: true,
    human_opponent: true,
    host_connected: true,
    room_created: true,
    browser_connected: true,
    controlled_joined: true,
    controlled_red: false,
    opponent_joined: false,
    python_connected: true,
    python_protocol_ready: true,
  });
  assert.deepEqual(readiness.snapshot().missing, ["controlled_player"]);
  readiness.update({ controlled_red: true });
  const lobby = readiness.snapshot();
  assert.equal(lobby.startup_mode, "public_human_queue");
  assert.equal(lobby.barrier_ready, true);
  assert.deepEqual(lobby.required, [
    "host", "public_room", "controlled_browser", "controlled_player", "python_controller",
  ]);
  assert.ok(!lobby.required.includes("private_room"));
  assert.ok(!lobby.required.includes("opponent_player"));
  assert.ok(!lobby.required.includes("active_game_state"));

  readiness.update({ opponent_joined: true, opponent_blue: true });
  assert.equal(readiness.snapshot().barrier_ready, true);
  readiness.update({ game_running: true });
  assert.deepEqual(readiness.snapshot().missing, [
    "controlled_game_surface", "controlled_input", "active_game_state",
  ]);
  readiness.update({
    controlled_surface_ready: true,
    controlled_input_ready: true,
    active_game_state: true,
  });
  assert.equal(readiness.snapshot().barrier_ready, true);
});

test("explicit public mode is resolved and cannot fall back to false source flags", () => {
  const readiness = new ReadinessTracker(startupConfigForMode("public_human_queue"));
  const initial = readiness.snapshot();
  assert.equal(initial.startup_mode, "public_human_queue");
  assert.equal(initial.public_room, true);
  assert.equal(initial.human_opponent, true);
  assert.equal(initial.opponent_control_required, false);
  readiness.update({
    public_room: false,
    human_opponent: false,
    opponent_control_required: true,
  });
  const afterDefaults = readiness.snapshot();
  assert.equal(afterDefaults.startup_mode, "public_human_queue");
  assert.equal(afterDefaults.public_room, true);
  assert.equal(afterDefaults.human_opponent, true);
  assert.equal(afterDefaults.opponent_control_required, false);
});

test("explicit private startup modes retain their readiness contracts", () => {
  const stationary = new ReadinessTracker(startupConfigForMode("stationary_opponent"));
  assert.equal(stationary.snapshot().startup_mode, "stationary_opponent");
  assert.ok(stationary.snapshot().required.includes("opponent_game_surface"));
  assert.ok(!stationary.snapshot().required.includes("opponent_input"));

  const dual = new ReadinessTracker(startupConfigForMode("dual_control"));
  assert.equal(dual.snapshot().startup_mode, "dual_control");
  assert.ok(dual.snapshot().required.includes("opponent_game_surface"));
  assert.ok(dual.snapshot().required.includes("opponent_input"));

  const human = new ReadinessTracker(startupConfigForMode("human_opponent"));
  assert.equal(human.snapshot().startup_mode, "human_opponent");
  assert.ok(human.snapshot().required.includes("opponent_player"));
  assert.ok(!human.snapshot().required.includes("opponent_game_surface"));
  assert.ok(!human.snapshot().required.includes("opponent_input"));
});

test("stale and duplicate action tracking is independent per client", () => {
  const readiness = new ReadinessTracker();
  readiness.update({
    host_connected: true, room_created: true,
    controlled_joined: true, opponent_joined: true,
    controlled_surface_ready: true, opponent_surface_ready: true,
    controlled_input_ready: true, opponent_input_ready: true,
    opponent_control_required: true,
    python_connected: true, python_protocol_ready: true,
    active_game_state: true,
  });
  const gate = new ActionGate(readiness);
  gate.observeState({ tick_id: 7, game_active: true });
  assert.equal(gate.accept({ tick_id: 7, client: "controlled", action: 4 }).ok, true);
  assert.equal(gate.accept({ tick_id: 7, client: "opponent", action: 3 }).ok, true);
  assert.equal(gate.accept({ tick_id: 7, client: "controlled", action: 4 }).reason, "duplicate_action");
  assert.equal(gate.accept({ tick_id: 7, client: "opponent", action: 3 }).reason, "duplicate_action");
});

test("stationary opponent rejection does not block controlled actions", () => {
  const readiness = new ReadinessTracker();
  readiness.update({
    host_connected: true, room_created: true,
    controlled_joined: true, opponent_joined: true,
    controlled_surface_ready: true, opponent_surface_ready: true,
    controlled_input_ready: true,
    python_connected: true, python_protocol_ready: true,
    active_game_state: true,
  });
  const gate = new ActionGate(readiness);
  gate.observeState({ tick_id: 12, game_active: true });
  const opponent = gate.accept({ tick_id: 12, client: "opponent", action: 3 });
  assert.equal(opponent.reason, "not_ready");
  assert.equal(opponent.diagnostic.client, "opponent");
  assert.ok(opponent.diagnostic.missing.includes("opponent_input"));
  assert.equal(gate.accept({ tick_id: 12, client: "controlled", action: 4 }).ok, true);
});

test("state tick ordering is monotonic", () => {
  const gate = new ActionGate();
  assert.equal(gate.observeState({ tick_id: 2, game_active: true }), true);
  assert.equal(gate.observeState({ tick_id: 2, game_active: true }), false);
  assert.equal(gate.observeState({ tick_id: 1, game_active: true }), false);
  assert.equal(gate.observeState({ tick_id: 3, game_active: true }), true);
});

test("new lifecycle accepts an identical opening tick and rejects stale callbacks", () => {
  const gate = new ActionGate();
  gate.setClientReady(true);
  assert.equal(gate.observeState({ tick_id: 7, lifecycle_id: 1, game_active: true }), true);
  assert.equal(gate.accept({
    tick_id: 7, lifecycle_id: 1, client: "controlled", action: 4,
  }).ok, true);
  gate.resetLifecycle(2);
  assert.equal(gate.observeState({ tick_id: 7, lifecycle_id: 2, game_active: true }), true);
  assert.equal(gate.accept({
    tick_id: 7, lifecycle_id: 1, client: "controlled", action: 4,
  }).reason, "stale_lifecycle");
  assert.equal(gate.accept({
    tick_id: 7, lifecycle_id: 2, client: "controlled", action: 4,
  }).ok, true);
});

test("bridge latency and rate reporting", () => {
  const metrics = new BridgeMetrics();
  metrics.startedAt = 1000;
  metrics.states = 40;
  metrics.actions = { controlled: 38, opponent: 34 };
  metrics.actionApplied = { controlled: 36, opponent: 32 };
  metrics.recordLatency("controlled", 8);
  metrics.recordLatency("controlled", 12);
  metrics.recordLatency("opponent", 15);
  metrics.recordPairedDifference(3);
  const report = metrics.report(3000);
  assert.equal(report.state_messages_per_second, 20);
  assert.equal(report.controlled_actions_per_second, 19);
  assert.equal(report.opponent_actions_per_second, 17);
  assert.equal(report.median_state_to_controlled_input_ms, 12);
  assert.equal(report.median_state_to_opponent_input_ms, 15);
  assert.equal(report.median_paired_application_difference_ms, 3);
});
