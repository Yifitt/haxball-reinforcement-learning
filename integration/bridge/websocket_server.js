import { pathToFileURL } from "node:url";

import { WebSocket, WebSocketServer } from "ws";

import {
  ActionGate,
  BridgeMetrics,
  message,
  parseMessage,
} from "./protocol.js";
import { ReadinessTracker } from "./readiness.js";
import { ControlledActionWatchdog } from "./action_watchdog.js";
import { bridgeConfigFromEnv, startupConfigForMode } from "./config.js";

function send(socket, payload) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

export function createBridgeServer({
  host = "127.0.0.1", port = 8765, startupMode = null,
} = {}) {
  const startupConfig = startupMode === null ? null : startupConfigForMode(startupMode);
  const server = new WebSocketServer({ host, port });
  const sockets = new Set();
  const agents = new Set();
  const browsers = new Set();
  const readiness = new ReadinessTracker(startupConfig);
  const gate = new ActionGate(readiness);
  const metrics = new BridgeMetrics();
  const actionWatchdog = new ControlledActionWatchdog();
  const stateArrival = new Map();
  const appliedByTick = new Map();
  let hostSocket = null;
  let lastRoomStatus = null;
  let lastRoomUrl = null;
  let shuttingDown = false;
  let lastPublishedReadiness = "";
  let lastBarrierReady = false;
  let stateRequestCount = 0;
  let watchdogRecoveryCount = 0;

  if (startupConfig) {
    console.log(
      `bridge_startup_config: mode=${startupConfig.startupMode} ` +
      `public_room=${startupConfig.publicRoom} ` +
      `human_opponent=${startupConfig.humanOpponent} ` +
      `opponent_control_required=${startupConfig.opponentControlRequired}`,
    );
  }

  const requestStateSnapshot = (reason, lifecycleId = gate.latestLifecycle) => {
    if (!hostSocket) return false;
    stateRequestCount += 1;
    send(hostSocket, message("state_request", {
      reason,
      lifecycle_id: lifecycleId,
      request_sequence: stateRequestCount,
      requested_timestamp: Date.now(),
    }));
    return true;
  };

  const broadcast = (targets, payload) => {
    for (const socket of targets) send(socket, payload);
  };

  const acknowledged = (targets) => new Set(
    [...targets].filter((socket) => socket.handshakeAcknowledged),
  );

  const acknowledgeHello = (socket) => {
    if (socket.handshakeAcknowledged) return false;
    socket.handshakeAcknowledged = true;
    send(socket, message("hello", { role: "bridge", accepted: true }));
    if (socket.role === "browser" && lastRoomUrl) {
      send(socket, message("room_status", {
        lifecycle: "waiting_for_players",
        room_url: lastRoomUrl,
      }));
    }
    if (lastRoomStatus) send(socket, lastRoomStatus);
    return true;
  };

  const reconcileStartupHandshakes = () => {
    if (readiness.explicitStartupMode !== "public_human_queue") return;
    const status = readiness.startupHandshakeStatus();
    if (status.missing.length !== 0) return;
    let completed = false;
    for (const socket of agents) completed = acknowledgeHello(socket) || completed;
    if (completed) {
      console.log(`startup_handshake: mode=${status.startup_mode} ready=true`);
    }
  };

  const publishReadiness = (force = false) => {
    const snapshot = readiness.snapshot();
    const serialized = JSON.stringify(snapshot);
    if (!force && serialized === lastPublishedReadiness) return;
    lastPublishedReadiness = serialized;
    reconcileStartupHandshakes();
    broadcast(acknowledged(sockets), message("readiness", snapshot));
    console.log(
      `startup_barrier: mode=${snapshot.startup_mode} ` +
      `source_public_room=${snapshot.public_room} ` +
      `source_human_opponent=${snapshot.human_opponent} ` +
      `source_opponent_control_required=${snapshot.opponent_control_required} ` +
      `ready=${snapshot.barrier_ready} ` +
      `missing=${snapshot.missing.length ? snapshot.missing.join(",") : "none"}`,
    );
    if (snapshot.barrier_ready && !lastBarrierReady && gate.gameActive) {
      actionWatchdog.readinessOpened();
      requestStateSnapshot("readiness_barrier_open");
    }
    lastBarrierReady = snapshot.barrier_ready;
  };

  const shutdown = () => {
    if (shuttingDown) return;
    shuttingDown = true;
    const payload = message("shutdown", { reason: "requested" });
    broadcast(sockets, payload);
    clearInterval(heartbeat);
    clearInterval(actionWatchdogTimer);
    setTimeout(() => {
      for (const socket of sockets) socket.close(1001, "shutdown");
      server.close();
    }, 50);
  };

  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.isAlive = true;
    socket.role = null;
    socket.handshakeAcknowledged = false;
    socket.on("pong", () => {
      socket.isAlive = true;
    });

    socket.on("message", (raw) => {
      let payload;
      try {
        payload = parseMessage(raw);
      } catch (error) {
        send(socket, message("error", { code: "bad_message", detail: error.message }));
        return;
      }

      if (payload.type === "hello") {
        if (!["host", "agent", "browser"].includes(payload.role)) {
          send(socket, message("error", { code: "bad_role" }));
          return;
        }
        if (socket.role !== null) return;
        socket.role = payload.role;
        if (payload.role === "host") {
          hostSocket = socket;
          readiness.update({ host_connected: true });
        }
        if (payload.role === "agent") {
          agents.add(socket);
          readiness.update({ python_connected: true, python_protocol_ready: true });
          console.log("python_controller_connected: true");
        }
        if (payload.role === "browser") {
          browsers.add(socket);
          readiness.update({ browser_connected: true });
        }
        if (payload.role !== "agent" || readiness.explicitStartupMode !== "public_human_queue") {
          acknowledgeHello(socket);
        }
        publishReadiness(true);
        return;
      }

      if (!socket.role) {
        send(socket, message("error", { code: "hello_required" }));
        return;
      }

      if (payload.type === "room_status" && socket.role === "host") {
        lastRoomStatus = payload;
        if (payload.room_url) lastRoomUrl = payload.room_url;
        if (readiness.observeRoomStatus(payload)) publishReadiness();
        broadcast(acknowledged(new Set([...agents, ...browsers])), payload);
      } else if (payload.type === "client_status" && socket.role === "browser") {
        gate.setClientReady(payload.ready);
        if (readiness.update({
          controlled_surface_ready: payload.ready === true,
          controlled_input_ready: payload.ready === true,
        })) publishReadiness();
        broadcast(acknowledged(agents), payload);
      } else if (payload.type === "readiness" && socket.role === "browser") {
        if (readiness.update({
          controlled_surface_ready: payload.controlled_surface_ready,
          opponent_surface_ready: payload.opponent_surface_ready,
          controlled_input_ready: payload.controlled_input_ready,
          opponent_input_ready: payload.opponent_input_ready,
        })) publishReadiness();
      } else if (payload.type === "readiness" && socket.role === "agent") {
        const conflict = readiness.startupTelemetryConflict(payload);
        if (conflict) {
          send(socket, message("error", {
            code: "startup_config_conflict", component: "python_controller",
            field: conflict, fatal: true,
          }));
          return;
        }
        if (readiness.update({
          python_protocol_ready: payload.python_protocol_ready,
          opponent_control_required: payload.opponent_control_required,
          human_opponent: payload.human_opponent,
          public_room: payload.public_room,
        })) {
          publishReadiness();
        }
      } else if (payload.type === "state" && socket.role === "host") {
        if (!gate.observeState(payload)) return;
        actionWatchdog.observeState(payload);
        if (!readiness.isReady()) metrics.preReadyStates += 1;
        if (readiness.update({
          game_running: payload.game_active === true,
          active_game_state: payload.game_active === true,
        })) publishReadiness();
        metrics.states += 1;
        stateArrival.set(payload.tick_id, Date.now());
        while (stateArrival.size > 256) stateArrival.delete(stateArrival.keys().next().value);
        broadcast(acknowledged(agents), payload);
      } else if (payload.type === "action" && socket.role === "agent") {
        const accepted = gate.accept(payload);
        if (!accepted.ok) {
          const nonfatalRejections = new Set([
            "not_ready", "stale_tick", "stale_lifecycle", "duplicate_action", "unknown_tick",
          ]);
          const fields = {
            code: accepted.reason,
            tick_id: payload.tick_id,
            client: payload.client,
            fatal: !nonfatalRejections.has(accepted.reason),
            ...accepted.diagnostic,
          };
          if (accepted.reason === "not_ready") {
            metrics.preReadyActions += 1;
            console.log(
              `bridge_not_ready: operation=${fields.operation} ` +
              `missing=${fields.missing.join(",")}`,
            );
          }
          if (["controlled", "opponent"].includes(payload.client)) {
            metrics.rejected[payload.client] += 1;
          }
          send(socket, message("error", fields));
          return;
        }
        metrics.actions[payload.client] += 1;
        broadcast(browsers, payload);
      } else if (payload.type === "action_applied" && socket.role === "browser") {
        if (!["controlled", "opponent"].includes(payload.client)) return;
        metrics.actionApplied[payload.client] += 1;
        actionWatchdog.observeActionApplied(payload);
        const started = stateArrival.get(payload.tick_id);
        if (started !== undefined) metrics.recordLatency(payload.client, Date.now() - started);
        const pair = appliedByTick.get(payload.tick_id) ?? {};
        pair[payload.client] = payload.applied_timestamp ?? Date.now();
        appliedByTick.set(payload.tick_id, pair);
        if (pair.controlled !== undefined && pair.opponent !== undefined) {
          metrics.recordPairedDifference(Math.abs(pair.controlled - pair.opponent));
          appliedByTick.delete(payload.tick_id);
        }
        while (appliedByTick.size > 256) appliedByTick.delete(appliedByTick.keys().next().value);
        broadcast(acknowledged(agents), payload);
      } else if (payload.type === "reset" && socket.role === "host") {
        gate.resetLifecycle(payload.lifecycle_id);
        actionWatchdog.reset(payload.lifecycle_id);
        stateArrival.clear();
        appliedByTick.clear();
        if (readiness.update({ active_game_state: false })) publishReadiness();
        broadcast(acknowledged(new Set([...agents, ...browsers])), payload);
      } else if (payload.type === "state_request" &&
                 (socket.role === "agent" || socket.role === "browser")) {
        requestStateSnapshot(payload.reason ?? "component_request", payload.lifecycle_id);
      } else if (payload.type === "shutdown") {
        shutdown();
      }
    });

    socket.on("close", () => {
      sockets.delete(socket);
      agents.delete(socket);
      browsers.delete(socket);
      const readinessWasReached = readiness.everReady;
      if (socket === hostSocket) {
        hostSocket = null;
        readiness.update({ host_connected: false, game_running: false, active_game_state: false });
      }
      if (socket.role) metrics.disconnects += 1;
      if (socket.role === "browser") {
        gate.setClientReady(false);
        readiness.update({
          browser_connected: false,
          controlled_surface_ready: false,
          opponent_surface_ready: false,
          controlled_input_ready: false,
          opponent_input_ready: false,
        });
      }
      if (socket.role === "agent") {
        if (agents.size === 0) {
          readiness.update({ python_connected: false, python_protocol_ready: false });
        }
      }
      if (socket.role) publishReadiness();
      if (!shuttingDown && readinessWasReached && ["host", "browser"].includes(socket.role)) {
        const component = socket.role === "browser" ? "controlled_browser" : "headless_host";
        broadcast(acknowledged(agents), message("error", {
          code: "component_disconnected",
          operation: "state_action_loop",
          component,
          fatal: socket.role === "browser" ? !readiness.state.public_room : true,
        }));
      }
    });
  });

  const heartbeat = setInterval(() => {
    for (const socket of sockets) {
      if (!socket.isAlive) {
        socket.terminate();
        continue;
      }
      socket.isAlive = false;
      socket.ping();
    }
  }, 10_000);
  heartbeat.unref();

  const actionWatchdogTimer = setInterval(() => {
    const recovery = actionWatchdog.poll(readiness.isReady());
    if (!recovery) return;
    watchdogRecoveryCount += 1;
    console.log(
      `controller_watchdog reason=${recovery.reason} recovery_count=${watchdogRecoveryCount}`,
    );
    gate.resetActions();
    stateArrival.clear();
    appliedByTick.clear();
    broadcast(acknowledged(new Set([...agents, ...browsers])), message("reset", {
      reason: "controller_watchdog",
      lifecycle_id: recovery.lifecycleId,
      tick_id: gate.latestTick,
      reset_timestamp: Date.now(),
    }));
    requestStateSnapshot("controller_watchdog", recovery.lifecycleId);
  }, 100);
  actionWatchdogTimer.unref();

  server.on("listening", () => {
    const address = server.address();
    console.log(`bridge_listening: ws://${address.address}:${address.port}`);
  });
  server.on("close", () => {
    console.log(`bridge_metrics: ${JSON.stringify(metrics.report())}`);
  });

  return { server, gate, metrics, readiness, actionWatchdog, requestStateSnapshot, shutdown };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const startupConfig = bridgeConfigFromEnv();
  createBridgeServer({
    host: process.env.HAXBALL_BRIDGE_HOST ?? "127.0.0.1",
    port: Number(process.env.HAXBALL_BRIDGE_PORT ?? 8765),
    startupMode: startupConfig.startupMode,
  });
}
