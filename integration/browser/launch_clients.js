import { chromium } from "playwright-core";
import WebSocket from "ws";

import { message, parseMessage } from "../bridge/protocol.js";
import { closeBrowserResources, joinConfiguredClients } from "./client_startup.js";
import { CLIENT_ROLES, DualInputRouter } from "./dual_input.js";
import { captureJoinDiagnostics } from "./join_diagnostics.js";
import { InputController } from "./input_controller.js";
import {
  formatPointerInterceptionDiagnostics,
  prepareKeyboardInput,
} from "./input_focus.js";
import { joinRoom } from "./controlled_client.js";
import { findGameSurfaceAcrossContext } from "./game_surface.js";
import { BotReconnectCoordinator } from "./bot_reconnect.js";

const bridgeUrl = process.env.HAXBALL_BRIDGE_URL ?? "ws://127.0.0.1:8765";
const controlledNickname = process.env.HAXBALL_CONTROLLED_NICK ?? "RL-Agent";
const opponentNickname = process.env.HAXBALL_OPPONENT_NICK ?? "Scripted-Opponent";
const opponentPolicy = process.env.HAXBALL_OPPONENT_POLICY ?? "stationary";
const humanOpponent = process.env.HAXBALL_HUMAN_OPPONENT === "1";
const publicRoom = process.env.HAXBALL_PUBLIC_ROOM === "1";
const deferredHumanSurface = humanOpponent;
const opponentControlEnabled = !humanOpponent && opponentPolicy !== "stationary";
const executablePath = process.env.CHROME_PATH ?? "/usr/bin/google-chrome";
const headed = process.argv.includes("--headed");

async function connectWithRetries(url, attempts = 3) {
  const endpoint = new URL(url);
  const port = endpoint.port || (endpoint.protocol === "wss:" ? "443" : "80");
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    console.log(
      `local_connection_attempt: component=controlled_browser transport=websocket ` +
      `host=${endpoint.hostname} port=${port} attempt=${attempt}`,
    );
    try {
      return await new Promise((resolve, reject) => {
        const socket = new WebSocket(url);
        socket.once("open", () => resolve(socket));
        socket.once("error", reject);
      });
    } catch (error) {
      lastError = error;
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 250 * attempt));
    }
  }
  throw lastError;
}

const socket = await connectWithRetries(bridgeUrl);
socket.send(JSON.stringify(message("hello", { role: "browser" })));

let requestedRoomUrl = process.env.HAXBALL_ROOM_URL || null;
const pendingMessages = [];
const joinedPlayers = new Map();
const hostJoinWaiters = new Map();
const reportedHostJoins = new Set();
let wakeMessage = null;
let activeGameObserved = false;
const activeGameWaiters = new Set();
socket.on("message", (raw) => {
  try {
    const payload = parseMessage(raw);
    if (payload.type === "room_status" && payload.lifecycle === "player_joined" && payload.player) {
      joinedPlayers.set(payload.player.name, payload.player);
      const waiters = hostJoinWaiters.get(payload.player.name) ?? [];
      waiters.forEach((resolveWaiter) => resolveWaiter(payload.player));
      hostJoinWaiters.delete(payload.player.name);
    }
    if (payload.type === "room_status" && payload.lifecycle === "player_left" && payload.player) {
      if (joinedPlayers.get(payload.player.name)?.id === payload.player.id) {
        joinedPlayers.delete(payload.player.name);
        activeGameObserved = false;
      }
    }
    if (payload.type === "room_status" && payload.lifecycle === "active_episode") {
      activeGameObserved = true;
      activeGameWaiters.forEach((resolveWaiter) => resolveWaiter());
      activeGameWaiters.clear();
    }
    if (payload.type === "room_status" && ["match_ended", "waiting_for_players"].includes(payload.lifecycle)) {
      activeGameObserved = false;
    }
    pendingMessages.push(payload);
    wakeMessage?.();
  } catch (error) {
    console.error(`browser_protocol_error: ${error.message}`);
  }
});
socket.on("close", () => {
  pendingMessages.push(message("shutdown", { reason: "bridge_disconnected" }));
  wakeMessage?.();
});

const nextMessage = async (timeoutMilliseconds = 60_000) => {
  if (pendingMessages.length) return pendingMessages.shift();
  await new Promise((resolveMessage, reject) => {
    const timeout = setTimeout(() => reject(new Error("bridge message timeout")), timeoutMilliseconds);
    wakeMessage = () => {
      clearTimeout(timeout);
      wakeMessage = null;
      resolveMessage();
    };
  });
  return pendingMessages.shift();
};

const clientIdForNickname = (nickname) => nickname === controlledNickname ? "controlled" : "opponent";
const reportHostJoin = (clientId, player) => {
  const key = `${clientId}:${player.id}`;
  if (reportedHostJoins.has(key)) return;
  reportedHostJoins.add(key);
  console.log(`host_join_status: client=${clientId} joined=true player_id=${player.id}`);
};
const waitForHostJoin = (nickname, timeoutMilliseconds = 30_000, { signal } = {}) => {
  const clientId = clientIdForNickname(nickname);
  if (joinedPlayers.has(nickname)) {
    const player = joinedPlayers.get(nickname);
    reportHostJoin(clientId, player);
    return Promise.resolve(player);
  }
  return new Promise((resolveJoin, reject) => {
    let settled = false;
    const removeWaiter = () => {
      const waiters = hostJoinWaiters.get(nickname) ?? [];
      const remaining = waiters.filter((waiter) => waiter !== wrappedResolve);
      if (remaining.length) hostJoinWaiters.set(nickname, remaining);
      else hostJoinWaiters.delete(nickname);
    };
    const finishReject = (error, { report = false } = {}) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      removeWaiter();
      signal?.removeEventListener("abort", onAbort);
      if (report) console.log(`host_join_status: client=${clientId} joined=false`);
      reject(error);
    };
    const timeout = setTimeout(() => {
      finishReject(
        new Error(`${nickname} nickname was submitted, but actual host room entry was not confirmed within ${timeoutMilliseconds / 1000} seconds`),
        { report: true },
      );
    }, timeoutMilliseconds);
    const wrappedResolve = (player) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      reportHostJoin(clientId, player);
      resolveJoin(player);
    };
    const onAbort = () => finishReject(new Error(`${nickname} host join wait cancelled`));
    const waiters = hostJoinWaiters.get(nickname) ?? [];
    waiters.push(wrappedResolve);
    hostJoinWaiters.set(nickname, waiters);
    if (signal?.aborted) onAbort();
    else signal?.addEventListener("abort", onAbort, { once: true });
  });
};
const waitForActiveGame = (timeoutMilliseconds = null) => {
  if (activeGameObserved) return Promise.resolve();
  return new Promise((resolveActive, reject) => {
    const timeout = timeoutMilliseconds === null ? null : setTimeout(() => {
      activeGameWaiters.delete(wrappedResolve);
      reject(new Error("active game was not observed before controlled input preparation"));
    }, timeoutMilliseconds);
    const wrappedResolve = () => {
      if (timeout !== null) clearTimeout(timeout);
      resolveActive();
    };
    activeGameWaiters.add(wrappedResolve);
  });
};

const waitForLobbyActivation = async (client) => {
  while (!activeGameObserved) {
    if (socket.readyState !== WebSocket.OPEN) throw new Error("bridge WebSocket closed in lobby");
    if (!joinedPlayers.has(controlledNickname)) throw new Error("controlled player left the lobby");
    if (!client?.context || client.context.pages().length === 0 ||
        client.context.pages().every((page) => page.isClosed())) {
      throw new Error("controlled browser page/context closed in lobby");
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 250));
  }
};

while (!requestedRoomUrl) {
  const payload = await nextMessage();
  if (payload.type === "room_status" && payload.room_url) requestedRoomUrl = payload.room_url;
  if (payload.type === "shutdown") throw new Error("bridge shut down before room became ready");
}

const browser = await chromium.launch({
  executablePath,
  headless: !headed,
  args: ["--disable-features=WebRtcHideLocalIpsWithMdns"],
});
let controlledContext = await browser.newContext();
let opponentContext = humanOpponent ? null : await browser.newContext();
const inputRouter = new DualInputRouter();
const { controllers, ready: inputReady } = inputRouter;
let closing = false;
let reconnectCoordinator = null;
let firstAppliedLifecycle = null;
let firstActionAppliedCount = 0;

const send = (type, fields = {}) => {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message(type, fields)));
  }
};
const sendReadiness = (fields) => send("readiness", fields);
const releaseAllControllers = async () => {
  await inputRouter.releaseAll();
};
const resetAllControllers = async () => {
  await inputRouter.resetAll();
  firstAppliedLifecycle = null;
};
const setClientInputReady = (clientId, ready) => {
  inputReady[clientId] = ready;
  inputRouter.setReady(clientId, ready);
  sendReadiness({ [`${clientId}_input_ready`]: ready });
};
const prepareClientController = async (clientId, client) => {
  const controller = new InputController(client.page.keyboard, {
    clientId,
    ready: false,
  });
  inputRouter.register(clientId, controller, false);
  await controller.releaseAll();
  try {
    const focusResult = await prepareKeyboardInput(client, client.page.keyboard);
    controller.setReady(true, focusResult);
    setClientInputReady(clientId, true);
    console.log(
      `input_overlay_diagnostics: client=${clientId} ` +
      formatPointerInterceptionDiagnostics(focusResult.pointerDiagnostics),
    );
    console.log(
      `input_focus_status: client=${clientId} ready=true method=${focusResult.method} ` +
      `active_element=${focusResult.activeElement} keyboard_verified=${focusResult.keyboardVerified}`,
    );
    console.log(`input_status: client=${clientId} ready=true`);
    return controller;
  } catch (error) {
    controller.setReady(false);
    setClientInputReady(clientId, false);
    await controller.releaseAll().catch(() => {});
    const artifact = await captureJoinDiagnostics(client.context, {
      clientId: `${clientId}-input`,
      stage: "PREPARING_KEYBOARD_INPUT",
      error,
    });
    console.error(
      `client_input_diagnostics: client=${clientId} persisted=false ` +
      `artifact_failures=${artifact.failures.length}`,
    );
    throw error;
  }
};
const initializeGameController = async (clientId, client) => {
  const surface = await findGameSurfaceAcrossContext(client.context, { timeout: 45_000 });
  Object.assign(client, surface);
  const box = await surface.canvas.boundingBox();
  const size = box ? `${Math.round(box.width)}x${Math.round(box.height)}` : "unknown";
  console.log(`client_surface_status: client=${clientId} ready=true size=${size}`);
  sendReadiness({ [`${clientId}_surface_ready`]: true });
  await prepareClientController(clientId, client);
};
let closePromise = null;
const close = (reason = "shutdown") => {
  if (closePromise) return closePromise;
  closePromise = (async () => {
  closing = true;
  reconnectCoordinator?.stop();
  inputReady.controlled = false;
  inputReady.opponent = false;
  for (const controller of Object.values(controllers)) controller?.setReady(false);
  await resetAllControllers();
  sendReadiness({
    controlled_surface_ready: false,
    opponent_surface_ready: false,
    controlled_input_ready: false,
    opponent_input_ready: false,
  });
  send("client_status", { ready: false, reason });
  socket.close();
  await closeBrowserResources(browser, [controlledContext, opponentContext].filter(Boolean));
  })();
  return closePromise;
};
process.once("SIGINT", () => { void close("signal"); });
process.once("SIGTERM", () => { void close("signal"); });

try {
  let { controlled, opponent } = await joinConfiguredClients(
    joinRoom,
    [controlledContext, requestedRoomUrl, controlledNickname, {
      clientId: "controlled",
      waitForHostJoin,
      requireGameSurface: !deferredHumanSurface,
      onSurfaceReady: () => sendReadiness({ controlled_surface_ready: true }),
    }],
    [opponentContext, requestedRoomUrl, opponentNickname, {
      clientId: "opponent",
      waitForHostJoin,
      onSurfaceReady: () => sendReadiness({ opponent_surface_ready: true }),
    }],
    { humanOpponent },
  );
  console.log(
    `host_players_confirmed: controlled_id=${controlled.player.id} ` +
    `opponent_id=${opponent?.player?.id ?? "human"}`,
  );
  if (deferredHumanSurface) {
    send("client_status", {
      ready: false,
      lobby_ready: true,
      controlled_nickname: controlledNickname,
    });
    console.log("bot_lobby_ready");
    await waitForLobbyActivation(controlled);
    await initializeGameController("controlled", controlled);
  } else {
    await waitForActiveGame(20_000);
    await prepareClientController("controlled", controlled);
  }
  if (opponentControlEnabled) {
    await prepareClientController("opponent", opponent);
  } else if (!humanOpponent) {
    console.log("input_status: client=opponent ready=false mode=stationary");
  }
  send("client_status", { ready: true, controlled_nickname: controlledNickname });
  console.log("clients_ready: true");

  const unhealthyReason = async (client) => {
    if (!client || !client.context || !client.page || client.page.isClosed()) return "closed_page";
    try {
      if (!client.context.pages().includes(client.page)) return "closed_context";
    } catch {
      return "closed_context";
    }
    if (publicRoom && !joinedPlayers.has(controlledNickname)) return "missing_bot_player_id";
    try {
      const surface = await findGameSurfaceAcrossContext(client.context, { timeout: 1_000 });
      Object.assign(client, surface);
    } catch {
      return "missing_game_surface";
    }
    const text = await client.frame.locator("body").innerText().catch(() => "");
    return /connection closed|room has been closed|failed to connect|you were kicked/i.test(text)
      ? "browser_disconnected" : null;
  };

  const resetClients = async (reason) => {
    inputReady.controlled = false;
    inputReady.opponent = false;
    for (const controller of Object.values(controllers)) controller?.setReady(false);
    await resetAllControllers();
    sendReadiness({
      controlled_surface_ready: false,
      opponent_surface_ready: false,
      controlled_input_ready: false,
      opponent_input_ready: false,
    });
    send("client_status", { ready: false, reason });
    await controlledContext?.close().catch(() => {});
    await opponentContext?.close().catch(() => {});
    controlledContext = null;
    opponentContext = null;
    controlled = null;
    opponent = null;
    joinedPlayers.delete(controlledNickname);
    joinedPlayers.delete(opponentNickname);
    for (let index = pendingMessages.length - 1; index >= 0; index -= 1) {
      if (pendingMessages[index].type === "action") pendingMessages.splice(index, 1);
    }
    reportedHostJoins.clear();
    activeGameObserved = false;
  };

  const reconnect = async (attempt) => {
    controlledContext = await browser.newContext();
    opponentContext = humanOpponent ? null : await browser.newContext();
    try {
      ({ controlled, opponent } = await joinConfiguredClients(
        joinRoom,
        [controlledContext, requestedRoomUrl, controlledNickname, {
          clientId: "controlled-reconnect",
          waitForHostJoin,
          requireGameSurface: !deferredHumanSurface,
          releaseInputs: releaseAllControllers,
          onSurfaceReady: () => sendReadiness({ controlled_surface_ready: true }),
        }],
        [opponentContext, requestedRoomUrl, opponentNickname, {
          clientId: "opponent-reconnect",
          waitForHostJoin,
          releaseInputs: releaseAllControllers,
          onSurfaceReady: () => sendReadiness({ opponent_surface_ready: true }),
        }],
        { humanOpponent },
      ));
      console.log(`bot_rejoined player_id=${controlled.player.id}`);
      if (deferredHumanSurface) {
        send("client_status", { ready: false, lobby_ready: true, reconnect_attempt: attempt });
        await waitForLobbyActivation(controlled);
        await initializeGameController("controlled", controlled);
        console.log("bot_controller_ready");
      } else {
        await waitForActiveGame(20_000);
        await prepareClientController("controlled", controlled);
      }
      if (opponentControlEnabled) await prepareClientController("opponent", opponent);
      send("client_status", { ready: true, reconnect_attempt: attempt });
    } catch (error) {
      await controlledContext?.close().catch(() => {});
      await opponentContext?.close().catch(() => {});
      controlledContext = null;
      opponentContext = null;
      controlled = null;
      opponent = null;
      throw error;
    }
  };

  reconnectCoordinator = new BotReconnectCoordinator({
    reconnect,
    reset: resetClients,
  });
  let transientHealthFailures = 0;
  const monitor = setInterval(async () => {
    if (closing || reconnectCoordinator.inFlight) return;
    const controlledReason = await unhealthyReason(controlled);
    const opponentReason = humanOpponent ? null : await unhealthyReason(opponent);
    const reason = controlledReason ?? opponentReason;
    if (!reason) {
      transientHealthFailures = 0;
      return;
    }
    if (reason === "missing_game_surface") {
      transientHealthFailures += 1;
      if (transientHealthFailures < 3) return;
    } else {
      transientHealthFailures = 0;
    }
    void reconnectCoordinator.request(reason);
  }, 2_000);

  while (!closing) {
    let payload;
    try {
      payload = await nextMessage(120_000);
    } catch (error) {
      if (error.message === "bridge message timeout") continue;
      throw error;
    }
    if (payload.type === "action") {
      const clientId = payload.client;
      if (!CLIENT_ROLES.includes(clientId)) {
        console.log(`browser_action_rejected: client=${clientId} reason=unknown_client`);
        continue;
      }
      const routed = inputRouter.enqueue(payload, async (applied) => {
        const lifecycleId = Number.isInteger(applied.lifecycle_id) ? applied.lifecycle_id : 0;
        if (firstAppliedLifecycle !== lifecycleId) {
          firstAppliedLifecycle = lifecycleId;
          firstActionAppliedCount += 1;
          console.log(
            `lifecycle_event event=first_action_applied count=${firstActionAppliedCount} ` +
            `lifecycle_id=${lifecycleId} timestamp=${Date.now()}`,
          );
        }
        send("action_applied", {
          tick_id: applied.tick_id,
          client: clientId,
          action: applied.action,
          applied_timestamp: Date.now(),
          lifecycle_id: lifecycleId,
        });
      });
      if (!routed.accepted) {
        await inputRouter.release(clientId).catch(() => {});
        console.log(`browser_action_rejected: client=${clientId} reason=${routed.reason}`);
        continue;
      }
      routed.completion.catch(async (error) => {
        const controller = routed.controller;
        controller.error = error;
        controller.setReady(false);
        setClientInputReady(clientId, false);
        await controller.releaseAll().catch(() => {});
        send("error", { code: "input_failure", client: clientId, detail: error.message });
        await close(`${clientId}_input_failure`);
      });
    } else if (payload.type === "reset") {
      await resetAllControllers();
    } else if (payload.type === "shutdown") {
      clearInterval(monitor);
      await close("bridge_shutdown");
    }
  }
} catch (error) {
  console.error(`client_error: ${error.message}`);
  await close("error");
  process.exitCode = 1;
}
