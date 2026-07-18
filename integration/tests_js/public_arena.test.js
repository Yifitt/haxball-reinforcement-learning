import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

async function createArena(configOverrides = {}) {
  const queueSource = await readFile(new URL("../headless_host/player_queue.js", import.meta.url), "utf8");
  const hostSource = await readFile(new URL("../headless_host/host.js", import.meta.url), "utf8");
  const players = new Map();
  const assignments = [];
  const adminChanges = [];
  const announcements = [];
  const logs = [];
  const kicks = [];
  const timers = [];
  const intervals = [];
  const initOptions = [];
  const messages = [];
  let starts = 0;
  let stops = 0;
  let recaptcha = null;
  let teamsLocked = false;
  const room = {
    setDefaultStadium() {}, setScoreLimit(value) { this.scoreLimit = value; },
    setTimeLimit(value) { this.timeLimit = value; },
    setTeamsLock(value) { teamsLocked = value; },
    setRequireRecaptcha(value) { recaptcha = value; },
    setPlayerTeam(id, team) {
      assignments.push([id, team]);
      if (players.has(id)) players.get(id).team = team;
    },
    setPlayerAdmin(id, value) { adminChanges.push([id, value]); },
    kickPlayer(id, reason) { kicks.push([id, reason]); },
    sendAnnouncement(text, target) { announcements.push([text, target]); },
    startGame() { starts += 1; }, stopGame() { stops += 1; },
    getPlayer(id) { return players.get(id) ?? null; },
    getPlayerDiscProperties() { return { x: 0, y: 0, xspeed: 0, yspeed: 0 }; },
    getDiscProperties() { return { x: 0, y: 0, xspeed: 0, yspeed: 0 }; },
    getScores() { return { red: 2, blue: 1, time: 0 }; },
  };
  const window = {
    __HAXBALL_INTEGRATION_CONFIG__: {
      token: "test-token-placeholder", controlDecimation: 3, roomName: "RL Bot | 1v1",
      controlledNickname: "RL-Agent", opponentNickname: "Scripted-Opponent",
      publicRoom: true, humanOpponent: true, maxPlayers: 12, scoreLimit: 5,
      startupMode: "public_human_queue",
      timeLimit: 0, maxNicknameLength: 25,
      enablePlayerQueue: true, matchesPerTurn: 1,
      ...configOverrides,
    },
    __hostBridgeSend: async (message) => { messages.push(message); },
    HBInit: (options) => { initOptions.push(options); return room; },
    addEventListener() {},
  };
  const context = vm.createContext({
    window, console: { log: (value) => logs.push(value), error() {} },
    performance: { now: () => 0 },
    setTimeout: (callback) => { timers.push(callback); return timers.length; },
    setInterval: (callback) => { intervals.push(callback); return intervals.length; },
    Date, Map, Set,
  });
  vm.runInContext(queueSource, context);
  vm.runInContext(hostSource, context);
  const join = (id, name) => {
    const player = { id, name, team: 0, admin: false };
    players.set(id, player);
    room.onPlayerJoin(player);
    return player;
  };
  const runTimer = () => timers.shift()?.();
  return {
    room, window, players, assignments, adminChanges, announcements, logs, kicks,
    timers, intervals, initOptions, messages, join, runTimer,
    starts: () => starts, stops: () => stops,
    teamsLocked: () => teamsLocked, recaptcha: () => recaptcha,
  };
}

const botReady = {
  type: "readiness", controlled_surface_ready: true, controlled_input_ready: true,
  browser_connected: true, controlled_joined: true,
  python_connected: true, python_protocol_ready: true,
};
const botLobbyReady = {
  ...botReady,
  controlled_surface_ready: false,
  controlled_input_ready: false,
};

test("public arena configures a safe room and starts only after bot readiness", async () => {
  const arena = await createArena();
  arena.room.onRoomLink("https://www.haxball.com/play?c=public-room");
  arena.join(1, "RL-Agent");
  arena.join(2, "Alice");
  arena.join(3, "Bob");
  assert.equal(arena.initOptions[0].public, true);
  assert.equal(arena.initOptions[0].maxPlayers, 12);
  assert.equal(arena.initOptions[0].noPlayer, true);
  assert.equal(arena.initOptions[0].password, undefined);
  assert.equal(arena.room.scoreLimit, 5);
  assert.equal(arena.room.timeLimit, 0);
  assert.equal(arena.teamsLocked(), true);
  assert.equal(arena.recaptcha(), false);
  assert.ok(arena.assignments.some(([id, team]) => id === 1 && team === 1));
  assert.equal(arena.players.get(2).team, 2);
  assert.equal(arena.players.get(3).team, 0);
  assert.deepEqual(arena.adminChanges, [[1, false], [2, false], [3, false]]);
  assert.equal(arena.timers.length, 0);
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  assert.equal(arena.players.get(2).team, 2);
  assert.equal(arena.timers.length, 1);
  arena.runTimer();
  assert.equal(arena.starts(), 1);
  assert.ok(arena.logs.includes("PUBLIC_ROOM_READY"));
  assert.ok(arena.logs.includes("bot=Red"));
  assert.ok(arena.logs.every((line) => !line.includes("test-token-placeholder")));
});

test("bot waits healthy in the lobby and a delayed human starts surface initialization", async () => {
  const arena = await createArena();
  const bot = arena.join(1, "RL-Agent");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botLobbyReady);
  assert.equal(bot.team, 1);
  assert.equal(arena.timers.length, 0);
  assert.ok(!arena.logs.some((line) => line.startsWith("bot_disconnected")));

  const human = arena.join(2, "Alice");
  assert.equal(human.team, 2);
  assert.equal(arena.timers.length, 1);
  arena.runTimer();
  assert.equal(arena.starts(), 1);
  arena.room.onGameStart();
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  assert.ok(arena.logs.includes("bot_controller_ready"));
});

test("public arena rotates Blue at official match end and enforces spectator teams", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  const alice = arena.join(2, "Alice");
  const bob = arena.join(3, "Bob");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  arena.room.onTeamVictory();
  assert.equal(arena.stops(), 1);
  arena.room.onGameStop();
  arena.runTimer();
  assert.equal(alice.team, 0);
  assert.equal(bob.team, 2);
  assert.equal(arena.starts(), 2);
  alice.team = 1;
  arena.room.onPlayerTeamChange(alice);
  assert.equal(alice.team, 0);
  bob.admin = true;
  arena.room.onPlayerAdminChange(bob);
  assert.deepEqual(arena.adminChanges.at(-1), [3, false]);
  assert.equal(arena.room.onPlayerChat, undefined);
});

test("queue-disabled public room keeps the active human across match restarts", async () => {
  const arena = await createArena({ enablePlayerQueue: false });
  arena.join(1, "RL-Agent");
  const alice = arena.join(2, "Alice");
  const bob = arena.join(3, "Bob");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  arena.room.onTeamVictory();
  arena.room.onGameStop();
  assert.equal(alice.team, 2);
  assert.equal(bob.team, 0);
});

test("official stops emit one match boundary before the next queued start", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Alice");
  arena.join(3, "Bob");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  arena.room.onTeamVictory();
  arena.room.onGameStop();
  arena.room.onTeamVictory();
  arena.room.onGameStop();
  const gameStops = arena.messages.filter(
    (message) => message.type === "reset" && message.reason === "game_stop",
  );
  assert.equal(gameStops.length, 1);
  assert.equal(arena.players.get(3).team, 2);
  assert.equal(arena.window.__HAXBALL_PLAYER_QUEUE__.snapshot().gameRunning, false);
  arena.runTimer();
  arena.room.onGameStart();
  assert.equal(arena.window.__HAXBALL_PLAYER_QUEUE__.snapshot().gameRunning, true);
});

test("repeated readiness cannot schedule duplicate game starts", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Alice");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  assert.equal(arena.timers.length, 1);
  arena.runTimer();
  assert.equal(arena.starts(), 1);
});

test("spectator departure does not interrupt the active match", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  const alice = arena.join(2, "Alice");
  const bob = arena.join(3, "Bob");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  arena.players.delete(bob.id);
  arena.room.onPlayerLeave(bob);
  assert.equal(arena.stops(), 0);
  assert.equal(alice.team, 2);
});

test("active-human leave and bot disconnect stop cleanly and gate restart", async () => {
  const arena = await createArena();
  const bot = arena.join(1, "RL-Agent");
  const alice = arena.join(2, "Alice");
  const bob = arena.join(3, "Bob");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  arena.players.delete(alice.id);
  arena.room.onPlayerLeave(alice);
  assert.equal(arena.stops(), 1);
  arena.room.onGameStop();
  arena.runTimer();
  assert.equal(bob.team, 2);
  arena.players.delete(bot.id);
  arena.room.onPlayerLeave(bot);
  assert.ok(arena.announcements.some(([text]) => text.includes("Bot disconnected")));
  const startsBeforeRecovery = arena.starts();
  arena.join(4, "RL-Agent");
  assert.equal(arena.starts(), startsBeforeRecovery);
  arena.window.__HAXBALL_BRIDGE_MESSAGE__({ ...botReady, controlled_input_ready: false });
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  assert.ok(arena.timers.length >= 1);
});

test("bot disconnect preserves the active challenger and restarts only after controller readiness", async () => {
  const arena = await createArena();
  const bot = arena.join(1, "RL-Agent");
  const alice = arena.join(2, "Alice");
  arena.join(3, "Bob");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();

  arena.players.delete(bot.id);
  arena.room.onPlayerLeave(bot);
  assert.equal(arena.stops(), 1);
  assert.equal(alice.team, 0);
  arena.room.onGameStop();
  const startsBeforeRecovery = arena.starts();
  const replacement = arena.join(4, "RL-Agent");
  assert.equal(replacement.team, 1);
  assert.equal(alice.team, 0);
  assert.equal(arena.starts(), startsBeforeRecovery);

  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  assert.equal(alice.team, 2);
  arena.runTimer();
  arena.room.onGameStart();
  assert.equal(arena.starts(), startsBeforeRecovery + 1);
  assert.ok(arena.logs.some((line) => line.startsWith("bot_disconnected")));
  assert.ok(arena.logs.some((line) => line.startsWith("bot_rejoined")));
  assert.ok(arena.logs.includes("bot_controller_ready"));
  assert.ok(arena.logs.includes("match_restarted"));
});

test("duplicate bot clients are rejected and never become human spectators", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "RL-Agent");
  const human = arena.join(3, "Alice");
  assert.deepEqual(arena.kicks, [[2, "Duplicate bot client"]]);
  assert.equal(human.team, 2);
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  assert.equal(human.team, 2);
  assert.deepEqual(Array.from(arena.window.__HAXBALL_PLAYER_QUEUE__.connectedHumanIds), [3]);
});

test("published opponent state includes only the active Blue human", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Alice");
  arena.join(3, "Queued-Spectator");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  arena.room.onGameTick();
  arena.room.onGameTick();
  arena.room.onGameTick();
  const state = arena.messages.find((message) => message.type === "state");
  assert.equal(state.opponent.id, 2);
  assert.equal(state.opponent.team, 2);
  assert.ok(!JSON.stringify(state.opponent).includes("Queued-Spectator"));
  const blueHumans = [...arena.players.values()].filter(
    (player) => player.name !== "RL-Agent" && player.team === 2,
  );
  assert.equal(blueHumans.length, 1);
});

test("stationary kickoff force-sends an active Red snapshot before any game tick", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Stationary-Human");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  const states = arena.messages.filter((message) => message.type === "state");
  assert.equal(states.length, 1);
  assert.equal(states[0].forced_snapshot, true);
  assert.equal(states[0].snapshot_reason, "game_start");
  assert.equal(states[0].controlled.team, 1);
  assert.equal(states[0].opponent.team, 2);
  assert.equal(states[0].state_sequence, 1);
  assert.deepEqual(JSON.parse(JSON.stringify(states[0].lifecycle_counters)), {
    game_start: 1, position_reset: 0, first_state_sent: 1,
  });
  assert.equal(typeof states[0].lifecycle_timestamps.game_start, "number");
  assert.equal(typeof states[0].lifecycle_timestamps.first_state_sent, "number");
});

test("identical rematch kickoff creates a fresh lifecycle snapshot", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Stationary-Human");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  const first = arena.messages.filter((message) => message.type === "state").at(-1);
  arena.room.onTeamVictory();
  arena.room.onGameStop();
  arena.runTimer();
  arena.room.onGameStart();
  const second = arena.messages.filter((message) => message.type === "state").at(-1);
  assert.notEqual(second.lifecycle_id, first.lifecycle_id);
  assert.equal(second.state_sequence, 1);
  assert.deepEqual(
    JSON.parse(JSON.stringify([second.controlled, second.opponent, second.ball])),
    JSON.parse(JSON.stringify([first.controlled, first.opponent, first.ball])),
  );
});

test("queue promotion force-sends a snapshot for the newly active Blue human", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Alice");
  arena.join(3, "Bob");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  arena.room.onTeamVictory();
  arena.room.onGameStop();
  arena.runTimer();
  arena.room.onGameStart();
  const promoted = arena.messages.filter((message) => message.type === "state").at(-1);
  assert.equal(promoted.forced_snapshot, true);
  assert.equal(promoted.opponent.id, 3);
  assert.equal(promoted.opponent.team, 2);
});

test("goal position reset clears the epoch and force-sends unchanged positions", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Stationary-Human");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  const first = arena.messages.filter((message) => message.type === "state").at(-1);
  arena.room.onTeamGoal(2);
  arena.room.onPositionsReset();
  const reset = arena.messages.filter((message) => message.type === "state").at(-1);
  assert.notEqual(reset.lifecycle_id, first.lifecycle_id);
  assert.equal(reset.snapshot_reason, "position_reset");
  assert.equal(reset.state_sequence, 1);
  assert.deepEqual(
    JSON.parse(JSON.stringify([reset.controlled, reset.opponent, reset.ball])),
    JSON.parse(JSON.stringify([first.controlled, first.opponent, first.ball])),
  );
});

test("duplicate game-start callback does not start another lifecycle or action loop", async () => {
  const arena = await createArena();
  arena.join(1, "RL-Agent");
  arena.join(2, "Stationary-Human");
  arena.window.__HAXBALL_BRIDGE_MESSAGE__(botReady);
  arena.runTimer();
  arena.room.onGameStart();
  const lifecycle = arena.window.__HAXBALL_PLAYER_QUEUE__.snapshot();
  const gameStartResets = () => arena.messages.filter(
    (message) => message.type === "reset" && message.reason === "game_start",
  );
  assert.equal(gameStartResets().length, 1);
  arena.room.onGameStart();
  assert.equal(gameStartResets().length, 1);
  assert.equal(arena.stops(), 0);
  assert.equal(arena.window.__HAXBALL_PLAYER_QUEUE__.snapshot().gameRunning, true);
  assert.equal(lifecycle.gameRunning, true);
});
