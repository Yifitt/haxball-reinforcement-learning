import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";

test("host starts exactly once after both confirmed players without waiting for actions", async () => {
  const source = await readFile(new URL("../headless_host/host.js", import.meta.url), "utf8");
  const timers = [];
  const logs = [];
  const messages = [];
  const assignments = [];
  const teams = new Map();
  let startCalls = 0;
  const room = {
    setDefaultStadium() {}, setScoreLimit() {}, setTimeLimit() {}, setTeamsLock() {},
    setPlayerTeam(id, team) { assignments.push([id, team]); teams.set(id, team); }, kickPlayer() {},
    startGame() { startCalls += 1; },
    getPlayer(id) { return { id, team: teams.get(id) }; },
    getPlayerDiscProperties(id) {
      return { x: id * 10, y: 0, xspeed: 0, yspeed: 0 };
    },
    getDiscProperties() { return { x: 0, y: 0, xspeed: 0, yspeed: 0 }; },
    getScores() { return { red: 0, blue: 0, time: 0 }; },
  };
  const window = {
    __HAXBALL_INTEGRATION_CONFIG__: {
      controlDecimation: 3,
      roomName: "test",
      controlledNickname: "RL-Agent",
      opponentNickname: "Scripted-Opponent",
    },
    __hostBridgeSend: async (payload) => { messages.push(payload); },
    HBInit: () => room,
    addEventListener() {},
  };
  vm.runInNewContext(source, {
    window,
    console: { log: (value) => logs.push(value), error() {} },
    performance: { now: () => 0 },
    setTimeout: (callback) => { timers.push(callback); },
    Date,
  });

  room.onPlayerJoin({ id: 1, name: "Scripted-Opponent" });
  assert.equal(timers.length, 0);
  room.onPlayerJoin({ id: 2, name: "RL-Agent" });
  assert.equal(timers.length, 1);
  timers.shift()();
  assert.equal(startCalls, 1);
  assert.deepEqual(assignments, [[1, 2], [2, 1]]);
  const joined = messages.filter((payload) => payload.lifecycle === "player_joined");
  assert.equal(joined.find((payload) => payload.player.client === "controlled").player.id, 2);
  assert.equal(joined.find((payload) => payload.player.client === "opponent").player.id, 1);
  room.onGameStart();
  room.onGameTick(); room.onGameTick(); room.onGameTick();
  const state = messages.find((payload) => payload.type === "state");
  assert.equal(state.controlled_player.id, 2);
  assert.equal(state.controlled_player.team, 1);
  assert.equal(state.opponent_player.id, 1);
  assert.equal(state.opponent_player.team, 2);
  assert.ok(logs.includes("host_game_start: players_confirmed=2 started=true"));
});

test("human opponent is authoritatively mapped to Blue and room URL is printed", async () => {
  const source = await readFile(new URL("../headless_host/host.js", import.meta.url), "utf8");
  const timers = [];
  const logs = [];
  const assignments = [];
  let startCalls = 0;
  const room = {
    setDefaultStadium() {}, setScoreLimit() {}, setTimeLimit() {}, setTeamsLock() {},
    setPlayerTeam(id, team) { assignments.push([id, team]); },
    kickPlayer() { throw new Error("human should not be kicked"); },
    startGame() { startCalls += 1; },
    getPlayer() { return null; }, getPlayerDiscProperties() { return null; },
    getDiscProperties() { return null; }, getScores() { return null; },
  };
  const window = {
    __HAXBALL_INTEGRATION_CONFIG__: {
      controlDecimation: 3, roomName: "test", controlledNickname: "RL-Agent",
      opponentNickname: "Scripted-Opponent", humanOpponent: true,
    },
    __hostBridgeSend: async () => {}, HBInit: () => room, addEventListener() {},
  };
  vm.runInNewContext(source, {
    window, console: { log: (value) => logs.push(value), error() {} },
    performance: { now: () => 0 }, setTimeout: (callback) => timers.push(callback), Date,
  });
  room.onRoomLink("https://www.haxball.com/play?c=private-code");
  room.onPlayerJoin({ id: 10, name: "RL-Agent" });
  room.onPlayerJoin({ id: 11, name: "Human Player" });
  assert.deepEqual(assignments, [[10, 1], [11, 2]]);
  timers.shift()();
  assert.equal(startCalls, 1);
  assert.ok(logs.includes("PRIVATE_ROOM_URL: https://www.haxball.com/play?c=private-code"));
});
