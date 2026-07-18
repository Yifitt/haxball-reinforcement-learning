import assert from "node:assert/strict";
import test from "node:test";

import { executeJoinLifecycle, JOIN_STAGES } from "../browser/join_lifecycle.js";

function adapter(overrides = {}) {
  const calls = [];
  return {
    calls,
    navigate: async (url) => calls.push(["navigate", url]),
    findNicknameForm: async () => null,
    enterNickname: async (_form, nickname) => calls.push(["nickname", nickname]),
    submitNickname: async () => calls.push(["submit"]),
    shouldReopenRoom: async () => false,
    prepareRetry: async () => calls.push(["prepare-retry"]),
    findGameSurface: async () => ({ page: "original", frame: "main", canvas: "game" }),
    waitForHostJoin: async (name) => ({ id: 1, name, team: 1 }),
    onFailure: async (stage, error) => calls.push(["failure", stage, error.message]),
    ...overrides,
  };
}

const run = (mock, overrides = {}) => executeJoinLifecycle({
  adapter: mock,
  nickname: "RL-Agent",
  roomUrl: "https://www.haxball.com/play?c=private",
  ...overrides,
});

test("room opens in the original page", async () => {
  const mock = adapter();
  const result = await run(mock);
  assert.equal(result.page, "original");
  assert.equal(result.stage, JOIN_STAGES.READY);
});

test("room popup is adopted when it contains the game", async () => {
  const mock = adapter({
    findGameSurface: async () => ({ page: "popup", frame: "popup-child", canvas: "game" }),
  });
  assert.equal((await run(mock)).page, "popup");
});

test("nickname form is submitted and private URL is reopened before canvas detection", async () => {
  const form = { input: "nickname", submit: "ok" };
  const mock = adapter({
    findNicknameForm: async () => form,
    shouldReopenRoom: async () => true,
  });
  await run(mock);
  assert.deepEqual(mock.calls, [
    ["navigate", "https://www.haxball.com/play?c=private"],
    ["nickname", "RL-Agent"],
    ["submit"],
    ["navigate", "https://www.haxball.com/play?c=private"],
    ["nickname", "RL-Agent"],
    ["submit"],
  ]);
});

test("canvas can appear in a child frame among multiple pages", async () => {
  const mock = adapter({
    findGameSurface: async () => ({ page: "room-page", frame: "game-child-frame", canvas: "game" }),
  });
  const result = await run(mock);
  assert.equal(result.page, "room-page");
  assert.equal(result.frame, "game-child-frame");
});

test("host join confirmation is authoritative", async () => {
  const mock = adapter({
    waitForHostJoin: async (name) => ({ id: 42, name, team: 1 }),
  });
  assert.deepEqual((await run(mock)).player, { id: 42, name: "RL-Agent", team: 1 });
});

test("human-opponent lobby join succeeds without requiring a game surface", async () => {
  let surfaceCalls = 0;
  const mock = adapter({
    findGameSurface: async () => {
      surfaceCalls += 1;
      throw new Error("only lobby canvases are present");
    },
  });
  const result = await run(mock, { requireGameSurface: false });
  assert.equal(result.player.name, "RL-Agent");
  assert.equal(result.stage, JOIN_STAGES.READY);
  assert.equal(surfaceCalls, 0);
  assert.ok(!mock.calls.some(([name]) => name === "failure"));
});

test("nickname submission is not accepted as entry until the host confirms it", async () => {
  const form = { input: "nickname", submit: "ok" };
  const order = [];
  const mock = adapter({
    findNicknameForm: async () => form,
    enterNickname: async () => order.push("nickname"),
    submitNickname: async () => order.push("submitted"),
    waitForHostJoin: async (name) => {
      order.push("host-confirmed");
      return { id: 1, name, team: 1 };
    },
    findGameSurface: async () => {
      order.push("surface");
      return { page: "room", frame: "child", canvas: "game" };
    },
  });
  await run(mock);
  assert.deepEqual(order, ["nickname", "submitted", "host-confirmed", "surface"]);
});

test("failed host entry gets exactly one bounded per-client rejoin", async () => {
  const forms = [{ attempt: 1 }, { attempt: 2 }];
  const retries = [];
  let hostAttempts = 0;
  let prepared = 0;
  const mock = adapter({
    findNicknameForm: async () => forms.shift() ?? null,
    waitForHostJoin: async (name) => {
      hostAttempts += 1;
      if (hostAttempts === 1) throw new Error("actual host room entry was not confirmed");
      return { id: 7, name, team: 1 };
    },
    onRetry: async (attempt, reason) => retries.push([attempt, reason]),
    prepareRetry: async () => { prepared += 1; },
  });
  const result = await run(mock);
  assert.equal(result.player.id, 7);
  assert.equal(hostAttempts, 2);
  assert.equal(prepared, 1);
  assert.equal(mock.calls.filter(([name]) => name === "submit").length, 2);
  assert.equal(retries.length, 1);
  assert.equal(retries[0][0], 2);
});

test("connection dialog is reported after one bounded retry", async () => {
  let attempts = 0;
  let preparations = 0;
  const mock = adapter({
    waitForHostJoin: async () => {
      attempts += 1;
      throw new Error("HaxBall connection error: WebRTC failure");
    },
    prepareRetry: async () => { preparations += 1; },
  });
  await assert.rejects(() => run(mock), /HaxBall connection error: WebRTC failure/);
  assert.equal(attempts, 2);
  assert.equal(preparations, 1);
});

test("host confirmation is awaited before the game canvas", async () => {
  const order = [];
  const mock = adapter({
    waitForHostJoin: async (name) => {
      order.push("host-confirmed");
      return { id: 42, name, team: 1 };
    },
    findGameSurface: async () => {
      order.push("game-surface");
      return { page: "room", frame: "child", canvas: "game" };
    },
  });
  await run(mock);
  assert.deepEqual(order, ["host-confirmed", "game-surface"]);
});

test("surface readiness is reported after discovery", async () => {
  const order = [];
  const mock = adapter({
    waitForHostJoin: async (name) => {
      order.push("host");
      return { id: 1, name, team: 1 };
    },
    findGameSurface: async () => {
      order.push("surface");
      return { page: "room", frame: "child", canvas: "game" };
    },
    onSurfaceReady: async () => order.push("surface-reported"),
  });
  await run(mock);
  assert.deepEqual(order, ["host", "surface", "surface-reported"]);
});

test("missing host confirmation fails before canvas detection", async () => {
  let surfaceChecks = 0;
  const mock = adapter({
    waitForHostJoin: async () => {
      throw new Error("controlled client UI loaded, but player did not join host within 20 seconds");
    },
    findGameSurface: async () => {
      surfaceChecks += 1;
      return { page: "room", frame: "child", canvas: "game" };
    },
  });
  await assert.rejects(() => run(mock), /did not join host within 20 seconds/);
  assert.equal(surfaceChecks, 0);
  assert.equal(mock.calls.filter(([name]) => name === "prepare-retry").length, 1);
});

test("visible connection failure is propagated instead of a generic canvas error", async () => {
  const mock = adapter({
    findGameSurface: async () => {
      throw new Error("HaxBall connection error: Connection failed");
    },
  });
  await assert.rejects(() => run(mock), /Connection failed/);
});

test("join timeout captures the failed stage and releases held input", async () => {
  let releases = 0;
  const mock = adapter({
    findGameSurface: async () => {
      throw new Error("No valid HaxBall game canvas; pages=2 frames=3");
    },
  });
  await assert.rejects(() => run(mock, { releaseInputs: async () => { releases += 1; } }), /pages=2/);
  assert.equal(releases, 1);
  assert.deepEqual(mock.calls.at(-1), [
    "failure",
    JOIN_STAGES.WAITING_FOR_GAME_SURFACE,
    "No valid HaxBall game canvas; pages=2 frames=3",
  ]);
});

test("room-list retry is bounded and records its observed reason", async () => {
  const form = { input: "nickname", submit: "ok" };
  const retries = [];
  const mock = adapter({
    findNicknameForm: async () => form,
    shouldReopenRoom: async () => true,
    onRetry: async (attempt, reason) => retries.push([attempt, reason]),
  });
  await run(mock);
  assert.deepEqual(retries, [[1, "returned_to_room_list_after_profile_setup"]]);
  assert.equal(mock.calls.filter(([name]) => name === "navigate").length, 2);
});
