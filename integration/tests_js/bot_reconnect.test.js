import assert from "node:assert/strict";
import test from "node:test";

import {
  BOT_RECONNECT_DELAYS_MS,
  BotReconnectCoordinator,
} from "../browser/bot_reconnect.js";

test("bot reconnect uses bounded exponential backoff and eventually succeeds", async () => {
  const sleeps = [];
  const attempts = [];
  let resets = 0;
  const coordinator = new BotReconnectCoordinator({
    reset: async () => { resets += 1; },
    sleep: async (delay) => { sleeps.push(delay); },
    reconnect: async (attempt) => {
      attempts.push(attempt);
      if (attempt < 4) throw new Error("still disconnected");
    },
    log: () => {},
  });
  assert.equal(await coordinator.request("player_left"), true);
  assert.equal(resets, 1);
  assert.deepEqual(sleeps, BOT_RECONNECT_DELAYS_MS);
  assert.deepEqual(attempts, [1, 2, 3, 4]);
});

test("concurrent watchdog triggers share one reconnect and cannot create duplicate bots", async () => {
  let releases;
  const gate = new Promise((resolve) => { releases = resolve; });
  let reconnects = 0;
  const coordinator = new BotReconnectCoordinator({
    reset: async () => {},
    sleep: async () => {},
    reconnect: async () => {
      reconnects += 1;
      await gate;
    },
    log: () => {},
  });
  const first = coordinator.request("closed_page");
  const second = coordinator.request("missing_bot_player_id");
  assert.equal(first, second);
  releases();
  assert.equal(await first, true);
  assert.equal(reconnects, 1);
});
