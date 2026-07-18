import assert from "node:assert/strict";
import test from "node:test";

import { ControlledActionWatchdog } from "../bridge/action_watchdog.js";

test("stationary Red kickoff requests one recovery without human movement", () => {
  let now = 1_000;
  const watchdog = new ControlledActionWatchdog({ now: () => now });
  watchdog.observeState({
    lifecycle_id: 3, game_active: true, controlled: { team: 1 },
  });
  now += 749;
  assert.equal(watchdog.poll(true), null);
  now += 1;
  assert.deepEqual(watchdog.poll(true), {
    lifecycleId: 3, reason: "no_fresh_controlled_action",
  });
  now += 5_000;
  assert.equal(watchdog.poll(true), null);
});

test("healthy applied actions re-arm without repeatedly restarting", () => {
  let now = 0;
  const watchdog = new ControlledActionWatchdog({ now: () => now });
  watchdog.observeState({
    lifecycle_id: 4, game_active: true, controlled: { team: 1 },
  });
  for (let index = 0; index < 20; index += 1) {
    now += 100;
    watchdog.observeActionApplied({ lifecycle_id: 4, client: "controlled" });
    assert.equal(watchdog.poll(true), null);
  }
  now += 750;
  assert.equal(watchdog.poll(true)?.lifecycleId, 4);
});

test("watchdog is disabled while unready and resets across reconnect lifecycle", () => {
  let now = 0;
  const watchdog = new ControlledActionWatchdog({ now: () => now });
  watchdog.observeState({
    lifecycle_id: 8, game_active: true, controlled: { team: 1 },
  });
  now = 2_000;
  assert.equal(watchdog.poll(false), null);
  watchdog.readinessOpened();
  assert.equal(watchdog.poll(true), null);
  now += 749;
  assert.equal(watchdog.poll(true), null);
  now += 1;
  assert.equal(watchdog.poll(true)?.lifecycleId, 8);
  watchdog.reset(9);
  assert.equal(watchdog.poll(true), null);
  watchdog.observeState({
    lifecycle_id: 9, game_active: true, controlled: { team: 1 },
  });
  now += 750;
  assert.equal(watchdog.poll(true)?.lifecycleId, 9);
});
