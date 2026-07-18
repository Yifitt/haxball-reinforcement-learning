import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { InputController } from "../browser/input_controller.js";
import { DualInputRouter } from "../browser/dual_input.js";
import { ActionGate } from "../bridge/protocol.js";

class MockKeyboard {
  constructor() { this.events = []; }
  async down(key) { this.events.push(["down", key]); }
  async up(key) { this.events.push(["up", key]); }
}

const root = fileURLToPath(new URL("../../", import.meta.url));
const python = process.env.PYTHON ?? "python";
const result = spawnSync(
  python,
  ["-m", "integration.scripts.smoke_real_haxball", "--offline-stationary-kickoff", "--json"],
  { cwd: root, encoding: "utf8" },
);
if (result.status !== 0) throw new Error(result.stderr || "checkpoint kickoff probe failed");
const report = JSON.parse(result.stdout);

const keyboard = new MockKeyboard();
const controller = new InputController(keyboard);
const router = new DualInputRouter();
router.register("controlled", controller, true);
const gate = new ActionGate();
gate.setClientReady(true);

for (let index = 0; index < report.lifecycles.length; index += 1) {
  const lifecycleId = report.lifecycles[index];
  const tickId = report.ticks[index];
  gate.resetLifecycle(lifecycleId);
  assert.equal(gate.observeState({
    tick_id: tickId, lifecycle_id: lifecycleId, game_active: true,
  }), true);
  const payload = {
    tick_id: tickId, lifecycle_id: lifecycleId,
    client: "controlled", action: report.actions[index],
  };
  assert.equal(gate.accept(payload).ok, true);
  const routed = router.enqueue(payload);
  assert.equal(routed.accepted, true);
  await routed.completion;
  assert.ok(controller.held.size > 0);
  await router.resetAll();
  assert.equal(controller.held.size, 0);
}

assert.equal(report.identical_opening_bodies, true);
assert.equal(report.inference_count, 2);
assert.deepEqual(report.ticks, [7, 7]);
assert.ok(keyboard.events.some(([event]) => event === "down"));
assert.ok(keyboard.events.some(([event]) => event === "up"));
console.log(JSON.stringify({
  stationary_kickoff_smoke: true,
  identical_rematch_positions: true,
  inference_count: report.inference_count,
  actions_applied: report.actions.length,
  held_keys_after_resets: controller.held.size,
}));
