import assert from "node:assert/strict";
import test from "node:test";

import { InputController } from "../browser/input_controller.js";

class MockKeyboard {
  constructor() { this.events = []; }
  async down(key) { this.events.push(["down", key]); }
  async up(key) { this.events.push(["up", key]); }
}

test("movement transitions release only obsolete keys", async () => {
  const keyboard = new MockKeyboard();
  const controller = new InputController(keyboard);
  await controller.applyAction(4); // right
  await controller.applyAction(6); // up-right
  assert.deepEqual([...controller.held].sort(), ["ArrowRight", "ArrowUp"]);
  assert.deepEqual(keyboard.events, [
    ["down", "ArrowRight"],
    ["down", "ArrowUp"],
  ]);
  await controller.applyAction(0);
  assert.equal(controller.held.size, 0);
  assert.deepEqual(keyboard.events.slice(-2), [
    ["up", "ArrowRight"],
    ["up", "ArrowUp"],
  ]);
});

test("diagonal movement holds exactly two direction keys", async () => {
  const controller = new InputController(new MockKeyboard());
  await controller.applyAction(7);
  assert.deepEqual([...controller.held].sort(), ["ArrowDown", "ArrowLeft"]);
});

test("kick is a bounded press", async () => {
  const keyboard = new MockKeyboard();
  const sleeps = [];
  const controller = new InputController(keyboard, {
    kickMilliseconds: 23,
    sleep: async (milliseconds) => sleeps.push(milliseconds),
  });
  await controller.applyAction(12);
  assert.deepEqual(sleeps, [23]);
  assert.deepEqual(keyboard.events, [
    ["down", "ArrowLeft"],
    ["down", "x"],
    ["up", "x"],
  ]);
  assert.equal(controller.kickHeld, false);
});

test("emergency release clears held movement and kick", async () => {
  const keyboard = new MockKeyboard();
  const controller = new InputController(keyboard);
  controller.held.add("ArrowDown");
  controller.kickHeld = true;
  await controller.releaseAll();
  assert.equal(controller.held.size, 0);
  assert.equal(controller.kickHeld, false);
  assert.deepEqual(keyboard.events, [["up", "ArrowDown"], ["up", "x"]]);
});

test("two client controllers hold and release keys independently", async () => {
  const controlledKeyboard = new MockKeyboard();
  const opponentKeyboard = new MockKeyboard();
  const controlled = new InputController(controlledKeyboard, { clientId: "controlled" });
  const opponent = new InputController(opponentKeyboard, { clientId: "opponent" });
  await controlled.applyAction(4);
  await opponent.applyAction(3);
  assert.deepEqual([...controlled.held], ["ArrowRight"]);
  assert.deepEqual([...opponent.held], ["ArrowLeft"]);
  await controlled.releaseAll();
  assert.equal(controlled.held.size, 0);
  assert.deepEqual([...opponent.held], ["ArrowLeft"]);
  assert.deepEqual(opponentKeyboard.events, [["down", "ArrowLeft"]]);
});

test("input controller tracks readiness and applied-action lifecycle", async () => {
  const controller = new InputController(new MockKeyboard(), {
    clientId: "opponent",
    ready: false,
  });
  await assert.rejects(() => controller.applyAction(3), /opponent input controller is not ready/);
  controller.setReady(true, { method: "dom_focus" });
  await controller.applyAction(3);
  assert.equal(controller.lastAppliedAction, 3);
  assert.equal(controller.appliedActionCount, 1);
  assert.deepEqual(controller.focusState, { method: "dom_focus" });
  await controller.releaseAll();
  assert.equal(controller.lastAppliedAction, null);
});
