import assert from "node:assert/strict";
import test from "node:test";

import {
  formatPointerInterceptionDiagnostics,
  runFocusStrategies,
  selectSafeCanvasPoint,
  verifyKeyboardDelivery,
} from "../browser/input_focus.js";

test("safe uncovered canvas point is selected and covered points are rejected", () => {
  const covered = { fx: 0.5, fy: 0.5, safe: false };
  const safe = { fx: 0.1, fy: 0.9, safe: true };
  assert.equal(selectSafeCanvasPoint([covered, safe]), safe);
  assert.equal(selectSafeCanvasPoint([covered]), null);
});

test("programmatic focus succeeds before click fallbacks", async () => {
  const calls = [];
  const result = await runFocusStrategies([
    { name: "dom_focus", run: async () => { calls.push("dom"); return { focused: true }; } },
    { name: "safe_point_click", run: async () => { calls.push("click"); return { focused: true }; } },
    { name: "forced_click", run: async () => { calls.push("force"); return { focused: true }; } },
  ], async () => true);
  assert.equal(result.method, "dom_focus");
  assert.deepEqual(calls, ["dom"]);
});

test("body focus runs when canvas focus is unavailable", async () => {
  const calls = [];
  const result = await runFocusStrategies([
    { name: "dom_focus", run: async () => { calls.push("dom"); return { focused: false }; } },
    { name: "body_focus", run: async () => { calls.push("body"); return { focused: true }; } },
    { name: "safe_point_click", run: async () => { calls.push("click"); return { focused: true }; } },
  ], async () => true);
  assert.equal(result.method, "body_focus");
  assert.deepEqual(calls, ["dom", "body"]);
});

test("failed verification advances to safe point before forced click", async () => {
  const calls = [];
  const result = await runFocusStrategies([
    { name: "dom_focus", run: async () => { calls.push("dom"); return { focused: true }; } },
    { name: "body_focus", run: async () => { calls.push("body"); return { focused: true }; } },
    { name: "safe_point_click", run: async () => { calls.push("safe"); return { focused: true }; } },
    { name: "forced_click", run: async () => { calls.push("force"); return { focused: true }; } },
  ], async (method) => method === "safe_point_click");
  assert.equal(result.method, "safe_point_click");
  assert.deepEqual(calls, ["dom", "body", "safe"]);
});

test("no successful focus strategy produces a bounded precise failure", async () => {
  await assert.rejects(() => runFocusStrategies([
    { name: "dom_focus", run: async () => ({ focused: false }) },
    { name: "body_focus", run: async () => { throw new Error("x".repeat(500)); } },
  ], async () => false), (error) => {
    assert.match(error.message, /dom_focus: focus verification failed/);
    assert.ok(error.message.length < 300);
    return true;
  });
});

test("overlay diagnostics are bounded and report coverage", () => {
  const text = formatPointerInterceptionDiagnostics({
    canvasBounds: { width: 1280, height: 720 },
    blocker: {
      tag: "div",
      className: "top-section".repeat(30),
      dataHook: "top-section",
      pointerEvents: "auto",
      bounds: { width: 1280, height: 720 },
    },
    coverage: "full",
    points: [{ safe: false }],
  });
  assert.match(text, /canvas=1280x720/);
  assert.match(text, /coverage=full safe_points=0/);
  assert.ok(text.length < 400);
});

test("keyboard verification observes real down/up plumbing", async () => {
  const calls = [];
  let evaluations = 0;
  const frame = {
    evaluate: async () => {
      evaluations += 1;
      return evaluations === 2;
    },
  };
  const keyboard = {
    down: async (key) => calls.push(["down", key]),
    up: async (key) => calls.push(["up", key]),
  };
  assert.equal(await verifyKeyboardDelivery(frame, keyboard), true);
  assert.deepEqual(calls, [["down", "ArrowLeft"], ["up", "ArrowLeft"]]);
});

test("keyboard verification always releases its probe key on failure", async () => {
  const calls = [];
  const frame = { evaluate: async () => undefined };
  const keyboard = {
    down: async () => { calls.push("down"); throw new Error("delivery failed"); },
    up: async () => calls.push("up"),
  };
  await assert.rejects(() => verifyKeyboardDelivery(frame, keyboard), /delivery failed/);
  assert.deepEqual(calls, ["down", "up"]);
});
