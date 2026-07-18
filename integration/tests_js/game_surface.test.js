import assert from "node:assert/strict";
import test from "node:test";

import {
  formatCanvasDiagnostics,
  gameCanvasDiagnosticError,
  selectGameCanvasCandidate,
} from "../browser/game_surface.js";

const canvas = (
  name,
  { width, height, visible = true, attached = true, domWidth = width, domHeight = height },
) => ({
  name,
  index: 0,
  attached,
  visible,
  domWidth: String(domWidth),
  domHeight: String(domHeight),
  box: width === null || height === null ? null : { width, height },
});

test("selects one valid large canvas", () => {
  const game = canvas("game", { width: 800, height: 400 });
  assert.equal(selectGameCanvasCandidate([game]), game);
});

test("ignores a 32x64 utility canvas beside the game canvas", () => {
  const utility = canvas("utility", { width: 32, height: 64 });
  const game = canvas("game", { width: 800, height: 400 });
  assert.equal(selectGameCanvasCandidate([utility, game]), game);
});

test("selects the largest of multiple visible valid canvases", () => {
  const medium = canvas("medium", { width: 640, height: 360 });
  const largest = canvas("largest", { width: 960, height: 540 });
  assert.equal(selectGameCanvasCandidate([medium, largest]), largest);
});

test("ignores hidden, detached, and zero-size canvases", () => {
  const hidden = canvas("hidden", { width: 1000, height: 600, visible: false });
  const detached = canvas("detached", { width: 1200, height: 700, attached: false });
  const zero = canvas("zero", { width: 0, height: 0 });
  const game = canvas("game", { width: 600, height: 300 });
  assert.equal(selectGameCanvasCandidate([hidden, detached, zero, game]), game);
});

test("failure diagnostics list every discovered canvas", () => {
  const utility = { ...canvas("utility", { width: 32, height: 64 }), index: 0 };
  const hidden = { ...canvas("hidden", { width: 800, height: 400, visible: false }), index: 1 };
  const error = gameCanvasDiagnosticError([utility, hidden]);
  assert.match(error.message, /minimum rendered size 300x150/);
  assert.match(error.message, /index=0 dom=32x64 rendered=32\.0x64\.0 visible=true/);
  assert.match(error.message, /index=1 dom=800x400 rendered=800\.0x400\.0 visible=false/);
  assert.equal(formatCanvasDiagnostics([]).includes("no canvas elements discovered"), true);
});

test("selection does not depend on DOM order", () => {
  const small = canvas("small", { width: 400, height: 200 });
  const largest = canvas("largest", { width: 900, height: 500 });
  const utility = canvas("utility", { width: 32, height: 64 });
  assert.equal(selectGameCanvasCandidate([small, largest, utility]).name, "largest");
  assert.equal(selectGameCanvasCandidate([utility, largest, small]).name, "largest");
  assert.equal(selectGameCanvasCandidate([largest, small, utility]).name, "largest");
});

test("canvas discovery is unaffected by a pointer-intercepting overlay", () => {
  const game = {
    ...canvas("game", { width: 1280, height: 720 }),
    pointerIntercepted: true,
    blocker: ".top-section",
  };
  assert.equal(selectGameCanvasCandidate([game]), game);
});
