import assert from "node:assert/strict";
import test from "node:test";

import {
  captureJoinDiagnostics,
  redactDiagnosticText,
  sanitizeUrl,
} from "../browser/join_diagnostics.js";

test("diagnostic URLs redact room codes, tokens, and fragments", () => {
  const sanitized = sanitizeUrl("https://www.haxball.com/play?c=private-value&token=credential-value#fragment");
  assert.equal(sanitized.includes("private-value"), false);
  assert.equal(sanitized.includes("credential-value"), false);
  assert.equal(sanitized.includes("fragment"), false);
  const text = redactDiagnosticText("failed https://example.test/path?auth=private-value token=credential-value");
  assert.equal(text.includes("private-value"), false);
});

test("failure diagnostics stay in memory and redact private values", async () => {
  let screenshotCalled = false;
  const frame = {
    url: () => "https://www.haxball.com/game",
    evaluate: async () => ({
      index: 0,
      url: "https://www.haxball.com/game",
      loading: false,
      canvas_count: 0,
      nickname_input: true,
      inputs: [],
      buttons: ["Ok"],
      text_excerpt: "Choose nickname",
      connection_error: null,
    }),
  };
  const page = {
    url: () => "https://www.haxball.com/play?c=private-value",
    title: async () => "Haxball Play",
    mainFrame: () => frame,
    frames: () => [frame],
    screenshot: async () => { screenshotCalled = true; },
  };
  const context = { pages: () => [page] };
  const result = await captureJoinDiagnostics(context, {
    clientId: "controlled",
    stage: "FAILED",
    error: new Error("join failed at https://www.haxball.com/play?c=private-value"),
  });
  assert.equal(result.failures.length, 0);
  assert.equal(screenshotCalled, false);
  assert.equal("directory" in result, false);
  assert.equal(JSON.stringify(result).includes("private-value"), false);
});
