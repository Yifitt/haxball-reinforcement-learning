import assert from "node:assert/strict";
import test from "node:test";

import { validateRoomUrl } from "../browser/room_url.js";

test("validates a private HaxBall room URL without exposing its code", () => {
  const result = validateRoomUrl("https://www.haxball.com/play?c=private-code");
  assert.deepEqual(result, {
    host: "www.haxball.com",
    path: "/play",
    roomCodePresent: true,
    roomCodeLength: 12,
    roomCodeHashMatch: true,
  });
  assert.equal(JSON.stringify(result).includes("private-code"), false);
});

test("rejects empty and double-encoded room codes", () => {
  assert.throws(
    () => validateRoomUrl("https://www.haxball.com/play?c="),
    /exactly one non-empty room code/,
  );
  assert.throws(
    () => validateRoomUrl("https://www.haxball.com/play?c=private%252Fcode"),
    /encoded more than once/,
  );
});

test("detects a navigation code that differs from the host-provided code", () => {
  const result = validateRoomUrl(
    "https://www.haxball.com/play?c=navigation-code",
    "https://www.haxball.com/play?c=host-code",
  );
  assert.equal(result.roomCodeHashMatch, false);
});

test("requires the exact official HTTPS play endpoint", () => {
  assert.throws(() => validateRoomUrl("http://www.haxball.com/play?c=code"), /HTTPS/);
  assert.throws(() => validateRoomUrl("https://haxball.com/play?c=code"), /www\.haxball\.com/);
  assert.throws(() => validateRoomUrl("https://www.haxball.com/headless?c=code"), /path/);
});
