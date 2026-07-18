import assert from "node:assert/strict";
import test from "node:test";

import { hostConfigFromEnv } from "../headless_host/config.js";
import { bridgeConfigFromEnv } from "../bridge/config.js";

test("public plus human opponent resolves explicit public queue startup mode", () => {
  const prior = { ...process.env };
  try {
    process.env.HAXBALL_HEADLESS_TOKEN = "test-token-placeholder";
    process.env.HAXBALL_PUBLIC_ROOM = "1";
    process.env.HAXBALL_HUMAN_OPPONENT = "1";
    process.env.HAXBALL_OPPONENT_POLICY = "stationary";
    process.env.HAXBALL_STARTUP_MODE = "public_human_queue";
    const config = hostConfigFromEnv();
    assert.equal(config.startupMode, "public_human_queue");
    assert.equal(config.publicRoom, true);
    assert.equal(config.humanOpponent, true);
    assert.equal(config.sourcePublicRoom, true);
    assert.equal(config.sourceHumanOpponent, true);
  } finally {
    for (const key of Object.keys(process.env)) {
      if (!(key in prior)) delete process.env[key];
    }
    Object.assign(process.env, prior);
  }
});

test("public mode rejects a conflicting stationary startup override", () => {
  const prior = { ...process.env };
  try {
    process.env.HAXBALL_HEADLESS_TOKEN = "test-token-placeholder";
    process.env.HAXBALL_PUBLIC_ROOM = "1";
    process.env.HAXBALL_HUMAN_OPPONENT = "1";
    process.env.HAXBALL_STARTUP_MODE = "stationary_opponent";
    assert.throws(() => hostConfigFromEnv(), /conflicts with resolved mode public_human_queue/);
  } finally {
    for (const key of Object.keys(process.env)) {
      if (!(key in prior)) delete process.env[key];
    }
    Object.assign(process.env, prior);
  }
});

test("bridge explicit public mode supplies authoritative flags without telemetry", () => {
  const config = bridgeConfigFromEnv({ HAXBALL_STARTUP_MODE: "public_human_queue" });
  assert.deepEqual(config, {
    startupMode: "public_human_queue",
    publicRoom: true,
    humanOpponent: true,
    opponentControlRequired: false,
  });
});

test("bridge rejects invalid modes and explicit flags that conflict with the mode", () => {
  assert.throws(
    () => bridgeConfigFromEnv({ HAXBALL_STARTUP_MODE: "unresolved" }),
    /invalid HAXBALL_STARTUP_MODE/,
  );
  assert.throws(
    () => bridgeConfigFromEnv({
      HAXBALL_STARTUP_MODE: "public_human_queue",
      HAXBALL_PUBLIC_ROOM: "0",
    }),
    /conflicts with HAXBALL_STARTUP_MODE public_human_queue/,
  );
});
