import assert from "node:assert/strict";
import test from "node:test";

import {
  closeBrowserResources,
  joinClientPair,
  joinConfiguredClients,
} from "../browser/client_startup.js";

test("client joins are staggered without waiting for the first to finish", async () => {
  const started = [];
  const delays = [];
  let releaseBoth;
  const bothStarted = new Promise((resolve) => { releaseBoth = resolve; });
  const joinClient = async (name) => {
    started.push(name);
    if (started.length === 2) releaseBoth();
    await bothStarted;
    return { name };
  };

  const result = await joinClientPair(
    joinClient,
    ["controlled"],
    ["opponent"],
    { staggerMilliseconds: 750, sleep: async (milliseconds) => delays.push(milliseconds) },
  );
  assert.deepEqual(started, ["controlled", "opponent"]);
  assert.deepEqual(delays, [750]);
  assert.equal(result.controlled.name, "controlled");
  assert.equal(result.opponent.name, "opponent");
});

test("a joined client remains connected while only the other client retries internally", async () => {
  let controlledConnected = false;
  let opponentAttempts = 0;
  const joinClient = async (name) => {
    if (name === "controlled") {
      controlledConnected = true;
      return { name, connected: true };
    }
    opponentAttempts += 1;
    assert.equal(controlledConnected, true);
    opponentAttempts += 1; // the client's bounded lifecycle retry
    assert.equal(controlledConnected, true);
    return { name, connected: true };
  };
  const result = await joinClientPair(
    joinClient, ["controlled"], ["opponent"], { staggerMilliseconds: 0 },
  );
  assert.equal(opponentAttempts, 2);
  assert.equal(result.controlled.connected, true);
});

test("shutdown closes both contexts before the browser process", async () => {
  const closed = [];
  const contexts = [
    { close: async () => closed.push("controlled-context") },
    { close: async () => { closed.push("opponent-context"); throw new Error("already closed"); } },
  ];
  const browser = { close: async () => closed.push("browser") };
  await closeBrowserResources(browser, contexts);
  assert.deepEqual(closed.slice(0, 2).sort(), ["controlled-context", "opponent-context"]);
  assert.equal(closed.at(-1), "browser");
});

test("human mode launches only the controlled browser client", async () => {
  const joined = [];
  const result = await joinConfiguredClients(
    async (name) => { joined.push(name); return { name }; },
    ["controlled"],
    ["scripted-opponent"],
    { humanOpponent: true },
  );
  assert.deepEqual(joined, ["controlled"]);
  assert.equal(result.controlled.name, "controlled");
  assert.equal(result.opponent, null);
});
