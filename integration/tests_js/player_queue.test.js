import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../headless_host/player_queue.js", import.meta.url), "utf8");

function createQueue(options) {
  const window = {};
  vm.runInNewContext(source, { window, Set });
  const queue = new window.HaxballPlayerQueue.PublicPlayerQueue(options);
  queue.setBotReady(true);
  return { queue, ...window.HaxballPlayerQueue };
}

function finishOfficialMatch(queue) {
  assert.equal(queue.beginMatch(), true);
  assert.equal(queue.completeMatch(), true);
  return queue.stopMatch();
}

const waiting = (queue) => Array.from(queue.waitingIds);

test("first human becomes active and later humans enter FIFO order", () => {
  const { queue } = createQueue();
  assert.equal(queue.addHuman(1).active, true);
  assert.equal(queue.addHuman(2).position, 1);
  assert.equal(queue.addHuman(3).position, 2);
  assert.deepEqual(waiting(queue), [2, 3]);
});

test("official match completion rotates outgoing active human to the tail", () => {
  const { queue } = createQueue();
  [1, 2, 3].forEach((id) => queue.addHuman(id));
  const transition = finishOfficialMatch(queue);
  assert.deepEqual([transition.outgoing, transition.promoted], [1, 2]);
  assert.equal(queue.activeHumanId, 2);
  assert.deepEqual(waiting(queue), [3, 1]);
});

test("active human receives an immediate rematch when nobody waits", () => {
  const { queue } = createQueue();
  queue.addHuman(1);
  const transition = finishOfficialMatch(queue);
  assert.equal(transition.rotated, false);
  assert.equal(queue.activeHumanId, 1);
  assert.deepEqual(waiting(queue), []);
});

test("queued disconnect is removed without changing the active match", () => {
  const { queue } = createQueue();
  [1, 2, 3].forEach((id) => queue.addHuman(id));
  queue.beginMatch();
  const result = queue.removeHuman(2);
  assert.equal(result.wasQueued, true);
  assert.equal(queue.activeHumanId, 1);
  assert.deepEqual(waiting(queue), [3]);
  assert.equal(queue.gameRunning, true);
});

test("active disconnect promotes only after the interrupted game stops", () => {
  const { queue } = createQueue();
  [1, 2, 3].forEach((id) => queue.addHuman(id));
  queue.beginMatch();
  assert.equal(queue.removeHuman(1).wasActive, true);
  assert.equal(queue.activeHumanId, null);
  assert.deepEqual(waiting(queue), [2, 3]);
  const transition = queue.stopMatch();
  assert.equal(transition.promoted, 2);
  assert.deepEqual(waiting(queue), [3]);
});

test("duplicate joins never create duplicate queue entries", () => {
  const { queue } = createQueue();
  queue.addHuman(1);
  queue.addHuman(2);
  assert.equal(queue.addHuman(2).duplicate, true);
  assert.deepEqual(waiting(queue), [2]);
});

test("matches-per-turn delays rotation until the configured official result", () => {
  const { queue } = createQueue({ enabled: true, matchesPerTurn: 2 });
  queue.addHuman(1);
  queue.addHuman(2);
  finishOfficialMatch(queue);
  assert.equal(queue.activeHumanId, 1);
  assert.deepEqual(waiting(queue), [2]);
  finishOfficialMatch(queue);
  assert.equal(queue.activeHumanId, 2);
  assert.deepEqual(waiting(queue), [1]);
});

test("disabled rotation preserves the original active human", () => {
  const { queue } = createQueue({ enabled: false, matchesPerTurn: 1 });
  queue.addHuman(1);
  queue.addHuman(2);
  finishOfficialMatch(queue);
  assert.equal(queue.activeHumanId, 1);
  assert.deepEqual(waiting(queue), [2]);
});

test("duplicate completion and stop callbacks are idempotent", () => {
  const { queue } = createQueue();
  queue.addHuman(1);
  queue.addHuman(2);
  queue.beginMatch();
  assert.equal(queue.completeMatch(), true);
  assert.equal(queue.completeMatch(), false);
  assert.equal(queue.stopMatch().rotated, true);
  assert.equal(queue.stopMatch().stopped, false);
  assert.equal(queue.activeHumanId, 2);
  assert.deepEqual(waiting(queue), [1]);
});

test("bot readiness changes preserve active and queued humans", () => {
  const { queue, QueueStates } = createQueue();
  [1, 2, 3].forEach((id) => queue.addHuman(id));
  queue.setBotReady(false);
  assert.equal(queue.state, QueueStates.WAITING_FOR_BOT);
  assert.equal(queue.activeHumanId, 1);
  assert.deepEqual(waiting(queue), [2, 3]);
  queue.setBotReady(true);
  assert.equal(queue.state, QueueStates.READY);
});

test("deterministic randomized lifecycle preserves every invariant", () => {
  const { queue } = createQueue({ enabled: true, matchesPerTurn: 3 });
  let randomState = 0x5eed1234;
  let nextId = 1;
  const random = () => {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState / 0x1_0000_0000;
  };
  for (let index = 0; index < 10_000; index += 1) {
    const operation = Math.floor(random() * 7);
    const connected = [...queue.connectedHumanIds];
    if (operation === 0) queue.addHuman(nextId++);
    else if (operation === 1 && connected.length) {
      queue.removeHuman(connected[Math.floor(random() * connected.length)]);
    } else if (operation === 2) queue.setBotReady(random() >= 0.25);
    else if (operation === 3) queue.beginMatch();
    else if (operation === 4) queue.completeMatch();
    else if (operation === 5) queue.stopMatch();
    else queue.reconcilePlayerQueue({ promote: !queue.gameRunning });
    assert.equal(queue.assertInvariants(), true);
  }
});
