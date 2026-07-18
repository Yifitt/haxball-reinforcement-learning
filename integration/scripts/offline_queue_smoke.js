import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../headless_host/player_queue.js", import.meta.url), "utf8");
const window = {};
vm.runInNewContext(source, { window, Set });
const { PublicPlayerQueue, QueueStates } = window.HaxballPlayerQueue;
const queue = new PublicPlayerQueue({ enabled: true, matchesPerTurn: 1 });

queue.setBotReady(true);
queue.addHuman(10);
queue.addHuman(20);
queue.addHuman(30);
assert.equal(queue.beginMatch(), true);
assert.equal(queue.state, QueueStates.MATCH_RUNNING);
assert.equal(queue.completeMatch(), true);
const firstRotation = queue.stopMatch();
assert.deepEqual(
  [firstRotation.outgoing, firstRotation.promoted, queue.activeHumanId, ...queue.waitingIds],
  [10, 20, 20, 30, 10],
);

assert.equal(queue.beginMatch(), true);
queue.removeHuman(20);
assert.equal(queue.activeHumanId, null);
assert.equal(queue.stopMatch().promoted, 30);
assert.deepEqual(Array.from(queue.waitingIds), [10]);
queue.setBotReady(false);
assert.equal(queue.state, QueueStates.WAITING_FOR_BOT);
queue.setBotReady(true);
assert.equal(queue.state, QueueStates.READY);
assert.equal(queue.assertInvariants(), true);

console.log(JSON.stringify({
  queue_smoke: true,
  active_human_id: queue.activeHumanId,
  waiting_human_ids: queue.waitingIds,
  state: queue.state,
}));
