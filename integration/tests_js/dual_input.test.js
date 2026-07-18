import assert from "node:assert/strict";
import test from "node:test";

import { DualInputRouter } from "../browser/dual_input.js";

class MockController {
  constructor(name) { this.name = name; this.actions = []; this.releases = 0; this.ready = false; }
  setReady(ready) { this.ready = ready; }
  async applyAction(action) { this.actions.push(action); }
  async releaseAll() { this.releases += 1; }
}

test("targeted actions route only to their client controller", async () => {
  const router = new DualInputRouter();
  const controlled = new MockController("controlled");
  const opponent = new MockController("opponent");
  router.register("controlled", controlled, true);
  router.register("opponent", opponent, true);
  const first = router.enqueue({ client: "controlled", action: 4 });
  const second = router.enqueue({ client: "opponent", action: 3 });
  await Promise.all([first.completion, second.completion]);
  assert.deepEqual(controlled.actions, [4]);
  assert.deepEqual(opponent.actions, [3]);
});

test("releasing one routed client does not release the other", async () => {
  const router = new DualInputRouter();
  const controlled = new MockController("controlled");
  const opponent = new MockController("opponent");
  router.register("controlled", controlled, true);
  router.register("opponent", opponent, true);
  await router.release("controlled");
  assert.equal(controlled.releases, 1);
  assert.equal(opponent.releases, 0);
});

test("unknown and not-ready target clients are rejected independently", () => {
  const router = new DualInputRouter();
  router.register("controlled", new MockController("controlled"), true);
  assert.equal(router.enqueue({ client: "opponent", action: 3 }).reason, "input_not_ready");
  assert.equal(router.enqueue({ client: "unknown", action: 3 }).reason, "unknown_client");
  assert.equal(router.ready.controlled, true);
});

test("episode reset releases both client controllers", async () => {
  const router = new DualInputRouter();
  const controlled = new MockController("controlled");
  const opponent = new MockController("opponent");
  router.register("controlled", controlled, true);
  router.register("opponent", opponent, true);
  await router.resetAll();
  assert.equal(controlled.releases, 1);
  assert.equal(opponent.releases, 1);
});

test("reset invalidates queued callbacks and releases held inputs before rematch", async () => {
  const router = new DualInputRouter();
  const controlled = new MockController("controlled");
  router.register("controlled", controlled, true);
  const stale = router.enqueue({ client: "controlled", action: 4 });
  await router.resetAll();
  await stale.completion;
  assert.deepEqual(controlled.actions, []);
  assert.equal(controlled.releases, 1);
  const fresh = router.enqueue({ client: "controlled", action: 3 });
  await fresh.completion;
  assert.deepEqual(controlled.actions, [3]);
});
