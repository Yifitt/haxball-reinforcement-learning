export const CLIENT_ROLES = Object.freeze(["controlled", "opponent"]);

export class DualInputRouter {
  constructor() {
    this.controllers = { controlled: null, opponent: null };
    this.ready = { controlled: false, opponent: false };
    this.queues = { controlled: Promise.resolve(), opponent: Promise.resolve() };
    this.generations = { controlled: 0, opponent: 0 };
  }

  register(client, controller, ready = false) {
    if (!CLIENT_ROLES.includes(client)) throw new Error(`unknown input client: ${client}`);
    this.controllers[client] = controller;
    this.ready[client] = ready === true;
    controller.setReady?.(ready);
    this.generations[client] += 1;
    this.queues[client] = Promise.resolve();
  }

  setReady(client, ready) {
    if (!CLIENT_ROLES.includes(client)) throw new Error(`unknown input client: ${client}`);
    this.ready[client] = ready === true;
    this.controllers[client]?.setReady(ready);
  }

  enqueue(payload, onApplied = async () => {}) {
    const client = payload.client;
    if (!CLIENT_ROLES.includes(client)) return { accepted: false, reason: "unknown_client" };
    const controller = this.controllers[client];
    if (!this.ready[client] || !controller) return { accepted: false, reason: "input_not_ready" };
    const generation = this.generations[client];
    const completion = this.queues[client].then(async () => {
      if (generation !== this.generations[client]) return;
      await controller.applyAction(payload.action);
      if (generation !== this.generations[client]) return;
      await onApplied(payload);
    });
    this.queues[client] = completion.catch(() => {});
    return { accepted: true, client, controller, completion };
  }

  async release(client) {
    if (!CLIENT_ROLES.includes(client)) throw new Error(`unknown input client: ${client}`);
    await this.controllers[client]?.releaseAll();
  }

  async releaseAll() {
    await Promise.all(CLIENT_ROLES.map((client) =>
      this.controllers[client]?.releaseAll().catch(() => {})));
  }

  async resetAll() {
    for (const client of CLIENT_ROLES) this.generations[client] += 1;
    const pending = CLIENT_ROLES.map((client) => this.queues[client]);
    await Promise.allSettled(pending);
    await this.releaseAll();
    for (const client of CLIENT_ROLES) this.queues[client] = Promise.resolve();
  }
}
