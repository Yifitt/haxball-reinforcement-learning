export const BOT_RECONNECT_DELAYS_MS = Object.freeze([1_000, 2_000, 5_000, 10_000]);

const defaultSleep = (milliseconds) =>
  new Promise((resolveSleep) => setTimeout(resolveSleep, milliseconds));

export class BotReconnectCoordinator {
  constructor({ reconnect, reset, sleep = defaultSleep, log = console.log } = {}) {
    if (typeof reconnect !== "function" || typeof reset !== "function") {
      throw new TypeError("reconnect and reset callbacks are required");
    }
    this.reconnect = reconnect;
    this.reset = reset;
    this.sleep = sleep;
    this.log = log;
    this.inFlight = null;
    this.stopped = false;
  }

  request(reason = "unhealthy") {
    if (this.stopped) return Promise.resolve(false);
    if (this.inFlight) return this.inFlight;
    this.inFlight = this.#run(reason).finally(() => {
      this.inFlight = null;
    });
    return this.inFlight;
  }

  async #run(reason) {
    this.log(`bot_disconnected reason=${reason}`);
    await this.reset(reason);
    for (let index = 0; index < BOT_RECONNECT_DELAYS_MS.length; index += 1) {
      if (this.stopped) return false;
      const delay = BOT_RECONNECT_DELAYS_MS[index];
      await this.sleep(delay);
      if (this.stopped) return false;
      this.log(`bot_reconnect_attempt attempt=${index + 1} delay_ms=${delay}`);
      try {
        await this.reconnect(index + 1);
        return true;
      } catch (error) {
        this.log(`bot_reconnect_failed attempt=${index + 1} reason=${error.message}`);
      }
    }
    return false;
  }

  stop() {
    this.stopped = true;
  }
}
