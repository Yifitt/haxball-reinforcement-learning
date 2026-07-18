import { getAction } from "../shared/actions.js";

export const DEFAULT_KEY_MAP = Object.freeze({
  up: "ArrowUp",
  down: "ArrowDown",
  left: "ArrowLeft",
  right: "ArrowRight",
  kick: "x",
});

const delay = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

export class InputController {
  constructor(
    keyboard,
    {
      keyMap = DEFAULT_KEY_MAP,
      kickMilliseconds = 35,
      sleep = delay,
      clientId = "controlled",
      ready = true,
    } = {},
  ) {
    this.keyboard = keyboard;
    this.clientId = clientId;
    this.keyMap = keyMap;
    this.kickMilliseconds = kickMilliseconds;
    this.sleep = sleep;
    this.held = new Set();
    this.kickHeld = false;
    this.ready = ready;
    this.focusState = null;
    this.lastAppliedAction = null;
    this.appliedActionCount = 0;
    this.error = null;
  }

  setReady(ready, focusState = this.focusState) {
    this.ready = ready === true;
    this.focusState = focusState;
  }

  async applyAction(actionId) {
    try {
      await this.applyActionUnsafe(actionId);
    } catch (error) {
      this.error = error;
      throw error;
    }
  }

  async applyActionUnsafe(actionId) {
    if (!this.ready) throw new Error(`${this.clientId} input controller is not ready`);
    const action = getAction(actionId);
    const desired = new Set(action.keys.map((key) => this.keyMap[key]));

    for (const key of [...this.held]) {
      if (!desired.has(key)) {
        await this.keyboard.up(key);
        this.held.delete(key);
      }
    }
    for (const key of desired) {
      if (!this.held.has(key)) {
        await this.keyboard.down(key);
        this.held.add(key);
      }
    }
    if (action.kick && !this.kickHeld) {
      this.kickHeld = true;
      await this.keyboard.down(this.keyMap.kick);
      try {
        await this.sleep(this.kickMilliseconds);
      } finally {
        await this.keyboard.up(this.keyMap.kick);
        this.kickHeld = false;
      }
    }
    this.lastAppliedAction = actionId;
    this.appliedActionCount += 1;
    this.error = null;
  }

  async releaseAll() {
    const keys = [...this.held];
    this.held.clear();
    if (this.kickHeld) {
      keys.push(this.keyMap.kick);
      this.kickHeld = false;
    }
    this.lastAppliedAction = null;
    for (const key of new Set(keys)) {
      try {
        await this.keyboard.up(key);
      } catch (error) {
        this.error = error;
        throw error;
      }
    }
  }
}
