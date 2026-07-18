export const DEFAULT_ACTION_WATCHDOG_MS = 750;

export class ControlledActionWatchdog {
  constructor({ thresholdMilliseconds = DEFAULT_ACTION_WATCHDOG_MS, now = Date.now } = {}) {
    this.thresholdMilliseconds = thresholdMilliseconds;
    this.now = now;
    this.reset();
  }

  reset(lifecycleId = null) {
    this.lifecycleId = lifecycleId;
    this.active = false;
    this.armedAt = null;
    this.lastAppliedAt = null;
    this.requestOutstanding = false;
  }

  observeState(state) {
    const lifecycleId = Number.isInteger(state.lifecycle_id) ? state.lifecycle_id : 0;
    const controlledOnRed = state.game_active === true && state.controlled?.team === 1;
    if (!controlledOnRed) {
      this.reset(lifecycleId);
      return;
    }
    if (this.lifecycleId !== lifecycleId || !this.active) {
      this.lifecycleId = lifecycleId;
      this.active = true;
      this.armedAt = this.now();
      this.lastAppliedAt = null;
      this.requestOutstanding = false;
    }
  }

  observeActionApplied(payload) {
    const lifecycleId = Number.isInteger(payload.lifecycle_id) ? payload.lifecycle_id : 0;
    if (!this.active || payload.client !== "controlled" || lifecycleId !== this.lifecycleId) return;
    this.lastAppliedAt = this.now();
    this.requestOutstanding = false;
  }

  readinessOpened() {
    if (!this.active) return;
    this.armedAt = this.now();
    this.lastAppliedAt = null;
    this.requestOutstanding = false;
  }

  poll(ready) {
    if (!ready || !this.active || this.requestOutstanding || this.armedAt === null) return null;
    const baseline = this.lastAppliedAt ?? this.armedAt;
    if (this.now() - baseline < this.thresholdMilliseconds) return null;
    this.requestOutstanding = true;
    return { lifecycleId: this.lifecycleId, reason: "no_fresh_controlled_action" };
  }
}
