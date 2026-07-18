export const PROTOCOL_VERSION = 1;
export const MESSAGE_TYPES = new Set([
  "hello",
  "room_status",
  "client_status",
  "readiness",
  "state",
  "action",
  "action_applied",
  "state_request",
  "reset",
  "error",
  "shutdown",
]);

export function message(type, fields = {}) {
  if (!MESSAGE_TYPES.has(type)) {
    throw new Error(`unknown message type: ${type}`);
  }
  return { protocol_version: PROTOCOL_VERSION, type, ...fields };
}

export function parseMessage(data) {
  let value;
  try {
    value = JSON.parse(data.toString());
  } catch {
    throw new Error("invalid JSON");
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("message must be an object");
  }
  if (value.protocol_version !== PROTOCOL_VERSION) {
    throw new Error("unsupported protocol version");
  }
  if (!MESSAGE_TYPES.has(value.type)) {
    throw new Error("unknown message type");
  }
  return value;
}

export class ActionGate {
  constructor(readiness = null) {
    this.readiness = readiness;
    this.latestTick = -1;
    this.latestLifecycle = 0;
    this.lastActionTick = { controlled: -1, opponent: -1 };
    this.gameActive = false;
    this.clientReady = false;
    this.stale = 0;
    this.invalid = 0;
    this.duplicates = 0;
  }

  observeState(state) {
    const lifecycle = Number.isInteger(state.lifecycle_id) ? state.lifecycle_id : 0;
    if (lifecycle < this.latestLifecycle) return false;
    if (lifecycle > this.latestLifecycle) this.resetLifecycle(lifecycle);
    if (!Number.isInteger(state.tick_id) || state.tick_id <= this.latestTick) {
      return false;
    }
    this.latestTick = state.tick_id;
    this.gameActive = state.game_active === true;
    return true;
  }

  setClientReady(ready) {
    this.clientReady = ready === true;
  }

  accept(action) {
    if (!["controlled", "opponent"].includes(action.client)) {
      this.invalid += 1;
      return { ok: false, reason: "unknown_client" };
    }
    if (!Number.isInteger(action.action) || action.action < 0 || action.action > 17) {
      this.invalid += 1;
      return { ok: false, reason: "invalid_action" };
    }
    const lifecycle = Number.isInteger(action.lifecycle_id)
      ? action.lifecycle_id : this.latestLifecycle;
    if (lifecycle !== this.latestLifecycle) {
      this.stale += 1;
      return { ok: false, reason: "stale_lifecycle" };
    }
    if (!Number.isInteger(action.tick_id) || action.tick_id < this.latestTick) {
      this.stale += 1;
      return { ok: false, reason: "stale_tick" };
    }
    if (action.tick_id === this.lastActionTick[action.client]) {
      this.duplicates += 1;
      return { ok: false, reason: "duplicate_action" };
    }
    if (action.tick_id !== this.latestTick) {
      this.invalid += 1;
      return { ok: false, reason: "unknown_tick" };
    }
    const barrierReady = this.readiness
      ? this.readiness.isReady() && this.readiness.isClientInputReady(action.client)
      : this.clientReady && this.gameActive;
    if (!barrierReady) {
      this.invalid += 1;
      return {
        ok: false,
        reason: "not_ready",
        diagnostic: this.readiness?.notReadyDiagnostic("apply_action", action.client),
      };
    }
    this.lastActionTick[action.client] = action.tick_id;
    return { ok: true };
  }

  resetActions() {
    this.lastActionTick = { controlled: -1, opponent: -1 };
  }

  resetLifecycle(lifecycle = this.latestLifecycle) {
    this.latestLifecycle = Number.isInteger(lifecycle) ? lifecycle : this.latestLifecycle;
    this.latestTick = -1;
    this.gameActive = false;
    this.resetActions();
  }
}

export class BridgeMetrics {
  constructor() {
    this.startedAt = Date.now();
    this.states = 0;
    this.actions = { controlled: 0, opponent: 0 };
    this.actionApplied = { controlled: 0, opponent: 0 };
    this.rejected = { controlled: 0, opponent: 0 };
    this.disconnects = 0;
    this.preReadyStates = 0;
    this.preReadyActions = 0;
    this.latencies = { controlled: [], opponent: [] };
    this.pairedApplicationDifferences = [];
  }

  recordLatency(client, milliseconds) {
    if (Number.isFinite(milliseconds) && milliseconds >= 0) {
      this.latencies[client].push(milliseconds);
      if (this.latencies[client].length > 4096) this.latencies[client].shift();
    }
  }

  recordPairedDifference(milliseconds) {
    if (Number.isFinite(milliseconds) && milliseconds >= 0) {
      this.pairedApplicationDifferences.push(milliseconds);
      if (this.pairedApplicationDifferences.length > 4096) {
        this.pairedApplicationDifferences.shift();
      }
    }
  }

  median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
  }

  report(now = Date.now()) {
    const seconds = Math.max((now - this.startedAt) / 1000, 1e-9);
    return {
      elapsed_seconds: seconds,
      state_messages_per_second: this.states / seconds,
      controlled_actions_per_second: this.actions.controlled / seconds,
      opponent_actions_per_second: this.actions.opponent / seconds,
      controlled_applied_actions_per_second: this.actionApplied.controlled / seconds,
      opponent_applied_actions_per_second: this.actionApplied.opponent / seconds,
      controlled_rejected_actions: this.rejected.controlled,
      opponent_rejected_actions: this.rejected.opponent,
      median_state_to_controlled_input_ms: this.median(this.latencies.controlled),
      median_state_to_opponent_input_ms: this.median(this.latencies.opponent),
      median_paired_application_difference_ms: this.median(this.pairedApplicationDifferences),
      disconnects: this.disconnects,
      pre_ready_states: this.preReadyStates,
      pre_ready_actions: this.preReadyActions,
    };
  }
}
