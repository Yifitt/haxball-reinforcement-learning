export const BASE_ACTION_READINESS_REQUIREMENTS = Object.freeze([
  ["host", (state) => state.host_connected],
  ["private_room", (state) => state.room_created],
  ["controlled_player", (state) => state.controlled_joined],
  ["opponent_player", (state) => state.opponent_joined],
  ["controlled_game_surface", (state) => state.controlled_surface_ready],
  ["opponent_game_surface", (state) => state.opponent_surface_ready],
  ["controlled_input", (state) => state.controlled_input_ready],
  ["python_controller", (state) => state.python_connected],
  ["active_game_state", (state) => state.active_game_state],
]);

export const STARTUP_MODES = Object.freeze({
  UNRESOLVED: "unresolved",
  STATIONARY_OPPONENT: "stationary_opponent",
  DUAL_CONTROL: "dual_control",
  HUMAN_OPPONENT: "human_opponent",
  PUBLIC_HUMAN_QUEUE: "public_human_queue",
});

const PUBLIC_LOBBY_REQUIREMENTS = Object.freeze([
  ["host", (state) => state.host_connected],
  ["public_room", (state) => state.public_room && state.room_created],
  ["controlled_browser", (state) => state.browser_connected],
  ["controlled_player", (state) => state.controlled_joined && state.controlled_red],
  ["python_controller", (state) => state.python_connected],
]);

const PUBLIC_ACTIVE_REQUIREMENTS = Object.freeze([
  ["opponent_player", (state) => state.opponent_joined && state.opponent_blue],
  ["controlled_game_surface", (state) => state.controlled_surface_ready],
  ["controlled_input", (state) => state.controlled_input_ready],
  ["active_game_state", (state) => state.active_game_state],
]);

const INITIAL_STATE = Object.freeze({
  bridge_listening: true,
  host_connected: false,
  room_created: false,
  controlled_joined: false,
  opponent_joined: false,
  controlled_red: false,
  opponent_blue: false,
  browser_connected: false,
  controlled_surface_ready: false,
  opponent_surface_ready: false,
  controlled_input_ready: false,
  opponent_input_ready: false,
  python_connected: false,
  python_protocol_ready: false,
  game_running: false,
  active_game_state: false,
  opponent_control_required: false,
  human_opponent: false,
  public_room: false,
});

export function resolvedStartupMode(state, explicitStartupMode = null) {
  if (explicitStartupMode !== null) return explicitStartupMode;
  if (!state.python_protocol_ready) return STARTUP_MODES.UNRESOLVED;
  if (state.public_room) return STARTUP_MODES.PUBLIC_HUMAN_QUEUE;
  if (state.human_opponent) return STARTUP_MODES.HUMAN_OPPONENT;
  if (state.opponent_control_required) return STARTUP_MODES.DUAL_CONTROL;
  return STARTUP_MODES.STATIONARY_OPPONENT;
}

export class ReadinessTracker {
  constructor(startupConfig = null) {
    this.explicitStartupMode = startupConfig?.startupMode ?? null;
    this.state = {
      ...INITIAL_STATE,
      ...(startupConfig ? {
        public_room: startupConfig.publicRoom,
        human_opponent: startupConfig.humanOpponent,
        opponent_control_required: startupConfig.opponentControlRequired,
      } : {}),
    };
    this.everReady = false;
  }

  update(fields) {
    let changed = false;
    for (const [name, value] of Object.entries(fields)) {
      if (!(name in this.state) || typeof value !== "boolean") continue;
      if (this.explicitStartupMode !== null && [
        "public_room", "human_opponent", "opponent_control_required",
      ].includes(name)) continue;
      if (this.state[name] !== value) {
        this.state[name] = value;
        changed = true;
      }
    }
    if (this.isReady()) this.everReady = true;
    return changed;
  }

  observeRoomStatus(payload) {
    const fields = {};
    if (payload.room_url) fields.room_created = true;
    if (payload.lifecycle === "player_joined" &&
        (payload.player?.client === "controlled" || payload.player?.team === 1)) {
      fields.controlled_joined = true;
      fields.controlled_red = payload.player?.team === 1;
    }
    if (payload.lifecycle === "player_joined" &&
        (payload.player?.client === "opponent" || payload.player?.team === 2)) {
      fields.opponent_joined = true;
      fields.opponent_blue = payload.player?.team === 2;
    }
    if (payload.lifecycle === "player_left" &&
        (payload.player?.client === "controlled" || payload.player?.team === 1)) {
      fields.controlled_joined = false;
      fields.controlled_red = false;
    }
    if (payload.lifecycle === "player_left" &&
        (payload.player?.client === "opponent" || payload.player?.team === 2)) {
      fields.opponent_joined = false;
      fields.opponent_blue = false;
    }
    if (payload.lifecycle === "active_episode") fields.game_running = true;
    if (["match_ended", "waiting_for_players"].includes(payload.lifecycle)) {
      fields.game_running = false;
      fields.active_game_state = false;
    }
    return this.update(fields);
  }

  requirementStatus() {
    const startupMode = resolvedStartupMode(this.state, this.explicitStartupMode);
    if (startupMode === STARTUP_MODES.UNRESOLVED) {
      const requirements = [
        ["python_controller", (state) => state.python_connected],
      ];
      const required = requirements.map(([name]) => name);
      const ready = requirements
        .filter(([, predicate]) => predicate(this.state))
        .map(([name]) => name);
      return {
        startup_mode: startupMode,
        required,
        ready,
        missing: required.filter((name) => !ready.includes(name)),
      };
    }
    if (startupMode === STARTUP_MODES.PUBLIC_HUMAN_QUEUE) {
      const requirements = this.state.game_running
        ? [...PUBLIC_LOBBY_REQUIREMENTS, ...PUBLIC_ACTIVE_REQUIREMENTS]
        : [...PUBLIC_LOBBY_REQUIREMENTS];
      const required = requirements.map(([name]) => name);
      const ready = requirements
        .filter(([, predicate]) => predicate(this.state))
        .map(([name]) => name);
      return {
        startup_mode: startupMode,
        required,
        ready,
        missing: required.filter((name) => !ready.includes(name)),
      };
    }
    let requirements = this.state.human_opponent
      ? BASE_ACTION_READINESS_REQUIREMENTS.filter(([name]) => name !== "opponent_game_surface")
      : [...BASE_ACTION_READINESS_REQUIREMENTS];
    if (this.state.human_opponent && !this.state.game_running) {
      const postStartRequirements = new Set([
        "controlled_game_surface", "controlled_input", "active_game_state",
      ]);
      requirements = requirements.filter(([name]) => !postStartRequirements.has(name));
    }
    if (this.state.opponent_control_required) {
      requirements = [...requirements, ["opponent_input", (state) => state.opponent_input_ready]];
    }
    const required = requirements.map(([name]) => name);
    const ready = requirements
      .filter(([, predicate]) => predicate(this.state))
      .map(([name]) => name);
    return {
      startup_mode: startupMode,
      required,
      ready,
      missing: required.filter((name) => !ready.includes(name)),
    };
  }

  startupHandshakeStatus() {
    const startupMode = resolvedStartupMode(this.state, this.explicitStartupMode);
    if (startupMode !== STARTUP_MODES.PUBLIC_HUMAN_QUEUE) {
      return this.requirementStatus();
    }
    const required = PUBLIC_LOBBY_REQUIREMENTS.map(([name]) => name);
    const ready = PUBLIC_LOBBY_REQUIREMENTS
      .filter(([, predicate]) => predicate(this.state))
      .map(([name]) => name);
    return {
      startup_mode: startupMode,
      required,
      ready,
      missing: required.filter((name) => !ready.includes(name)),
    };
  }

  startupTelemetryConflict(payload) {
    if (this.explicitStartupMode === null) return null;
    if (payload.startup_mode !== undefined && payload.startup_mode !== this.explicitStartupMode) {
      return "startup_mode";
    }
    for (const name of ["public_room", "human_opponent", "opponent_control_required"]) {
      if (payload[name] !== undefined && payload[name] !== this.state[name]) return name;
    }
    return null;
  }

  isReady() {
    return this.requirementStatus().missing.length === 0;
  }

  isClientInputReady(client) {
    if (client === "controlled") return this.state.controlled_input_ready;
    if (client === "opponent") {
      return this.state.opponent_control_required && this.state.opponent_input_ready;
    }
    return false;
  }

  snapshot() {
    const requirements = this.requirementStatus();
    return {
      ...this.state,
      barrier_ready: requirements.missing.length === 0,
      ...requirements,
    };
  }

  notReadyDiagnostic(operation = "apply_action", client = "controlled") {
    const requirements = this.requirementStatus();
    const clientInput = `${client}_input`;
    if (!this.isClientInputReady(client) && !requirements.missing.includes(clientInput)) {
      requirements.missing = [...requirements.missing, clientInput];
      requirements.required = [...requirements.required, clientInput];
    }
    let component = `${client}_browser`;
    if (requirements.missing.includes("python_controller")) {
      component = "python_controller";
    } else if (requirements.missing.includes("active_game_state")) {
      component = "active_game_state";
    } else if (requirements.missing.some((name) => [
      "host", "private_room", "public_room", "controlled_browser", "opponent_player",
    ].includes(name))) {
      component = "headless_host";
    }
    return { operation, client, component, ...requirements };
  }
}
