export const STARTUP_MODE_CONFIG = Object.freeze({
  stationary_opponent: Object.freeze({
    publicRoom: false,
    humanOpponent: false,
    opponentControlRequired: false,
  }),
  dual_control: Object.freeze({
    publicRoom: false,
    humanOpponent: false,
    opponentControlRequired: true,
  }),
  human_opponent: Object.freeze({
    publicRoom: false,
    humanOpponent: true,
    opponentControlRequired: false,
  }),
  public_human_queue: Object.freeze({
    publicRoom: true,
    humanOpponent: true,
    opponentControlRequired: false,
  }),
});

export function startupConfigForMode(startupMode) {
  const flags = STARTUP_MODE_CONFIG[startupMode];
  if (!flags) {
    throw new Error(
      `invalid HAXBALL_STARTUP_MODE ${startupMode}; expected ${Object.keys(STARTUP_MODE_CONFIG).join(", ")}`,
    );
  }
  return Object.freeze({ startupMode, ...flags });
}

function explicitBoolean(environment, name) {
  if (!(name in environment)) return undefined;
  if (!["0", "1"].includes(environment[name])) {
    throw new Error(`${name} must be 0 or 1`);
  }
  return environment[name] === "1";
}

export function bridgeConfigFromEnv(environment = process.env) {
  const suppliedMode = environment.HAXBALL_STARTUP_MODE;
  const publicRoom = explicitBoolean(environment, "HAXBALL_PUBLIC_ROOM");
  const humanOpponent = explicitBoolean(environment, "HAXBALL_HUMAN_OPPONENT");
  const opponentControlRequired = explicitBoolean(
    environment, "HAXBALL_OPPONENT_CONTROL_REQUIRED",
  );
  const opponentPolicy = environment.HAXBALL_OPPONENT_POLICY ?? "stationary";
  const inferredMode = publicRoom === true
    ? "public_human_queue"
    : humanOpponent === true
      ? "human_opponent"
      : (opponentControlRequired === true || opponentPolicy !== "stationary")
        ? "dual_control" : "stationary_opponent";
  const config = startupConfigForMode(suppliedMode ?? inferredMode);
  const suppliedFlags = { publicRoom, humanOpponent, opponentControlRequired };
  for (const [name, value] of Object.entries(suppliedFlags)) {
    if (value !== undefined && value !== config[name]) {
      throw new Error(
        `${name}=${value} conflicts with HAXBALL_STARTUP_MODE ${config.startupMode}`,
      );
    }
  }
  return config;
}
