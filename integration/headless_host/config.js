export function hostConfigFromEnv() {
  const token = process.env.HAXBALL_HEADLESS_TOKEN;
  if (!token) {
    throw new Error("HAXBALL_HEADLESS_TOKEN is required for real-room launch");
  }
  const publicRoom = process.env.HAXBALL_PUBLIC_ROOM === "1";
  const configuredHumanOpponent = process.env.HAXBALL_HUMAN_OPPONENT === "1";
  const opponentPolicy = process.env.HAXBALL_OPPONENT_POLICY ?? "stationary";
  const resolvedStartupMode = publicRoom
    ? "public_human_queue"
    : configuredHumanOpponent
      ? "human_opponent"
      : opponentPolicy !== "stationary" ? "dual_control" : "stationary_opponent";
  const suppliedStartupMode = process.env.HAXBALL_STARTUP_MODE;
  if (suppliedStartupMode && suppliedStartupMode !== resolvedStartupMode) {
    throw new Error(
      `HAXBALL_STARTUP_MODE ${suppliedStartupMode} conflicts with resolved mode ${resolvedStartupMode}`,
    );
  }
  const integer = (name, fallback, minimum, maximum) => {
    const value = Number(process.env[name] ?? fallback);
    if (!Number.isInteger(value) || value < minimum || value > maximum) {
      throw new Error(`${name} must be an integer from ${minimum} to ${maximum}`);
    }
    return value;
  };
  const queueAfkTimeout = Number(process.env.HAXBALL_QUEUE_AFK_TIMEOUT ?? 0);
  if (queueAfkTimeout !== 0) {
    throw new Error("HAXBALL_QUEUE_AFK_TIMEOUT must be 0; AFK detection is disabled");
  }
  return {
    token,
    roomName: process.env.HAXBALL_ROOM_NAME ?? (publicRoom ? "RL Bot | 1v1" : "RL Bot | Private Test"),
    password: publicRoom ? undefined : (process.env.HAXBALL_ROOM_PASSWORD || undefined),
    controlledNickname: process.env.HAXBALL_CONTROLLED_NICK ?? "RL-Agent",
    opponentNickname: process.env.HAXBALL_OPPONENT_NICK ?? "Scripted-Opponent",
    humanOpponent: publicRoom || configuredHumanOpponent,
    publicRoom,
    startupMode: resolvedStartupMode,
    sourcePublicRoom: publicRoom,
    sourceHumanOpponent: configuredHumanOpponent,
    sourceOpponentPolicy: opponentPolicy,
    maxPlayers: integer("HAXBALL_MAX_PLAYERS", publicRoom ? 12 : 2, 2, 30),
    scoreLimit: integer("HAXBALL_SCORE_LIMIT", publicRoom ? 5 : 3, 0, 14),
    timeLimit: integer("HAXBALL_TIME_LIMIT", publicRoom ? 0 : 3, 0, 60),
    maxNicknameLength: integer("HAXBALL_MAX_NICKNAME_LENGTH", 25, 1, 50),
    enablePlayerQueue: publicRoom && process.env.HAXBALL_ENABLE_PLAYER_QUEUE !== "0",
    matchesPerTurn: integer("HAXBALL_MATCHES_PER_TURN", 1, 1, 100),
    queueAfkTimeout,
    bridgeUrl: process.env.HAXBALL_BRIDGE_URL ?? "ws://127.0.0.1:8765",
    controlDecimation: Number(process.env.HAXBALL_CONTROL_DECIMATION ?? 3),
  };
}
