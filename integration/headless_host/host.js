(() => {
  "use strict";

  const VERSION = 1;
  const config = window.__HAXBALL_INTEGRATION_CONFIG__;
  if (!config) throw new Error("missing injected host configuration");
  if (!Number.isInteger(config.controlDecimation) || config.controlDecimation < 1) {
    throw new Error("controlDecimation must be a positive integer");
  }
  if (config.controlledNickname === config.opponentNickname) {
    throw new Error("controlled and opponent nicknames must be distinct for authoritative mapping");
  }

  const send = (type, fields = {}) => {
    window.__hostBridgeSend({ protocol_version: VERSION, type, ...fields })
      .catch((error) => console.error(`host_bridge_send_failed: ${error.message}`));
  };
  send("hello", { role: "host" });

  const room = window.HBInit({
    roomName: config.roomName,
    maxPlayers: config.maxPlayers ?? 2,
    public: config.publicRoom === true,
    noPlayer: true,
    password: config.password,
    token: config.token,
  });
  room.setDefaultStadium("Classic");
  room.setScoreLimit(config.scoreLimit ?? 3);
  room.setTimeLimit(config.timeLimit ?? 3);
  room.setTeamsLock(true);
  room.setRequireRecaptcha?.(false);

  const publicArena = config.publicRoom === true;
  if (publicArena && config.startupMode !== "public_human_queue") {
    throw new Error("public room requires public_human_queue startup mode");
  }
  console.log(
    `host_startup_mode mode=${config.startupMode ?? "legacy_private"} ` +
    `source_public_room=${config.publicRoom === true} ` +
    `source_human_opponent=${config.humanOpponent === true}`,
  );
  const queueApi = window.HaxballPlayerQueue;
  if (publicArena && !queueApi) throw new Error("public player queue module was not loaded");
  const playerQueue = publicArena
    ? new queueApi.PublicPlayerQueue({
      enabled: config.enablePlayerQueue !== false,
      matchesPerTurn: config.matchesPerTurn ?? 1,
    })
    : null;
  const queueAnnouncements = queueApi?.QueueAnnouncements;

  let controlledId = null;
  let opponentId = null; // Private-room opponent only; public uses playerQueue.activeHumanId.
  let reportedActiveHumanId = null;
  let tickId = 0;
  let hostTickId = 0;
  let lifecycleId = 0;
  let lifecycleStateSequence = 0;
  let gameStartCount = 0;
  let positionResetCount = 0;
  let firstStateSentCount = 0;
  let gameStartTimestamp = null;
  let positionResetTimestamp = null;
  let firstStateSentTimestamp = null;
  let gameActive = false;
  let resetting = false;
  let lastGoal = null;
  let lastTouchTeam = null;
  let startPending = false;
  let startGeneration = 0;
  let botStartupReady = !publicArena;
  let botControllerReady = !publicArena;
  let botUnavailable = publicArena;
  let botRecoveryPending = false;
  const staleBotIds = new Set();
  const lastQueuePositions = new Map();
  let metricStartedAt = performance.now();
  let metricTicks = 0;
  let metricStates = 0;

  const status = (lifecycle, fields = {}) =>
    send("room_status", { lifecycle, game_active: gameActive && !resetting, ...fields });
  const announce = (text, targetId = null) => {
    room.sendAnnouncement?.(text, targetId, 0x9fd3ff, "normal", 1);
  };
  const playerName = (id) =>
    room.getPlayer(id)?.name?.slice(0, config.maxNicknameLength ?? 25) ?? "player";
  const activeOpponentId = () => publicArena ? playerQueue.activeHumanId : opponentId;
  const playersReady = () => controlledId !== null && activeOpponentId() !== null;
  const canStart = () => {
    if (!playersReady() || gameActive || resetting) return false;
    if (!publicArena) return true;
    return botStartupReady && playerQueue.state === queueApi.QueueStates.READY;
  };

  const cancelScheduledStart = () => {
    startGeneration += 1;
    startPending = false;
  };
  const scheduleStart = () => {
    if (!canStart() || startPending) return false;
    const generation = ++startGeneration;
    startPending = true;
    status("starting_game");
    setTimeout(() => {
      if (generation !== startGeneration) return;
      startPending = false;
      if (!canStart()) return;
      console.log("host_game_start: players_confirmed=2 started=true");
      status("game_start_requested", { players_confirmed: 2 });
      room.startGame();
    }, 350);
    return true;
  };

  const playerStatus = (lifecycle, id, team, client, fields = {}) => {
    status(lifecycle, {
      player: {
        id,
        name: playerName(id),
        team,
        client,
        ...(lifecycle === "player_joined" ? { joined_timestamp: Date.now() } : {}),
        ...fields,
      },
    });
  };

  const announceQueuePositions = () => {
    if (!publicArena) return;
    const current = new Map();
    playerQueue.waitingIds.forEach((id, index) => {
      const position = index + 1;
      current.set(id, position);
      if (lastQueuePositions.get(id) !== position && room.getPlayer(id)) {
        announce(queueAnnouncements.position(position), id);
      }
    });
    lastQueuePositions.clear();
    for (const [id, position] of current) lastQueuePositions.set(id, position);
  };

  const syncPublicQueue = ({ announcePromotion = false, allowActiveBlue = true } = {}) => {
    if (!publicArena) return;
    playerQueue.reconcilePlayerQueue({ promote: !gameActive });
    const desiredActive = allowActiveBlue && !(botRecoveryPending && !botStartupReady)
      ? playerQueue.activeHumanId
      : null;
    if (reportedActiveHumanId !== desiredActive) {
      const previous = reportedActiveHumanId;
      reportedActiveHumanId = null;
      if (previous !== null) {
        if (room.getPlayer(previous)) room.setPlayerTeam(previous, 0);
        playerStatus("player_left", previous, 2, "opponent", { reason: "queue_rotation" });
      }
      if (desiredActive !== null && room.getPlayer(desiredActive)) {
        room.setPlayerTeam(desiredActive, 2);
        playerStatus("player_joined", desiredActive, 2, "opponent");
        reportedActiveHumanId = desiredActive;
        if (announcePromotion) {
          announce(queueAnnouncements.turnStarted(), desiredActive);
          announce(queueAnnouncements.promoted());
          if (playerQueue.waitingIds.length) {
            announce(queueAnnouncements.waiting(playerQueue.waitingIds.length));
          }
        }
      }
    }
    for (const id of playerQueue.connectedHumanIds) {
      if (id !== desiredActive && room.getPlayer(id)?.team !== 0) room.setPlayerTeam(id, 0);
    }
    announceQueuePositions();
  };

  const suspendHumansForBot = () => {
    if (!publicArena || reportedActiveHumanId === null) return;
    const active = reportedActiveHumanId;
    reportedActiveHumanId = null;
    if (room.getPlayer(active)) room.setPlayerTeam(active, 0);
    playerStatus("player_left", active, 2, "opponent", { reason: "bot_unavailable" });
    announceQueuePositions();
  };

  const markBotUnavailable = (reason, { playerAlreadyLeft = false } = {}) => {
    const firstDetection = !botUnavailable || controlledId !== null;
    botUnavailable = true;
    botRecoveryPending = true;
    botStartupReady = false;
    botControllerReady = false;
    playerQueue?.setBotReady(false);
    cancelScheduledStart();
    suspendHumansForBot();
    const staleId = controlledId;
    controlledId = null;
    if (staleId !== null && !playerAlreadyLeft && room.getPlayer(staleId)) {
      staleBotIds.add(staleId);
      room.kickPlayer(staleId, "Bot controller reconnect", false);
    }
    if (firstDetection) {
      console.log(`bot_disconnected reason=${reason}`);
      announce("Bot disconnected; match paused while recovery is attempted.");
    }
    resetting = true;
    send("reset", { reason: "bot_disconnect", tick_id: tickId, lifecycle_id: lifecycleId });
    if (gameActive) room.stopGame();
    else status("waiting_for_players");
  };

  const numericDisc = (disc) => {
    if (!disc) return null;
    const values = [disc.x, disc.y, disc.xspeed, disc.yspeed];
    if (!values.every(Number.isFinite)) return null;
    return { x: disc.x, y: disc.y, vx: disc.xspeed, vy: disc.yspeed };
  };
  const playerState = (id, team, client) => {
    if (id === null) return null;
    const player = room.getPlayer(id);
    const disc = numericDisc(room.getPlayerDiscProperties(id));
    if (!player || player.team !== team || !disc) return null;
    return { id: player.id, team, client, ...disc };
  };
  const compactState = () => {
    const controlled = playerState(controlledId, 1, "controlled");
    const opponent = playerState(activeOpponentId(), 2, "opponent");
    const ball = numericDisc(room.getDiscProperties(0));
    const scores = room.getScores();
    const validActive = gameActive && !resetting && controlled && opponent && ball && scores;
    return {
      tick_id: tickId,
      timestamp: Date.now(),
      game_active: Boolean(validActive),
      controlled,
      opponent,
      controlled_player: controlled,
      opponent_player: opponent,
      ball,
      match: {
        controlled_side: "red",
        controlled_score: scores?.red ?? 0,
        opponent_score: scores?.blue ?? 0,
        elapsed_time: scores?.time ?? 0,
        last_goal_event: lastGoal,
        last_touch_team: lastTouchTeam,
      },
    };
  };

  const beginObservationLifecycle = (reason) => {
    lifecycleId += 1;
    lifecycleStateSequence = 0;
    firstStateSentTimestamp = null;
    positionResetTimestamp = null;
    const now = Date.now();
    if (reason === "game_start") {
      gameStartCount += 1;
      gameStartTimestamp = now;
      console.log(
        `lifecycle_event event=game_start count=${gameStartCount} ` +
        `lifecycle_id=${lifecycleId} timestamp=${now}`,
      );
    }
    return lifecycleId;
  };

  const sendStateSnapshot = (reason, { forced = false } = {}) => {
    const snapshot = compactState();
    if (gameActive && !resetting && !snapshot.game_active) return false;
    if (forced && (!snapshot.game_active || snapshot.controlled?.team !== 1)) return false;
    tickId += 1;
    lifecycleStateSequence += 1;
    const now = Date.now();
    if (lifecycleStateSequence === 1) {
      firstStateSentCount += 1;
      firstStateSentTimestamp = now;
      console.log(
        `lifecycle_event event=first_state_sent count=${firstStateSentCount} ` +
        `lifecycle_id=${lifecycleId} timestamp=${now} reason=${reason}`,
      );
    }
    send("state", {
      ...snapshot,
      tick_id: tickId,
      timestamp: now,
      lifecycle_id: lifecycleId,
      state_sequence: lifecycleStateSequence,
      forced_snapshot: forced,
      snapshot_reason: reason,
      lifecycle_timestamps: {
        game_start: gameStartTimestamp,
        position_reset: positionResetTimestamp,
        first_state_sent: firstStateSentTimestamp,
      },
      lifecycle_counters: {
        game_start: gameStartCount,
        position_reset: positionResetCount,
        first_state_sent: firstStateSentCount,
      },
    });
    return true;
  };

  room.onRoomLink = (url) => {
    status("waiting_for_players", { room_url: url });
    if (publicArena) {
      console.log("PUBLIC_ROOM_READY");
      console.log(`room_name=${config.roomName}`);
      console.log(`room_url=${url}`);
      console.log("bot=Red");
      console.log("human=Blue");
      console.log(`player_queue=${playerQueue.enabled ? "enabled" : "disabled"}`);
    } else {
      console.log(`PRIVATE_ROOM_URL: ${url}`);
    }
  };

  room.onPlayerJoin = (player) => {
    if (player.name === config.controlledNickname) {
      if (controlledId !== null) {
        room.kickPlayer(player.id, "Duplicate bot client", false);
        return;
      }
      controlledId = player.id;
      room.setPlayerTeam(player.id, 1);
      if (publicArena) {
        room.setPlayerAdmin?.(player.id, false);
        if (botRecoveryPending) console.log(`bot_rejoined player_id=${player.id}`);
      }
      playerStatus("player_joined", player.id, 1, "controlled");
      scheduleStart();
      return;
    }

    if (publicArena) {
      if (player.name.length > (config.maxNicknameLength ?? 25)) {
        room.kickPlayer(player.id, `Nickname too long (max ${config.maxNicknameLength ?? 25})`, false);
        return;
      }
      const joined = playerQueue.addHuman(player.id);
      if (joined.duplicate) return;
      room.setPlayerAdmin?.(player.id, false);
      if (joined.active) {
        syncPublicQueue({ announcePromotion: true });
      } else {
        room.setPlayerTeam(player.id, 0);
        playerStatus("player_joined", player.id, 0, "spectator");
        syncPublicQueue();
      }
      scheduleStart();
      return;
    }

    if (opponentId === null &&
        (player.name === config.opponentNickname ||
         (config.humanOpponent === true && player.name !== config.controlledNickname))) {
      opponentId = player.id;
      room.setPlayerTeam(player.id, 2);
      playerStatus("player_joined", player.id, 2, "opponent");
      scheduleStart();
      return;
    }
    room.kickPlayer(player.id, "Private integration test room", false);
  };

  room.onPlayerLeave = (player) => {
    const botLeft = player.id === controlledId || staleBotIds.has(player.id);
    staleBotIds.delete(player.id);
    if (botLeft && publicArena) {
      markBotUnavailable("player_left", { playerAlreadyLeft: true });
      playerStatus("player_left", player.id, 1, "controlled");
      return;
    }
    if (player.id === controlledId) controlledId = null;

    if (publicArena && playerQueue.connectedHumanIds.has(player.id)) {
      const wasActive = playerQueue.activeHumanId === player.id;
      if (reportedActiveHumanId === player.id) reportedActiveHumanId = null;
      const removal = playerQueue.removeHuman(player.id);
      playerStatus("player_left", player.id, wasActive ? 2 : 0,
        wasActive ? "opponent" : "spectator");
      if (!removal.wasActive) {
        syncPublicQueue();
        return;
      }
      cancelScheduledStart();
      resetting = true;
      send("reset", { reason: "player_disconnect", tick_id: tickId, lifecycle_id: lifecycleId });
      if (gameActive) {
        room.stopGame();
      } else {
        syncPublicQueue({ announcePromotion: removal.promoted !== null });
        resetting = false;
        scheduleStart();
      }
      return;
    }

    const wasOpponent = player.id === opponentId;
    if (wasOpponent) opponentId = null;
    playerStatus("player_left", player.id, player.team,
      wasOpponent ? "opponent" : null);
    if (!wasOpponent) return;
    cancelScheduledStart();
    resetting = true;
    send("reset", { reason: "player_disconnect", tick_id: tickId, lifecycle_id: lifecycleId });
    if (gameActive) room.stopGame();
  };

  room.onGameStart = () => {
    cancelScheduledStart();
    if (gameActive) {
      sendStateSnapshot("duplicate_game_start", { forced: true });
      return;
    }
    if (publicArena && (resetting || !botStartupReady || activeOpponentId() === null)) {
      room.stopGame();
      return;
    }
    if (publicArena && !playerQueue.beginMatch()) {
      room.stopGame();
      return;
    }
    gameActive = true;
    resetting = false;
    beginObservationLifecycle("game_start");
    lastGoal = null;
    lastTouchTeam = null;
    send("reset", {
      reason: "game_start", tick_id: tickId, lifecycle_id: lifecycleId,
      reset_timestamp: Date.now(),
    });
    status("active_episode");
    sendStateSnapshot("game_start", { forced: true });
    if (publicArena && botRecoveryPending) {
      botRecoveryPending = false;
      console.log("match_restarted");
    }
  };

  room.onGameStop = () => {
    if (publicArena && !gameActive && !playerQueue.gameRunning &&
        !playerQueue.completionObserved && !playerQueue.interruptionPending) {
      return;
    }
    cancelScheduledStart();
    gameActive = false;
    resetting = true;
    status(playersReady() ? "match_ended" : "waiting_for_players");
    // Recorder/controller consumers must see the previous match end before any
    // promoted player can generate a new active state.
    send("reset", { reason: "game_stop", tick_id: tickId, lifecycle_id: lifecycleId });
    if (publicArena) {
      const transition = playerQueue.stopMatch();
      syncPublicQueue({
        announcePromotion: transition.promoted !== null && transition.promoted !== transition.outgoing,
      });
      resetting = false;
      scheduleStart();
    } else {
      resetting = false;
      scheduleStart();
    }
  };

  room.onTeamGoal = (team) => {
    resetting = true;
    beginObservationLifecycle("goal_reset");
    lastGoal = { team: team === 1 ? "red" : "blue", tick_id: tickId };
    status("goal_event", { last_goal_event: lastGoal });
    send("reset", {
      reason: "goal", tick_id: tickId, scoring_team: lastGoal.team,
      last_touch_team: lastTouchTeam,
      lifecycle_id: lifecycleId,
      reset_timestamp: Date.now(),
    });
  };
  room.onPlayerBallKick = (player) => {
    if (player.id === controlledId) lastTouchTeam = "red";
    else if (player.id === activeOpponentId()) lastTouchTeam = "blue";
  };
  room.onPositionsReset = () => {
    resetting = false;
    positionResetCount += 1;
    positionResetTimestamp = Date.now();
    console.log(
      `lifecycle_event event=position_reset count=${positionResetCount} ` +
      `lifecycle_id=${lifecycleId} timestamp=${positionResetTimestamp}`,
    );
    status("active_episode");
    sendStateSnapshot("position_reset", { forced: true });
  };
  room.onTeamVictory = () => {
    if (publicArena && (!gameActive || playerQueue.completionObserved)) return;
    resetting = true;
    status("match_ended");
    if (publicArena) {
      const completed = playerQueue.completeMatch();
      if (completed && gameActive) room.stopGame();
      return;
    }
    setTimeout(() => {
      if (gameActive) room.stopGame();
    }, 350);
  };

  room.onPlayerTeamChange = (changedPlayer) => {
    if (!publicArena) return;
    const requiredTeam = changedPlayer.id === controlledId ? 1 :
      changedPlayer.id === reportedActiveHumanId ? 2 : 0;
    if (changedPlayer.team !== requiredTeam) room.setPlayerTeam(changedPlayer.id, requiredTeam);
  };
  room.onPlayerAdminChange = (changedPlayer) => {
    if (publicArena && changedPlayer.id !== controlledId && changedPlayer.admin) {
      room.setPlayerAdmin?.(changedPlayer.id, false);
    }
  };

  window.__HAXBALL_BRIDGE_MESSAGE__ = (payload) => {
    if (payload?.type === "state_request") {
      const reason = payload.reason === "controller_watchdog"
        ? "controller_watchdog" : "readiness_refresh";
      sendStateSnapshot(reason, { forced: true });
      return;
    }
    if (!publicArena || payload?.type !== "readiness") return;
    const startupReady = payload.browser_connected === true &&
      payload.controlled_joined === true && payload.python_connected === true &&
      payload.python_protocol_ready === true;
    const controllerReady = startupReady && payload.controlled_surface_ready === true &&
      payload.controlled_input_ready === true;
    const startupLost = botStartupReady && !startupReady;
    const controllerLostDuringMatch = botControllerReady && !controllerReady && gameActive;
    if (startupLost) {
      markBotUnavailable("lobby_controller_disconnected");
      return;
    }
    if (controllerLostDuringMatch) {
      markBotUnavailable("controller_not_ready");
      return;
    }
    const controllerBecameReady = !botControllerReady && controllerReady;
    botStartupReady = startupReady;
    botControllerReady = controllerReady;
    playerQueue.setBotReady(startupReady);
    if (startupReady && controlledId !== null) {
      botUnavailable = false;
      syncPublicQueue();
      resetting = false;
      scheduleStart();
    }
    if (controllerBecameReady) {
      console.log("bot_controller_ready");
      announce("Bot ready on Red.");
    }
  };

  room.onGameTick = () => {
    hostTickId += 1;
    metricTicks += 1;
    if (hostTickId % config.controlDecimation === 0 && !resetting) {
      sendStateSnapshot("periodic_tick");
      metricStates += 1;
    }
    const now = performance.now();
    if (now - metricStartedAt >= 5000) {
      const seconds = (now - metricStartedAt) / 1000;
      status("metrics", {
        host_tick_hz: metricTicks / seconds,
        state_message_hz: metricStates / seconds,
      });
      metricStartedAt = now;
      metricTicks = 0;
      metricStates = 0;
    }
  };

  window.addEventListener("beforeunload", () => {
    send("shutdown", { reason: "host_unload" });
  });
  window.__HAXBALL_ROOM__ = room;
  window.__HAXBALL_PLAYER_QUEUE__ = playerQueue;
})();
