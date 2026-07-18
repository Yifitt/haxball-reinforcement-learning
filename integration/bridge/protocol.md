# Local bridge protocol v1

Every WebSocket message is compact JSON containing `protocol_version: 1` and a
`type`. Connections first send `hello` with role `host`, `browser`, or `agent`.
Supported types are `hello`, `room_status`, `client_status`, `readiness`, `state`, `action`,
`action_applied`, `reset`, `error`, and `shutdown`.

The host emits monotonically increasing `state.tick_id` values. The agent replies
with `{protocol_version: 1, type: "action", tick_id, client, action}` where
`client` is `controlled` or `opponent` and `action` is 0–17. Stale and duplicate
tracking, readiness, application latency, and rejection counts are independent per
client. Readiness is tracked independently for the host, room, players, both
browser surfaces/inputs, Python protocol handshake, and active game state.
Pre-ready actions receive a structured nonfatal `not_ready` response. It binds to
`127.0.0.1` by default.

Readiness has explicit `stationary_opponent`, `dual_control`, `human_opponent`,
and `public_human_queue` modes. The orchestrator supplies the resolved mode and
canonical public/human/opponent-control flags to the bridge before it listens;
an explicit mode is validated, locked, and never re-inferred from later false
telemetry defaults. Public lobby readiness deliberately excludes
`private_room`, `opponent_player`, and `active_game_state`, allowing the Python
handshake to finish while the FIFO room waits empty. Once the queue promotes a
human to Blue and the game becomes active, controlled canvas/input, Blue player,
and active state join the action barrier. A valid agent `hello` marks the Python
controller connected immediately. Its acknowledgment is retained until the
public infrastructure requirements are ready, reconciled after every relevant
update, and sent once regardless of Python-first or host-first ordering.
