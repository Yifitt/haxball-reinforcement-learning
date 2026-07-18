# Live integration

The integration stack connects a Python policy controller to a local WebSocket
bridge, a HaxBall Headless Host page, and controlled browser input. The unified
entry point is:

```bash
python -m integration.scripts.smoke_real_haxball --help
```

Run offline checks first:

```bash
npm test
npm run smoke:offline
npm run smoke:public-handshake
npm run smoke:queue
```

Live room creation requires `HAXBALL_HEADLESS_TOKEN` in the process environment.
Use `.env.example` only as a variable inventory; the project does not load it.
Private modes support a stationary/scripted browser opponent or an ordinary human
opponent. Public mode keeps the policy on Red and rotates humans through Blue in
FIFO order; additional humans remain spectators. Queue state is process-local and
is cleared when the room stack exits.

The bridge binds to loopback by default. Browser failure inspection is bounded,
redacted, and kept in memory; the normal live path does not create screenshots,
text dumps, or gameplay files.

Protocol details are in `integration/bridge/protocol.md`. Prefer the unified
Python entry point over launching `headless_host/launch_host.js`,
`bridge/websocket_server.js`, and `browser/launch_clients.js` independently.
