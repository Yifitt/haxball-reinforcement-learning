# Persistent project instructions

## Documentation is part of the definition of done

The root `README.md` is the canonical public entry point. Every user-visible CLI,
architecture, checkpoint, deployment, privacy, setup, or workflow change must be
reflected there and in focused documentation when applicable.

- Keep commands portable and based on the active Python interpreter.
- Keep credentials, private URLs, local identities, and machine paths out of
  source, tests, documentation, and generated examples.
- Update executable-path inventory when scripts are added, removed, or renamed.
- Prefer dry runs, temporary output directories, and offline smokes for
  verification. Never use an authenticated room as a routine test.
- Do not overwrite release or experiment checkpoints during validation.

Run `python scripts/check_documentation.py --checklist` and the complete test
suites before a release-oriented change is considered complete.

## Public-room queue invariants

- The deployment bot is always Red; at most one human is Blue, and all other
  connected humans are FIFO spectators identified only by live room ID.
- Rotation occurs only through the official match-end/game-stop lifecycle.
- Queue mutation uses the process-local reconciliation and promotion methods.
- Bot/controller recovery preserves order and invalidates stale start callbacks.
- AFK eviction remains disabled until a reliable activity signal exists.
- Every reset clears bridge tick/action caches, Python episode tracking, browser
  action generations, and held inputs before a complete active snapshot is used.
- The controlled-action watchdog requests at most one recovery until progress
  resumes and never creates a second Python action loop.
- The empty public lobby completes its infrastructure handshake before player and
  game-surface readiness become required.
