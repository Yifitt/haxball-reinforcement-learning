# HaxBall Reinforcement Learning

A research project for training a 1v1 HaxBall policy with PPO and self-play in a
pinned Rust simulator, evaluating portable PyTorch checkpoints, and deploying a
selected policy through HaxBall Headless Host, a local WebSocket bridge, and a
browser controller.

## Project status

The simulator, PPO pipeline, checkpoint contract, offline integration smokes,
private bot play, and public FIFO arena are implemented and tested. This is a
research prototype, not a hosted service. No public benchmark or Elo claim is
made. Model checkpoints are local artifacts and are intentionally absent from
Git.

## Architecture

The training path uses the pinned HaxBallGym Rust core for vectorized CPU
rollouts. `sim_training/train.py` supplies randomized kickoff resets, scripted and frozen
self-play opponents, PPO updates, fixed-seed evaluation, and portable checkpoint
metadata. The independent `haxball_env` implementation remains a physics
reference and is not the training backend.

Live play is a separate deployment boundary:

```text
PyTorch checkpoint -> Python controller -> local WebSocket bridge
                                           |              |
                                    Headless Host     browser input
                                           |              |
                                           +---- HaxBall --+
```

Public mode keeps the bot on Red, assigns one human to Blue, and leaves additional
players as spectators in a process-local FIFO queue. Private mode supports either
a human opponent or a stationary/scripted browser opponent.

## Repository structure

```text
haxball_env/          independent Python physics reference
policy_contract/      observation, action, and checkpoint compatibility
sim_training/         PPO, self-play, evaluation, tournament, and promotion logic
integration/          Headless Host, WebSocket bridge, browser control, and tests
scripts/              setup, documentation, smoke, and benchmark helpers
tests/                Python unit and offline integration tests
docs/                 focused architecture and operations notes
external/             revision pins; third-party source is restored locally
checkpoints/          local-only training and release artifacts
```

## Requirements

- Python 3.12 or newer
- Rust stable with Cargo
- Node.js 20 or newer with npm
- Git and a C/C++ build toolchain for the native Python extension
- Chromium or Chrome only for live play and the optional browser smoke
- A Headless Host token only for live room creation

Linux is the primary tested platform. CPU training is currently the supported
rollout path.

## Installation

```bash
git clone https://github.com/Yifitt/haxball-reinforcement-learning.git
cd haxball_reinforcement_learning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,integration,training]"
./scripts/bootstrap_haxballgym.sh
npm ci --prefix integration
```

Add the `render` extra if the independent Python renderer is needed:

```bash
python -m pip install -e ".[render]"
```

`scripts/bootstrap_haxballgym.sh` checks out the revision in
`external/HAXBALLGYM_REVISION` and installs `haxball_core` and `haxballgym` into
the active Python environment. Set `PYTHON=/path/to/python` only when selecting a
different active interpreter intentionally. The restored upstream checkout,
native build output, virtual environments, and Node modules are ignored by Git.

## Rust simulator build

The bootstrap command builds the PyO3 extension. Rust-only tests can then run
against the pinned checkout:

```bash
cargo test --locked --manifest-path external/HaxballGym/rust/haxball_core/Cargo.toml
```

The upstream `Cargo.lock` is retained by its source project and used with
`--locked`; this repository does not ignore lock files globally. Only the
generated `target/` tree is ignored.

## Running tests

All commands are offline after dependencies and the pinned simulator checkout are
installed:

```bash
python -m pytest -q
npm test --prefix integration
cargo test --locked --manifest-path external/HaxballGym/rust/haxball_core/Cargo.toml
python scripts/check_documentation.py
node integration/scripts/offline_smoke.js
node integration/scripts/offline_public_handshake_smoke.js
node integration/scripts/offline_queue_smoke.js
node integration/scripts/offline_stationary_kickoff_smoke.js
```

The optional `integration/scripts/browser_smoke.js` opens local browser clients
and should be run only when a deterministic browser environment is available.

## Training

### PPO smoke training

This is a small real update/evaluation run and writes only beneath the ignored
checkpoint directory:

```bash
python -m sim_training.train \
  --mode smoke \
  --seed 7 \
  --checkpoint-dir checkpoints/smoke
```

Inspect a configuration without creating a checkpoint:

```bash
python -m sim_training.train \
  --mode full \
  --seed 7 \
  --checkpoint-dir checkpoints/stage1 \
  --dry-run
```

### Standard training

```bash
python -m sim_training.train \
  --mode full \
  --seed 7 \
  --curriculum-stage 1 \
  --checkpoint-dir checkpoints/stage1
```

Later curriculum stages accept `--previous-policy-checkpoint` and
`--self-play-checkpoint`. Stage 4 uses named immutable anchors through
`--frozen-checkpoint LABEL=PATH`, optional approved seeds through
`--seed-self-play-checkpoint LABEL=PATH`, and a promotion report for a fresh run.
Use `--initialize-from` to start a new optimizer/counter history from weights;
use `--resume` only to continue the original experiment directory.

## Evaluation and self-play comparison

`sim_training/evaluate.py` evaluates a portable checkpoint on fixed held-out
simulator seeds:

```bash
python -m sim_training.evaluate \
  checkpoints/releases/selfplay_v1/model.pt \
  --episodes 128 \
  --n-envs 64 \
  --max-decisions 300
```

For a Stage 4 pool, also pass its local metadata:

```bash
python -m sim_training.evaluate \
  checkpoints/releases/selfplay_v1/model.pt \
  --self-play-pool-metadata checkpoints/stage4/self_play_pool_metadata.json
```

`sim_training/tournament.py` compares candidates under a checkpoint root;
`--apply-pool-cleanup` mutates pool membership and should be used only after
review. `sim_training/chase_diagnostic.py`,
`sim_training/benchmark_haxballgym.py`, and
`external/HaxballGym/rust/haxball_core/bench.py` provide focused diagnostics and
benchmarks.

## Live HaxBall integration

Copy a compatible `model.pt` and `policy_metadata.json` into an ignored release
directory first. Never commit either file. Export credentials in the shell; the
project does not automatically load `.env` files:

```bash
export HAXBALL_HEADLESS_TOKEN="replace_with_your_token"
```

Private stationary-opponent play:

```bash
python -m integration.scripts.smoke_real_haxball \
  --policy checkpoint \
  --checkpoint checkpoints/releases/selfplay_v1/model.pt \
  --duration 180
```

Private bot-versus-human play:

```bash
python -m integration.scripts.smoke_real_haxball \
  --human-opponent \
  --policy checkpoint \
  --checkpoint checkpoints/releases/selfplay_v1/model.pt \
  --duration 180
```

Public FIFO arena:

```bash
python -m integration.scripts.smoke_real_haxball \
  --public-room \
  --enable-player-queue \
  --max-players 12 \
  --matches-per-turn 1 \
  --queue-afk-timeout 0 \
  --score-limit 5 \
  --time-limit 0 \
  --room-name "RL Bot | 1v1" \
  --duration 3600
```

Public mode selects `checkpoints/releases/selfplay_v1/model.pt`, forces the
checkpoint policy and stationary opponent control, and enables the player queue
by default. `--disable-player-queue` is available for controlled maintenance.
AFK eviction is deliberately disabled, so `--queue-afk-timeout` must remain `0`.
The empty-lobby handshake completes before any Blue player joins. Disconnects and
official match completion are handled by the same lifecycle state machine.

The lower-level maintained entry points are
`integration/headless_host/launch_host.js`,
`integration/bridge/websocket_server.js`,
`integration/browser/launch_clients.js`, and
`integration/scripts/random_agent.py`. Prefer the Python orchestrator above so
all components share one validated startup configuration.

## Checkpoints

Every portable release consists of `model.pt` plus `policy_metadata.json`.
Training directories may additionally contain optimizer/RNG state, metrics,
periodic snapshots, and self-play pool metadata. The entire `checkpoints/` tree
is ignored: publish source and instructions, not weights or generated metrics.
Back up valuable releases outside the repository before pruning local runs.

The deployment loader supports the existing three-head MLP checkpoint contract.
Checkpoint metadata is validated for observation/action versions, dimensions,
architecture, and action repeat before inference.

## Reproducibility

Pass an explicit `--seed` to training and keep the generated
`run_configuration.json`, checkpoint metadata, revision pin, and pool metadata
together outside Git. Evaluation uses a fixed seed set. Native package versions
and the HaxBallGym revision are stored in checkpoint metadata. CPU scheduling and
different dependency builds can still cause numerical variation.

## Privacy

Public human matches are not recorded. Player names, authentication data, IP
addresses, chat, and gameplay frames are not persisted by this project. Browser
failure diagnostics remain in memory and redact room URLs and credential-like
query values. The removed human-scenario pipeline is not part of the public
codebase.

## Utility scripts

- `scripts/bootstrap_haxballgym.sh`: restore and install the pinned simulator.
- `scripts/check_documentation.py`: verify README executable-path coverage.
- `scripts/smoke_random_agent.py`: smoke the independent Python environment.
- `scripts/watch_scripted_match.py`: render a scripted reference match.
- `scripts/benchmark_env.py`: benchmark the independent Python environment.
- `integration/scripts/offline_smoke.js`: offline bridge protocol smoke.
- `integration/scripts/offline_public_handshake_smoke.js`: empty public lobby handshake.
- `integration/scripts/offline_queue_smoke.js`: FIFO queue smoke.
- `integration/scripts/offline_stationary_kickoff_smoke.js`: release inference smoke.
- `integration/scripts/smoke_real_haxball.py`: unified offline/live orchestrator.

## Known limitations

- Live play depends on HaxBall, browser automation, WebRTC, and a valid Headless
  Host token; these are intentionally excluded from CI.
- Training is CPU-only in the current Rust rollout adapter.
- Public queue AFK detection is disabled.
- Checkpoints are not distributed with the repository.
- The independent Python physics environment is a reference, not the PPO backend.

## Roadmap

- Expand deterministic simulator-versus-live fidelity tests.
- Improve checkpoint provenance and reproducible release packaging.
- Add measured evaluation reports without committing generated run trees.
- Harden reconnect and browser compatibility across supported platforms.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and contribution expectations
and [SECURITY.md](SECURITY.md) for private vulnerability reporting and token
handling.

