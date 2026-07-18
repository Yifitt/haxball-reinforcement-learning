# PPO and self-play operations

Current training is implemented by `sim_training.train`. It preserves the
16-value observation contract, 18 canonical actions, Red/Blue mirroring,
randomized one-goal episodes, kick-pulse execution, fixed evaluation seeds, and
browser-compatible portable checkpoints.

Inspect the complete CLI before starting a run:

```bash
python -m sim_training.train --help
```

Use `--dry-run` to validate configuration without creating an output directory.
Use `--initialize-from` for a new optimizer and counters, or `--resume` for an
honest continuation with its original `run_configuration.json`. Never point a
fresh run at an existing checkpoint directory.

Stage 4 combines permanent anchors, approved self-play seeds, bounded pool
membership, random learner sides, and periodic tournament-approved snapshots.
General checkpoint comparison remains available through
`sim_training.tournament`; it does not replace a release unless an owner performs
that separate local operation.

Evaluation example:

```bash
python -m sim_training.evaluate \
  checkpoints/releases/selfplay_v1/model.pt \
  --episodes 128 --n-envs 64 --max-decisions 300
```

Reports contain wins, draws, losses, goals, own-goal attribution, duration,
timeouts, opponent/kickoff/side breakdowns, entropy, action diversity, and kick
metrics. No benchmark result should be published without retaining the exact
checkpoint metadata, simulator revision, pool metadata, command, and seeds.
