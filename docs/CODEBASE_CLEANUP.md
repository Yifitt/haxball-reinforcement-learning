# Public-release cleanup history

The first-upload cleanup removed local caches, browser diagnostics, private
defaults, absolute developer paths, generated player data, and the experimental
human-match recording and human-scenario reset pipeline. Ordinary bot-versus-human
play and the public FIFO queue remain supported without persistence.

The standard PPO/self-play implementation, general evaluation and tournament
tools, the complete `baseline_v2_control_20m` local experiment, and the selected
control iteration 2250 checkpoint were deliberately retained. A hash-matched
neutral local release copy lives under
`checkpoints/releases/baseline_v2_control_iter2250/`; the checkpoint tree remains
ignored by Git.

The independent `haxball_env` physics reference and pinned HaxBallGym simulator
workflow also remain. HaxballGym is vendored as ordinary source under
`external/HaxballGym`; its upstream revision is recorded separately. Virtual
environments, native build output, Node modules, datasets, and checkpoints remain
local-only.
