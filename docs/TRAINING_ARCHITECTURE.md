# Training architecture

- **Simulation:** pinned HaxBallGym Rust engine, vectorized over CPU environments
  with deterministic randomized kickoff states.
- **Learning:** PPO with staged scripted opponents, immutable self-play
  generations, permanent anchors, fixed-seed evaluation, and gated promotion.
- **Contract:** 16 mirrored observations and 18 canonical actions; movement is
  repeated over physics ticks while kick is pulsed once.
- **Artifact:** portable PyTorch MLP weights plus JSON metadata shared by
  simulator evaluation and live inference.
- **Deployment:** Headless Host, local WebSocket bridge, and a controlled browser
  for the Red agent. Humans join through ordinary HaxBall clients.

The browser/WebRTC path is a deployment boundary, not a rollout backend.
`haxball_env` remains a tested independent physics reference and is not imported
by current training.

Fresh experiments use `--initialize-from`; interruption recovery uses `--resume`
and restores optimizer, counters, NumPy/Torch/reset/opponent/side RNG state,
curriculum configuration, and self-play pool metadata. Configuration drift is
rejected before continuation.
