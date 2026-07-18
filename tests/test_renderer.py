import sys

import pytest

from haxball_env import EnvConfig, HaxBallEnv


def test_rendering_unused_does_not_import_pygame() -> None:
    assert "pygame" not in sys.modules
    env = HaxBallEnv(render_mode=None)
    env.reset(seed=0)
    env.step(0)
    env.close()
    assert "pygame" not in sys.modules


def test_human_renderer_runs_headlessly_and_handles_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    env = HaxBallEnv(EnvConfig(render_fps=1_000), render_mode="human")
    try:
        env.reset(seed=0)
        for _ in range(3):
            env.step(0)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        env.render()
        assert env.window_closed
    finally:
        env.close()
        env.close()
