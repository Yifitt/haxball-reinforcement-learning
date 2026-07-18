from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".venv", "node_modules", "checkpoints", ".git", "HaxballGym"}


def repository_text() -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            documents.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return documents


def test_source_has_no_developer_home_or_private_defaults() -> None:
    forbidden = ("/home/" + "yifit/", "Yiğit" + " RL Bot", "yigit" + "2914")
    matches = [
        path.relative_to(ROOT)
        for path, content in repository_text()
        if any(value in content for value in forbidden)
    ]
    assert matches == []


def test_source_has_no_obvious_live_headless_credential() -> None:
    marker = "thr1" + "."
    matches = [
        path.relative_to(ROOT)
        for path, content in repository_text()
        if marker in content
    ]
    assert matches == []


def test_environment_example_is_placeholder_only() -> None:
    assert (ROOT / ".env.example").read_text(encoding="utf-8") == (
        "HAXBALL_HEADLESS_TOKEN=replace_with_your_token\n"
    )


def test_generated_and_sensitive_paths_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".env", "!.env.example", "*.token", "checkpoints/", "node_modules/",
        "data/raw/", "data/processed/", "browser-profiles/", "screenshots/",
        "target/", "*.log",
    ):
        assert pattern in ignore


def test_browser_failure_inspection_has_no_disk_writer() -> None:
    source = (ROOT / "integration/browser/join_diagnostics.js").read_text(encoding="utf-8")
    assert "node:fs" not in source
    assert ".screenshot(" not in source
    assert "writeFile(" not in source
