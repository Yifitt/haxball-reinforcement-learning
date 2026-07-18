#!/usr/bin/env python3
"""Offline README drift checks for repository-owned executable entry points."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# These top-level JavaScript programs do not use the Python ``__main__`` marker.
JAVASCRIPT_ENTRYPOINTS = {
    "integration/bridge/websocket_server.js",
    "integration/browser/launch_clients.js",
    "integration/headless_host/launch_host.js",
}

# Vendored upstream examples and one-off executable test modules are governed by
# external/HaxballGym's own documentation. Repository-owned wrappers and the
# upstream Rust build/test commands are documented in this project's README.
DOCUMENTATION_IGNORE = {
    "external/HaxballGym/headless-bot/bot.js",
    "external/HaxballGym/headless-bot/export_policy.py",
    "external/HaxballGym/headless-bot/nh_oracle.js",
    "external/HaxballGym/headless-bot/play.py",
    "external/HaxballGym/rl/play.py",
    "external/HaxballGym/rl/train.py",
    "external/HaxballGym/haxballgym/haxballgym/replays.py",
}

SCRIPT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:scripts|sim_training|integration|external/HaxballGym)/"
    r"[A-Za-z0-9_./-]+\.(?:py|js|sh))"
    r"(?![A-Za-z0-9_.-])"
)


@dataclass(frozen=True)
class DocumentationAudit:
    executable_scripts: tuple[str, ...]
    missing_from_readme: tuple[str, ...]
    stale_documented_paths: tuple[str, ...]
    ignored_scripts: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_from_readme and not self.stale_documented_paths


def _has_python_entrypoint(path: Path) -> bool:
    return 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")


def discover_executable_scripts(root: Path = ROOT) -> set[str]:
    """Return maintained executable scripts, excluding tests and package markers."""
    scripts: set[str] = set(JAVASCRIPT_ENTRYPOINTS)
    for directory in ("scripts", "integration/scripts", "sim_training"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.iterdir():
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if path.suffix == ".py" and _has_python_entrypoint(path):
                scripts.add(relative)
            elif path.suffix in {".js", ".sh"}:
                scripts.add(relative)

    # This is the supported upstream benchmark entry point referenced by the
    # repository's simulator operations, unlike the ignored upstream examples.
    upstream_benchmark = root / "external/HaxballGym/rust/haxball_core/bench.py"
    if upstream_benchmark.is_file():
        scripts.add(upstream_benchmark.relative_to(root).as_posix())
    scripts.update(path for path in DOCUMENTATION_IGNORE if (root / path).is_file())
    return scripts


def documented_script_paths(readme_text: str) -> set[str]:
    return set(SCRIPT_PATH_RE.findall(readme_text))


def audit_repository(root: Path = ROOT, readme: Path | None = None) -> DocumentationAudit:
    readme_path = readme or root / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    executable = discover_executable_scripts(root)
    ignored = executable & DOCUMENTATION_IGNORE
    required = executable - DOCUMENTATION_IGNORE
    missing = sorted(path for path in required if path not in text)
    documented = documented_script_paths(text)
    stale = sorted(path for path in documented if not (root / path).is_file())
    return DocumentationAudit(
        executable_scripts=tuple(sorted(required)),
        missing_from_readme=tuple(missing),
        stale_documented_paths=tuple(stale),
        ignored_scripts=tuple(sorted(ignored)),
    )


def checklist() -> str:
    return "\n".join((
        "Documentation update checklist:",
        "- Update project status and architecture when behavior or maturity changes.",
        "- Update the script reference and examples for every CLI change.",
        "- Document every new executable before considering the change complete.",
        "- Remove or correct renamed and deleted paths/features.",
        "- Update data, checkpoint, configuration, safety, and test contracts when changed.",
        "- Keep secrets, private URLs, network addresses, and recorded identities out of docs.",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check README executable inventory and documented script paths offline."
    )
    parser.add_argument(
        "--checklist", action="store_true", help="also print the README maintenance checklist"
    )
    args = parser.parse_args()
    audit = audit_repository()
    if audit.missing_from_readme:
        print("Executable scripts missing from README.md:")
        for path in audit.missing_from_readme:
            print(f"- {path}")
    if audit.stale_documented_paths:
        print("Documented script paths that no longer exist:")
        for path in audit.stale_documented_paths:
            print(f"- {path}")
    if audit.ok:
        print(
            f"README script inventory is current: "
            f"{len(audit.executable_scripts)} executable paths documented."
        )
    if args.checklist:
        print(checklist())
    return 0 if audit.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
