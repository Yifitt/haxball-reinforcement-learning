#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_dir="$project_root/external"
haxballgym_dir="$external_dir/HaxballGym"
revision_file="$external_dir/HAXBALLGYM_REVISION"

if [[ -n "${1:-}" ]]; then
  echo "usage: $0" >&2
  exit 2
fi

if [[ ! -d "$haxballgym_dir" ]]; then
  echo "vendored HaxballGym source is missing: $haxballgym_dir" >&2
  exit 1
fi
if [[ ! -f "$haxballgym_dir/LICENSE" ||
      ! -f "$haxballgym_dir/rust/haxball_core/pyproject.toml" ||
      ! -f "$haxballgym_dir/haxballgym/pyproject.toml" ]]; then
  echo "vendored HaxballGym source is incomplete" >&2
  exit 1
fi
if [[ ! -f "$revision_file" ]]; then
  echo "vendored HaxballGym revision record is missing: $revision_file" >&2
  exit 1
fi
haxballgym_revision="$(tr -d '[:space:]' < "$revision_file")"
if [[ ! "$haxballgym_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "vendored HaxballGym revision record is invalid" >&2
  exit 1
fi

python_executable="${PYTHON:-python}"
if ! command -v "$python_executable" >/dev/null 2>&1; then
  echo "active Python interpreter not found: $python_executable" >&2
  exit 1
fi
if ! "$python_executable" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
  echo "Python 3.12 or newer is required" >&2
  exit 1
fi

if ! command -v rustc >/dev/null 2>&1; then
  cargo_bin="${CARGO_HOME:-${HOME}/.cargo}/bin"
  export PATH="$cargo_bin:$PATH"
fi
if ! command -v rustc >/dev/null 2>&1; then
  echo "Rust stable is required; install it with rustup using the minimal profile" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$python_executable" \
    --editable "$haxballgym_dir/rust/haxball_core" \
    --editable "$haxballgym_dir/haxballgym"
else
  "$python_executable" -m pip install \
    --editable "$haxballgym_dir/rust/haxball_core" \
    --editable "$haxballgym_dir/haxballgym"
fi
echo "HaxballGym ready at $haxballgym_revision"
