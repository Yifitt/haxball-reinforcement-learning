#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_dir="$project_root/external"
haxballgym_dir="$external_dir/HaxballGym"
haxballgym_revision="$(tr -d '[:space:]' < "$external_dir/HAXBALLGYM_REVISION")"
fetch_updates=false

if [[ "${1:-}" == "--fetch" ]]; then
  fetch_updates=true
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--fetch]" >&2
  exit 2
fi

mkdir -p "$external_dir"

ensure_checkout() {
  local url="$1"
  local target="$2"
  local revision="$3"
  if [[ ! -d "$target/.git" ]]; then
    git clone "$url" "$target"
  elif [[ "$fetch_updates" == true ]]; then
    git -C "$target" fetch --all --tags --prune
  fi
  if ! git -C "$target" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    echo "revision $revision is unavailable in $target; rerun with --fetch" >&2
    exit 1
  fi
  git -C "$target" checkout --detach "$revision"
}

ensure_checkout \
  https://github.com/HaxballGym/HaxballGym.git \
  "$haxballgym_dir" \
  "$haxballgym_revision"

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
