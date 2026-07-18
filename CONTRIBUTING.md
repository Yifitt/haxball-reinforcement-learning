# Contributing

Create a Python 3.12 virtual environment, install the project extras, run
`scripts/bootstrap_haxballgym.sh`, and run `npm ci --prefix integration` as shown
in the README. Before submitting a change, run the Python, JavaScript, and Rust
tests plus `python scripts/check_documentation.py`.

Keep changes focused, add tests for behavior changes, preserve the observation and
action contracts unless a versioned migration is intentional, and update public
commands when a CLI changes. Do not commit credentials, `.env` files, checkpoints,
datasets, browser profiles, screenshots, logs, or generated experiment output.

Do not add human gameplay recording or identity persistence without an explicit
privacy and security review. Live-room tests must be opt-in and must never run in
CI.

This repository currently has no software license. Contributions should not
assume a license will be selected until the owner adds one explicitly.
