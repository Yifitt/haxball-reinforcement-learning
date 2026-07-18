from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ACTION_FILE = Path(__file__).parents[1] / "shared" / "actions.json"
ACTIONS: tuple[dict[str, Any], ...] = tuple(json.loads(ACTION_FILE.read_text()))

if len(ACTIONS) != 18 or any(action["id"] != index for index, action in enumerate(ACTIONS)):
    raise RuntimeError("actions.json must contain action IDs 0 through 17 in order")


def get_action(action_id: int) -> dict[str, Any]:
    if isinstance(action_id, bool) or not isinstance(action_id, int) or not 0 <= action_id < 18:
        raise ValueError(f"invalid action: {action_id!r}")
    return ACTIONS[action_id]
