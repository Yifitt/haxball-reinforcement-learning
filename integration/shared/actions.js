import { readFileSync } from "node:fs";

const actionUrl = new URL("./actions.json", import.meta.url);
export const ACTIONS = Object.freeze(JSON.parse(readFileSync(actionUrl, "utf8")));

if (ACTIONS.length !== 18 || ACTIONS.some((action, index) => action.id !== index)) {
  throw new Error("actions.json must contain action IDs 0 through 17 in order");
}

export function getAction(actionId) {
  if (!Number.isInteger(actionId) || actionId < 0 || actionId >= ACTIONS.length) {
    throw new RangeError(`invalid action: ${actionId}`);
  }
  return ACTIONS[actionId];
}
