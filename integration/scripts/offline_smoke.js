import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { ACTIONS } from "../shared/actions.js";
import { InputController } from "../browser/input_controller.js";
import { message, parseMessage } from "../bridge/protocol.js";

class MockKeyboard {
  constructor() { this.events = []; }
  async down(key) { this.events.push(["down", key]); }
  async up(key) { this.events.push(["up", key]); }
}

const keyboard = new MockKeyboard();
const controller = new InputController(keyboard, { sleep: async () => {} });
for (const action of ACTIONS) await controller.applyAction(action.id);
await controller.releaseAll();

const opponentKeyboard = new MockKeyboard();
const opponentController = new InputController(opponentKeyboard, { clientId: "opponent" });
await controller.applyAction(4);
await opponentController.applyAction(3);
await controller.releaseAll();
if (![...opponentController.held].includes("ArrowLeft")) {
  throw new Error("dual controller key state was not independent");
}
await opponentController.releaseAll();

const parsed = parseMessage(JSON.stringify(message(
  "action", { tick_id: 1, client: "opponent", action: 7 })));
if (parsed.action !== 7 || parsed.client !== "opponent" ||
    ACTIONS.length !== 18 || controller.held.size !== 0) {
  throw new Error("JavaScript offline smoke failed");
}

const root = fileURLToPath(new URL("../../", import.meta.url));
const python = process.env.PYTHON ?? "python";
const result = spawnSync(
  python,
  ["-m", "integration.scripts.smoke_real_haxball", "--offline", "--json"],
  { cwd: root, encoding: "utf8" },
);
if (result.status !== 0) throw new Error(result.stderr || "Python offline smoke failed");
const pythonReport = JSON.parse(result.stdout);
if (
  pythonReport.action_count !== ACTIONS.length ||
  pythonReport.protocol_version !== 1 ||
  !pythonReport.random_actions.every((action) => Number.isInteger(action) && action >= 0 && action < 18)
) {
  throw new Error("Python/JavaScript protocol disagreement");
}
console.log(`protocol_version: ${pythonReport.protocol_version}`);
console.log(`canonical_actions: ${ACTIONS.length}`);
console.log(`input_events: ${keyboard.events.length}`);
console.log("dual_controller_independence: true");
console.log("python_javascript_agreement: true");
