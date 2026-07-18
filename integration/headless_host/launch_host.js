import { resolve } from "node:path";

import { chromium } from "playwright-core";
import WebSocket from "ws";

import { hostConfigFromEnv } from "./config.js";
import { parseMessage } from "../bridge/protocol.js";

const config = hostConfigFromEnv();
console.log(
  `headless_host_startup: mode=${config.startupMode} ` +
  `source_public_room=${config.sourcePublicRoom} ` +
  `source_human_opponent=${config.sourceHumanOpponent} ` +
  `source_opponent_policy=${config.sourceOpponentPolicy}`,
);
const headed = process.argv.includes("--headed");
const bridgeEndpoint = new URL(config.bridgeUrl);
const bridgePort = bridgeEndpoint.port || (bridgeEndpoint.protocol === "wss:" ? "443" : "80");
const executablePath = process.env.CHROME_PATH ?? "/usr/bin/google-chrome";
const connectBridge = async () => {
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    console.log(
      `local_connection_attempt: component=headless_host transport=websocket ` +
      `host=${bridgeEndpoint.hostname} port=${bridgePort} attempt=${attempt}`,
    );
    try {
      return await new Promise((resolveSocket, reject) => {
        const candidate = new WebSocket(config.bridgeUrl);
        candidate.once("open", () => resolveSocket(candidate));
        candidate.once("error", reject);
      });
    } catch (error) {
      lastError = error;
      await new Promise((resolveDelay) => setTimeout(resolveDelay, attempt * 250));
    }
  }
  throw lastError;
};

const bridgeSocket = await connectBridge();
const browser = await chromium.launch({
  executablePath,
  headless: !headed,
  args: ["--disable-features=WebRtcHideLocalIpsWithMdns"],
});

let closing = false;
let page = null;
let lastReadiness = null;
const close = async () => {
  if (closing) return;
  closing = true;
  await page?.evaluate(() => {
    const room = window.__HAXBALL_ROOM__;
    if (room?.getScores()) room.stopGame();
  }).catch(() => {});
  bridgeSocket.close();
  await browser.close();
};
process.once("SIGINT", close);
process.once("SIGTERM", close);

try {
  const context = await browser.newContext({ bypassCSP: true });
  await context.exposeBinding("__hostBridgeSend", async (_source, payload) => {
    if (bridgeSocket.readyState !== WebSocket.OPEN) {
      throw new Error("local bridge is disconnected");
    }
    bridgeSocket.send(JSON.stringify(payload));
  });
  bridgeSocket.on("message", (raw) => {
    try {
      const payload = parseMessage(raw);
      if (payload.type === "shutdown") close();
      if (payload.type === "readiness") {
        lastReadiness = payload;
      }
      if (payload.type === "readiness" || payload.type === "state_request") {
        page?.evaluate((message) => {
          window.__HAXBALL_BRIDGE_MESSAGE__?.(message);
        }, payload).catch(() => {});
      }
    } catch (error) {
      console.error(`host_bridge_protocol_error: ${error.message}`);
    }
  });
  bridgeSocket.on("close", () => close());
  page = await context.newPage();
  page.on("console", (event) => console.log(`host_page: ${event.text()}`));
  await page.goto("https://www.haxball.com/headless", {
    waitUntil: "domcontentloaded",
    timeout: 30_000,
  });
  const frame = page.mainFrame();
  await frame.waitForFunction(() => typeof window.HBInit === "function", null, {
    timeout: 30_000,
  });
  await frame.evaluate((value) => {
    window.__HAXBALL_INTEGRATION_CONFIG__ = value;
  }, config);
  await frame.addScriptTag({ path: resolve("headless_host/player_queue.js") });
  await frame.addScriptTag({ path: resolve("headless_host/host.js") });
  if (lastReadiness?.type === "readiness") {
    await frame.evaluate((message) => window.__HAXBALL_BRIDGE_MESSAGE__?.(message), lastReadiness);
  }
  console.log("headless_host_started: true");
  await new Promise((resolveDone) => browser.on("disconnected", resolveDone));
} catch (error) {
  console.error(`headless_host_error: ${error.message}`);
  await close();
  process.exitCode = 1;
}
