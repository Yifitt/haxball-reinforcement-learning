import { chromium } from "playwright-core";

import { gameFrame } from "../browser/controlled_client.js";
import { findGameCanvas } from "../browser/game_surface.js";
import { InputController } from "../browser/input_controller.js";
import { prepareKeyboardInput } from "../browser/input_focus.js";
import { summarizeContext } from "../browser/join_diagnostics.js";

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH ?? "/usr/bin/google-chrome",
  headless: true,
});

try {
  const context = await browser.newContext();
  let ipcProbe = false;
  await context.exposeBinding("__integrationProbe", async (_source, payload) => {
    ipcProbe = payload?.probe === true;
    return "ack";
  });
  const page = await context.newPage();
  await page.goto("https://www.haxball.com/play", {
    waitUntil: "domcontentloaded",
    timeout: 30_000,
  });
  let frame = await gameFrame(page);
  const joinSummary = await summarizeContext(context, "browser-smoke");
  const nicknameDetected = joinSummary.pages
    .flatMap((candidate) => candidate.frames)
    .some((candidate) => candidate.nickname_input);
  if (!nicknameDetected) throw new Error("live HaxBall nickname form was not detected");
  const nicknameInput = frame.locator('.choose-nickname-view input[data-hook="input"]');
  await nicknameInput.fill("RL-Browser-Smoke");
  await frame.locator('.choose-nickname-view button[data-hook="ok"]').click();
  await nicknameInput.waitFor({ state: "hidden", timeout: 10_000 });
  await page.goto("https://www.haxball.com/play", {
    waitUntil: "domcontentloaded",
    timeout: 30_000,
  });
  frame = await gameFrame(page);
  const retryNickname = frame.locator('.choose-nickname-view input[data-hook="input"]');
  await retryNickname.fill("RL-Browser-Smoke");
  await frame.locator('.choose-nickname-view button[data-hook="ok"]').click();
  await retryNickname.waitFor({ state: "hidden", timeout: 10_000 });
  await frame.getByText("Room list", { exact: true }).waitFor({ state: "visible", timeout: 10_000 });
  const ipcReply = await frame.evaluate(() => window.__integrationProbe({ probe: true }));
  if (ipcReply !== "ack" || !ipcProbe) throw new Error("frame-to-Node IPC probe failed");
  await frame.evaluate(() => {
    window.__integrationKeyEvents = [];
    document.body.tabIndex = -1;
    document.body.focus();
    document.addEventListener("keydown", (event) =>
      window.__integrationKeyEvents.push(["down", event.code]));
    document.addEventListener("keyup", (event) =>
      window.__integrationKeyEvents.push(["up", event.code]));
    const utility = document.createElement("canvas");
    utility.width = 32;
    utility.height = 64;
    utility.style.cssText = "position:fixed;left:0;top:0;width:32px;height:64px";
    const game = document.createElement("canvas");
    game.width = 640;
    game.height = 360;
    game.style.cssText = "position:fixed;left:40px;top:40px;width:640px;height:360px;z-index:9999";
    const overlay = document.createElement("div");
    overlay.className = "top-section";
    overlay.dataset.hook = "top-section";
    overlay.style.cssText = "position:fixed;left:40px;top:40px;width:640px;height:360px;z-index:10000;pointer-events:auto";
    document.body.append(utility, game, overlay);
  });
  const gameCanvas = await findGameCanvas(frame);
  const gameCanvasBox = await gameCanvas.boundingBox();
  if (!gameCanvasBox || gameCanvasBox.width < 300 || gameCanvasBox.height < 150) {
    throw new Error("browser smoke selected an invalid game canvas");
  }
  const normalClickIntercepted = await gameCanvas.click({ timeout: 300 }).then(
    () => false,
    () => true,
  );
  if (!normalClickIntercepted) throw new Error("overlay did not reproduce pointer interception");
  const focusResult = await prepareKeyboardInput(
    { page, frame, canvas: gameCanvas },
    page.keyboard,
  );
  if (focusResult.method !== "dom_focus" || !focusResult.keyboardVerified) {
    throw new Error("programmatic focus did not verify real keyboard delivery");
  }
  if (await gameCanvas.getAttribute("tabindex") !== null) {
    throw new Error("temporary canvas tabindex was not restored");
  }
  const duplicateFocus = await prepareKeyboardInput(
    { page, frame, canvas: gameCanvas },
    page.keyboard,
  );
  if (!duplicateFocus.alreadyPrepared) throw new Error("focus initialization was not idempotent");
  await gameCanvas.evaluate((element) => {
    element.blur();
    element.setAttribute("tabindex", "7");
    element.removeAttribute("data-haxball-integration-keyboard-ready");
  });
  await prepareKeyboardInput({ page, frame, canvas: gameCanvas }, page.keyboard);
  if (await gameCanvas.getAttribute("tabindex") !== "7") {
    throw new Error("existing canvas tabindex was not preserved");
  }
  const controller = new InputController(page.keyboard, { kickMilliseconds: 5 });
  await controller.applyAction(4);
  await controller.applyAction(15);
  await controller.applyAction(0);
  await controller.releaseAll();
  const events = await frame.evaluate(() => window.__integrationKeyEvents);
  const required = [
    ["down", "ArrowRight"],
    ["down", "ArrowUp"],
    ["down", "KeyX"],
    ["up", "KeyX"],
    ["up", "ArrowUp"],
    ["up", "ArrowRight"],
  ];
  for (const expected of required) {
    if (!events.some((event) => event[0] === expected[0] && event[1] === expected[1])) {
      throw new Error(`missing browser key event: ${expected.join(" ")}`);
    }
  }
  const hostPage = await context.newPage();
  await hostPage.goto("https://www.haxball.com/headless", {
    waitUntil: "domcontentloaded",
    timeout: 30_000,
  });
  const hostFrame = hostPage.mainFrame();
  await hostFrame.waitForFunction(() => typeof window.HBInit === "function", null, {
    timeout: 30_000,
  });
  console.log(`browser_page_title: ${await page.title()}`);
  console.log(
    `join_page_enumeration: pages=${joinSummary.page_count} ` +
    `frames=${joinSummary.pages.reduce((count, candidate) => count + candidate.frames.length, 0)} ` +
    `nickname_input=${nicknameDetected}`,
  );
  console.log("nickname_submission_and_renavigation: true");
  console.log(`browser_key_events: ${events.length}`);
  console.log(`selected_game_canvas: ${gameCanvasBox.width}x${gameCanvasBox.height}`);
  console.log(`overlay_click_intercepted: ${normalClickIntercepted}`);
  console.log(`input_focus_method: ${focusResult.method}`);
  console.log(`keyboard_focus_verified: ${focusResult.keyboardVerified}`);
  console.log("local_ipc_from_haxball_page: true");
  console.log("official_headless_api_loaded: true");
  console.log("browser_shutdown_clean: true");
} finally {
  await browser.close();
}
