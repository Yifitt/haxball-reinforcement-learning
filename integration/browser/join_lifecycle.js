import { findGameSurfaceAcrossContext } from "./game_surface.js";
import {
  attachPageDiagnostics,
  captureJoinDiagnostics,
  sanitizeUrl,
  summarizeContext,
} from "./join_diagnostics.js";
import { validateRoomUrl } from "./room_url.js";

export const JOIN_STAGES = Object.freeze({
  NAVIGATING: "NAVIGATING",
  WAITING_FOR_JOIN_UI: "WAITING_FOR_JOIN_UI",
  ENTERING_NICKNAME: "ENTERING_NICKNAME",
  SUBMITTING_JOIN: "SUBMITTING_JOIN",
  WAITING_FOR_ROOM: "WAITING_FOR_ROOM",
  WAITING_FOR_GAME_SURFACE: "WAITING_FOR_GAME_SURFACE",
  WAITING_FOR_HOST_CONFIRMATION: "WAITING_FOR_HOST_CONFIRMATION",
  READY: "READY",
  FAILED: "FAILED",
});

export async function executeJoinLifecycle({
  adapter,
  nickname,
  roomUrl,
  requireGameSurface = true,
  onStage = () => {},
  releaseInputs = async () => {},
}) {
  let stage = JOIN_STAGES.NAVIGATING;
  const setStage = async (next) => {
    stage = next;
    await onStage(next);
  };
  const submitJoin = async (attempt) => {
    await setStage(JOIN_STAGES.NAVIGATING);
    await adapter.navigate(roomUrl);
    await setStage(JOIN_STAGES.WAITING_FOR_JOIN_UI);
    const nicknameForm = await adapter.findNicknameForm();
    if (nicknameForm) {
      await setStage(JOIN_STAGES.ENTERING_NICKNAME);
      await adapter.enterNickname(nicknameForm, nickname);
      await setStage(JOIN_STAGES.SUBMITTING_JOIN);
      await adapter.submitNickname(nicknameForm);
      await setStage(JOIN_STAGES.WAITING_FOR_ROOM);
      if (await adapter.shouldReopenRoom()) {
        await adapter.onRetry?.(attempt, "returned_to_room_list_after_profile_setup");
        await adapter.navigate(roomUrl);
        await setStage(JOIN_STAGES.WAITING_FOR_JOIN_UI);
        const retryForm = await adapter.findNicknameForm();
        if (retryForm) {
          await setStage(JOIN_STAGES.ENTERING_NICKNAME);
          await adapter.enterNickname(retryForm, nickname);
          await setStage(JOIN_STAGES.SUBMITTING_JOIN);
          await adapter.submitNickname(retryForm);
        }
      }
    }
    await setStage(JOIN_STAGES.WAITING_FOR_ROOM);
    await setStage(JOIN_STAGES.WAITING_FOR_HOST_CONFIRMATION);
    return adapter.waitForHostJoin(nickname);
  };
  try {
    let player;
    let firstError;
    for (let attempt = 1; attempt <= 2; attempt += 1) {
      try {
        player = await submitJoin(attempt);
        break;
      } catch (error) {
        if (attempt === 2) throw error;
        firstError = error;
        await adapter.onRetry?.(2, `host_entry_not_confirmed: ${error.message}`);
        await adapter.prepareRetry?.();
      }
    }
    if (!player) throw firstError ?? new Error("host did not confirm room entry");
    if (!requireGameSurface) {
      await setStage(JOIN_STAGES.READY);
      return { player, stage };
    }
    await setStage(JOIN_STAGES.WAITING_FOR_GAME_SURFACE);
    const surface = await adapter.findGameSurface();
    await adapter.onSurfaceReady?.(surface);
    await setStage(JOIN_STAGES.READY);
    return { ...surface, player, stage };
  } catch (error) {
    await releaseInputs().catch(() => {});
    await onStage(JOIN_STAGES.FAILED);
    await adapter.onFailure?.(stage, error);
    throw error;
  }
}

const CONNECTION_ERROR = /failed to connect|connection (?:failed|closed|timed out)|room not found|room (?:has been )?closed|room full|webrtc failure|disconnected|you were kicked|banned|invalid room link|browser incompatib/i;

async function visibleConnectionError(context) {
  for (const page of context.pages()) {
    for (const frame of page.frames()) {
      const text = await frame.locator("body").innerText().catch(() => "");
      const match = text.match(CONNECTION_ERROR);
      if (match) return match[0];
    }
  }
  return null;
}

async function waitForConnectionError(context, signal) {
  while (!signal.aborted) {
    const error = await visibleConnectionError(context);
    if (error) throw new Error(`HaxBall connection error: ${error}`);
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  throw new Error("connection-error watcher cancelled");
}

async function findNicknameForm(context) {
  for (const page of context.pages()) {
    for (const frame of page.frames()) {
      const input = frame.locator('.choose-nickname-view input[data-hook="input"]');
      if (await input.isVisible().catch(() => false)) {
        const submit = frame.locator('.choose-nickname-view button[data-hook="ok"]');
        return { page, frame, input, submit };
      }
    }
  }
  return null;
}

async function waitForNicknameFormOrRoomProgress(context, timeout = 5_000) {
  const deadline = Date.now() + timeout;
  do {
    const form = await findNicknameForm(context);
    if (form) return form;
    for (const page of context.pages()) {
      for (const frame of page.frames()) {
        if (await frame.locator("canvas").count().catch(() => 0)) return null;
        const text = await frame.locator("body").innerText().catch(() => "");
        if (/room list|connecting|joining|connection failed|room closed|room full/i.test(text)) {
          return null;
        }
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  } while (Date.now() < deadline);
  return null;
}

export async function joinPrivateRoom({
  context,
  roomUrl,
  nickname,
  clientId,
  waitForHostJoin,
  onSurfaceReady = async () => {},
  timeout = 45_000,
  releaseInputs = async () => {},
  requireGameSurface = true,
}) {
  const roomUrlValidation = validateRoomUrl(roomUrl);
  console.log(
    `room_url_validation: client=${clientId} host=${roomUrlValidation.host} ` +
    `path=${roomUrlValidation.path} room_code_present=${roomUrlValidation.roomCodePresent} ` +
    `room_code_length=${roomUrlValidation.roomCodeLength} ` +
    `room_code_hash_match=${roomUrlValidation.roomCodeHashMatch}`,
  );
  const events = [];
  const knownPages = new Set();
  const attach = (page) => {
    if (knownPages.has(page)) return;
    knownPages.add(page);
    attachPageDiagnostics(page, events, clientId);
  };
  context.pages().forEach(attach);
  context.on("page", attach);
  let navigationPage = context.pages()[0] ?? await context.newPage();
  attach(navigationPage);
  let currentStage = JOIN_STAGES.NAVIGATING;
  const stage = async (next) => {
    currentStage = next;
    const summary = await summarizeContext(context, clientId);
    const frameCount = summary.pages.reduce((count, page) => count + page.frames.length, 0);
    console.log(
      `client_join_stage: client=${clientId} stage=${next} pages=${summary.page_count} ` +
      `frames=${frameCount} url=${sanitizeUrl(navigationPage.url())}`,
    );
  };

  const adapter = {
    navigate: async (target) => {
      const navigationValidation = validateRoomUrl(target, roomUrl);
      if (!navigationValidation.roomCodeHashMatch) {
        throw new Error("navigation room code does not match the host-provided room code");
      }
      if (navigationPage.isClosed()) navigationPage = await context.newPage();
      await navigationPage.goto(target, { waitUntil: "domcontentloaded", timeout });
      console.log(
        `client_navigation: client=${clientId} initial_url=${sanitizeUrl(target)} ` +
        `final_url=${sanitizeUrl(navigationPage.url())} pages=${context.pages().length}`,
      );
    },
    findNicknameForm: () => waitForNicknameFormOrRoomProgress(context),
    enterNickname: async (form, value) => {
      navigationPage = form.page;
      await form.input.fill(value);
      await form.input.blur();
      form.expectedNickname = value;
      form.inputValueConfirmed = await form.input.inputValue() === value;
      if (!form.inputValueConfirmed) {
        throw new Error("nickname input value was not retained before submission");
      }
    },
    submitNickname: async (form) => {
      const inputValueConfirmed = form.inputValueConfirmed === true &&
        await form.input.inputValue().catch(() => "") === form.expectedNickname;
      if (!inputValueConfirmed) {
        throw new Error("nickname input value did not match immediately before submission");
      }
      let submitMethod;
      if (await form.submit.isVisible().catch(() => false)) {
        submitMethod = "button";
        await form.submit.click();
      } else {
        submitMethod = "enter";
        await form.input.press("Enter");
      }
      await form.input.waitFor({ state: "hidden", timeout: Math.min(timeout, 10_000) });
      console.log(
        `nickname_submission: client=${clientId} frame_url=${sanitizeUrl(form.frame.url())} ` +
        `input_value_confirmed=${inputValueConfirmed} submit_method=${submitMethod} state_changed=true`,
      );
    },
    shouldReopenRoom: async () => {
      const deadline = Date.now() + Math.min(timeout, 3_000);
      do {
        for (const page of context.pages()) {
          for (const frame of page.frames()) {
            const text = await frame.locator("body").innerText().catch(() => "");
            const explicitFailure = text.match(CONNECTION_ERROR);
            if (explicitFailure) {
              throw new Error(`HaxBall connection error: ${explicitFailure[0]}`);
            }
            if (/\bRoom list\b/i.test(text)) return true;
            const buttons = await frame.locator("button").allTextContents().catch(() => []);
            if (buttons.some((value) => /^\s*Leave\s*$/i.test(value))) return false;
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 100));
      } while (Date.now() < deadline);
      return false;
    },
    onRetry: async (attempt, reason) => {
      console.log(`join_retry: client=${clientId} attempt=${attempt} reason=${reason}`);
    },
    prepareRetry: async () => {
      await Promise.allSettled(context.pages().map((page) => page.close()));
      navigationPage = await context.newPage();
      attach(navigationPage);
    },
    findGameSurface: async () => {
      try {
        return await findGameSurfaceAcrossContext(context, { timeout });
      } catch (surfaceError) {
        const summary = await summarizeContext(context, clientId);
        const visibleError = summary.pages
          .flatMap((page) => page.frames)
          .find((frame) => frame.connection_error)?.connection_error;
        if (visibleError) throw new Error(`HaxBall connection error: ${visibleError}`);
        throw surfaceError;
      }
    },
    onSurfaceReady: async (surface) => {
      const box = await surface.canvas.boundingBox();
      const size = box ? `${Math.round(box.width)}x${Math.round(box.height)}` : "unknown";
      console.log(`client_surface_status: client=${clientId} ready=true size=${size}`);
      await onSurfaceReady(surface);
    },
    waitForHostJoin: async (name) => {
      const controller = new AbortController();
      try {
        return await Promise.race([
          waitForHostJoin(name, Math.min(timeout, 30_000), { signal: controller.signal }),
          waitForConnectionError(context, controller.signal),
        ]);
      } finally {
        controller.abort();
      }
    },
    onFailure: async (failedStage, error) => {
      const artifact = await captureJoinDiagnostics(context, {
        clientId,
        stage: failedStage,
        error,
        events,
      });
      console.error(
        `client_join_diagnostics: client=${clientId} stage=${failedStage} ` +
        `persisted=false artifact_failures=${artifact.failures.length}`,
      );
    },
  };

  return executeJoinLifecycle({
    adapter,
    nickname,
    roomUrl,
    onStage: (next) => stage(next),
    releaseInputs,
    requireGameSurface,
  });
}
