import { joinPrivateRoom } from "./join_lifecycle.js";

export async function gameFrame(page) {
  await page.waitForSelector("iframe.gameframe", { timeout: 30_000 });
  const frame = page.frames().find((candidate) => candidate !== page.mainFrame());
  if (!frame) throw new Error("HaxBall game frame did not load");
  return frame;
}

export async function joinRoom(context, roomUrl, nickname, options = {}) {
  const result = await joinPrivateRoom({ context, roomUrl, nickname, ...options });
  const page = result.page ?? context.pages().find((candidate) => !candidate.isClosed()) ?? null;
  return { ...result, page, context };
}
