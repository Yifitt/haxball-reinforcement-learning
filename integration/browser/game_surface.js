const GAME_CANVAS_ATTRIBUTE = "data-haxball-integration-game-canvas";

export const DEFAULT_MINIMUM_CANVAS_SIZE = Object.freeze({
  width: 300,
  height: 150,
});

export function selectGameCanvasCandidate(
  canvases,
  minimumSize = DEFAULT_MINIMUM_CANVAS_SIZE,
) {
  return canvases
    .filter((canvas) =>
      canvas.attached &&
      canvas.visible &&
      canvas.box !== null &&
      canvas.box.width >= minimumSize.width &&
      canvas.box.height >= minimumSize.height)
    .sort((left, right) =>
      (right.box.width * right.box.height) - (left.box.width * left.box.height))[0] ?? null;
}

export function formatCanvasDiagnostics(
  canvases,
  minimumSize = DEFAULT_MINIMUM_CANVAS_SIZE,
) {
  const lines = canvases.length
    ? canvases.map((canvas) => {
      const rendered = canvas.box
        ? `${canvas.box.width.toFixed(1)}x${canvas.box.height.toFixed(1)}`
        : "none";
      return `index=${canvas.index} dom=${canvas.domWidth ?? "unset"}x${canvas.domHeight ?? "unset"} ` +
        `rendered=${rendered} visible=${canvas.visible} attached=${canvas.attached}`;
    })
    : ["no canvas elements discovered"];
  return [
    `No valid HaxBall game canvas (minimum rendered size ${minimumSize.width}x${minimumSize.height}).`,
    ...lines,
  ].join("\n");
}

export function gameCanvasDiagnosticError(
  canvases,
  minimumSize = DEFAULT_MINIMUM_CANVAS_SIZE,
) {
  return new Error(formatCanvasDiagnostics(canvases, minimumSize));
}

async function inspectCanvas(handle, index) {
  let attributes = { attached: false, domWidth: null, domHeight: null };
  try {
    attributes = await handle.evaluate((canvas) => ({
      attached: canvas.isConnected,
      domWidth: canvas.getAttribute("width"),
      domHeight: canvas.getAttribute("height"),
    }));
  } catch {
    return { index, ...attributes, visible: false, box: null, handle };
  }
  if (!attributes.attached) {
    return { index, ...attributes, visible: false, box: null, handle };
  }
  const visible = await handle.isVisible().catch(() => false);
  const box = await handle.boundingBox().catch(() => null);
  return { index, ...attributes, visible, box, handle };
}

export async function inspectFrameCanvases(frame) {
  const handles = await frame.locator("canvas").elementHandles();
  return Promise.all(handles.map(inspectCanvas));
}

async function markSelectedCanvas(frame, selected) {
  await selected.handle.evaluate((canvas, attribute) => {
    for (const marked of document.querySelectorAll(`canvas[${attribute}]`)) {
      marked.removeAttribute(attribute);
    }
    canvas.setAttribute(attribute, "true");
  }, GAME_CANVAS_ATTRIBUTE);
  return frame.locator(`canvas[${GAME_CANVAS_ATTRIBUTE}="true"]`);
}

export async function findGameCanvas(
  frame,
  {
    timeout = 30_000,
    minimumSize = DEFAULT_MINIMUM_CANVAS_SIZE,
    pollMilliseconds = 100,
  } = {},
) {
  const allCanvases = frame.locator("canvas");
  const deadline = Date.now() + timeout;
  try {
    await allCanvases.first().waitFor({ state: "attached", timeout });
  } catch {
    throw gameCanvasDiagnosticError([], minimumSize);
  }

  let observations = [];
  do {
    observations = await inspectFrameCanvases(frame);
    const selected = selectGameCanvasCandidate(observations, minimumSize);
    if (selected) {
      try {
        return await markSelectedCanvas(frame, selected);
      } catch {
        // The game replaced the selected canvas; inspect the new set on the next poll.
      }
    }
    await new Promise((resolve) => setTimeout(resolve, pollMilliseconds));
  } while (Date.now() < deadline);

  throw gameCanvasDiagnosticError(observations, minimumSize);
}

export async function findGameSurfaceAcrossContext(
  context,
  {
    timeout = 30_000,
    minimumSize = DEFAULT_MINIMUM_CANVAS_SIZE,
    pollMilliseconds = 100,
  } = {},
) {
  const deadline = Date.now() + timeout;
  let observations = [];
  do {
    observations = [];
    for (const page of context.pages()) {
      for (const frame of page.frames()) {
        const canvases = await inspectFrameCanvases(frame).catch(() => []);
        observations.push(...canvases.map((canvas) => ({ ...canvas, page, frame })));
      }
    }
    const selected = selectGameCanvasCandidate(observations, minimumSize);
    if (selected) {
      try {
        const canvas = await markSelectedCanvas(selected.frame, selected);
        return { page: selected.page, frame: selected.frame, canvas };
      } catch {
        // A navigation replaced the winning canvas. Re-enumerate all pages and frames.
      }
    }
    await new Promise((resolve) => setTimeout(resolve, pollMilliseconds));
  } while (Date.now() < deadline);
  throw gameCanvasDiagnosticError(observations, minimumSize);
}

export async function focusGameCanvas(frame, options = {}) {
  const { click = true, ...findOptions } = options;
  const timeout = findOptions.timeout ?? 30_000;
  const deadline = Date.now() + timeout;
  let lastError;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const remaining = Math.max(1, deadline - Date.now());
      const canvas = await findGameCanvas(frame, { ...findOptions, timeout: remaining });
      if (click) await canvas.click({ timeout: Math.min(2_000, remaining) });
      const focused = await canvas.evaluate((element) => {
        element.tabIndex = -1;
        element.focus({ preventScroll: true });
        return document.activeElement === element;
      });
      if (!focused) throw new Error("selected game canvas could not receive focus");
      return canvas;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

export async function focusGameSurface(context, options = {}) {
  const { click = true, timeout = 30_000, ...findOptions } = options;
  const deadline = Date.now() + timeout;
  let lastError;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const remaining = Math.max(1, deadline - Date.now());
      const surface = await findGameSurfaceAcrossContext(context, {
        ...findOptions,
        timeout: remaining,
      });
      if (click) await surface.canvas.click({ timeout: Math.min(2_000, remaining) });
      const focused = await surface.canvas.evaluate((element) => {
        element.tabIndex = -1;
        element.focus({ preventScroll: true });
        return document.activeElement === element;
      });
      if (!focused) throw new Error("selected game canvas could not receive focus");
      return surface;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}
