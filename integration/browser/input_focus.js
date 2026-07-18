const PROBE_KEY = "ArrowLeft";
const MAX_DIAGNOSTIC_TEXT = 120;

const bounded = (value) => String(value ?? "").replace(/\s+/g, " ").trim().slice(0, MAX_DIAGNOSTIC_TEXT);
const roundedBox = (box) => box && ({
  x: Math.round(box.x * 10) / 10,
  y: Math.round(box.y * 10) / 10,
  width: Math.round(box.width * 10) / 10,
  height: Math.round(box.height * 10) / 10,
});

export function selectSafeCanvasPoint(points) {
  return points.find((point) => point.safe === true) ?? null;
}

export function formatPointerInterceptionDiagnostics(diagnostics) {
  const blocker = diagnostics.blocker
    ? `${bounded(diagnostics.blocker.tag)}.${bounded(diagnostics.blocker.className)}` +
      `[data-hook=${bounded(diagnostics.blocker.dataHook)}]`
    : "none";
  const canvas = diagnostics.canvasBounds;
  const overlay = diagnostics.blocker?.bounds;
  return `canvas=${canvas.width}x${canvas.height} ` +
    `blocker=${blocker} blocker_bounds=${overlay ? `${overlay.width}x${overlay.height}` : "none"} ` +
    `pointer_events=${bounded(diagnostics.blocker?.pointerEvents ?? "none")} ` +
    `coverage=${diagnostics.coverage} safe_points=${diagnostics.points.filter((point) => point.safe).length}`;
}

export async function inspectPointerInterception(canvas) {
  return canvas.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const fractions = [
      [0.5, 0.5], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
      [0.25, 0.5], [0.75, 0.5], [0.5, 0.25], [0.5, 0.75],
    ];
    const describe = (target) => {
      if (!target) return null;
      const targetBox = target.getBoundingClientRect();
      return {
        tag: target.tagName.toLowerCase().slice(0, 40),
        className: String(target.className ?? "").replace(/\s+/g, " ").slice(0, 120),
        dataHook: String(target.getAttribute?.("data-hook") ?? "").slice(0, 120),
        pointerEvents: getComputedStyle(target).pointerEvents,
        bounds: {
          x: targetBox.x, y: targetBox.y,
          width: targetBox.width, height: targetBox.height,
        },
      };
    };
    const points = fractions.map(([fx, fy]) => {
      const x = box.left + box.width * fx;
      const y = box.top + box.height * fy;
      const hit = document.elementFromPoint(x, y);
      return { fx, fy, x, y, safe: hit === element, hit: describe(hit) };
    });
    const blockerPoint = points.find((point) => !point.safe && point.hit);
    const blocker = blockerPoint?.hit ?? null;
    const safeCount = points.filter((point) => point.safe).length;
    return {
      canvasBounds: { x: box.x, y: box.y, width: box.width, height: box.height },
      blocker,
      coverage: safeCount === 0 ? "full" : safeCount === points.length ? "none" : "partial",
      points,
      activeElement: describe(document.activeElement),
      canvasTabIndex: element.getAttribute("tabindex"),
      documentHasFocus: document.hasFocus(),
    };
  });
}

async function focusDomElement(locator, activeName) {
  return locator.evaluate((element, name) => {
    const originalTabIndex = element.getAttribute("tabindex");
    const assignedTemporaryTabIndex = originalTabIndex === null;
    if (assignedTemporaryTabIndex) element.setAttribute("tabindex", "-1");
    element.focus({ preventScroll: true });
    return {
      focused: document.activeElement === element,
      activeElement: name,
      originalTabIndex,
      assignedTemporaryTabIndex,
    };
  }, activeName);
}

async function restoreTemporaryTabIndex(locator, focusResult) {
  if (!focusResult?.assignedTemporaryTabIndex) return;
  await locator.evaluate((element) => {
    if (element.getAttribute("tabindex") === "-1") element.removeAttribute("tabindex");
  }).catch(() => {});
}

export async function verifyKeyboardDelivery(frame, keyboard, key = PROBE_KEY) {
  const probeId = `__haxballKeyboardProbe_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  await frame.evaluate(({ id, expectedCode }) => {
    const record = { down: false, up: false };
    record.onDown = (event) => { if (event.code === expectedCode) record.down = true; };
    record.onUp = (event) => { if (event.code === expectedCode) record.up = true; };
    window.addEventListener("keydown", record.onDown, true);
    window.addEventListener("keyup", record.onUp, true);
    window[id] = record;
  }, { id: probeId, expectedCode: key });
  let deliveryError = null;
  try {
    await keyboard.down(key);
  } catch (error) {
    deliveryError = error;
  } finally {
    await keyboard.up(key).catch(() => {});
  }
  const verified = await frame.evaluate((id) => {
    const record = window[id];
    if (!record) return false;
    window.removeEventListener("keydown", record.onDown, true);
    window.removeEventListener("keyup", record.onUp, true);
    delete window[id];
    return record.down === true && record.up === true;
  }, probeId);
  if (deliveryError) throw deliveryError;
  return verified;
}

export async function runFocusStrategies(strategies, verify) {
  const failures = [];
  for (const strategy of strategies) {
    try {
      const result = await strategy.run();
      if (!result?.focused) {
        failures.push(`${strategy.name}: focus verification failed`);
        continue;
      }
      if (!await verify(strategy.name, result)) {
        failures.push(`${strategy.name}: keyboard verification failed`);
        continue;
      }
      return { method: strategy.name, ...result, keyboardVerified: true };
    } catch (error) {
      failures.push(`${strategy.name}: ${bounded(error.message)}`);
    }
  }
  throw new Error(`keyboard focus preparation failed; ${failures.join("; ")}`);
}

export async function prepareKeyboardInput(
  { page, frame, canvas },
  keyboard,
  {
    verificationKey = PROBE_KEY,
    allowForcedClick = false,
    verify = (method) => verifyKeyboardDelivery(frame, keyboard, verificationKey, method),
  } = {},
) {
  await page.bringToFront();
  if (page.isClosed() || frame.isDetached()) {
    throw new Error("keyboard focus target page or frame is detached");
  }
  const attached = await canvas.evaluate((element) => element.isConnected).catch(() => false);
  if (!attached) throw new Error("keyboard focus target canvas is detached");

  const pointerDiagnostics = await inspectPointerInterception(canvas);
  const alreadyPrepared = await canvas.evaluate((element) =>
    element.getAttribute("data-haxball-integration-keyboard-ready") === "true" &&
    document.hasFocus());
  if (alreadyPrepared) {
    return {
      method: "dom_focus",
      activeElement: "canvas",
      keyboardVerified: true,
      alreadyPrepared: true,
      pointerDiagnostics: {
        ...pointerDiagnostics,
        canvasBounds: roundedBox(pointerDiagnostics.canvasBounds),
        blocker: pointerDiagnostics.blocker && {
          ...pointerDiagnostics.blocker,
          bounds: roundedBox(pointerDiagnostics.blocker.bounds),
        },
      },
    };
  }
  const safePoint = selectSafeCanvasPoint(pointerDiagnostics.points);
  const canvasBox = await canvas.boundingBox();
  let canvasFocus;
  let bodyFocus;
  const strategies = [
    {
      name: "dom_focus",
      run: async () => {
        canvasFocus = await focusDomElement(canvas, "canvas");
        return canvasFocus;
      },
    },
    {
      name: "body_focus",
      run: async () => {
        bodyFocus = await focusDomElement(frame.locator("body"), "body");
        return bodyFocus;
      },
    },
  ];
  if (safePoint && canvasBox) {
    strategies.push({
      name: "safe_point_click",
      run: async () => {
        const x = canvasBox.x + canvasBox.width * safePoint.fx;
        const y = canvasBox.y + canvasBox.height * safePoint.fy;
        await page.mouse.click(x, y);
        const focused = await frame.evaluate(() => document.hasFocus());
        return { focused, activeElement: "page", clickPoint: { x, y } };
      },
    });
  }
  if (allowForcedClick) {
    strategies.push({
      name: "forced_click",
      run: async () => {
        await canvas.click({ force: true, timeout: 2_000 });
        const focused = await frame.evaluate(() => document.hasFocus());
        return { focused, activeElement: "page" };
      },
    });
  }

  try {
    const result = await runFocusStrategies(strategies, verify);
    await canvas.evaluate((element) => element.setAttribute(
      "data-haxball-integration-keyboard-ready", "true"));
    return {
      ...result,
      pointerDiagnostics: {
        ...pointerDiagnostics,
        canvasBounds: roundedBox(pointerDiagnostics.canvasBounds),
        blocker: pointerDiagnostics.blocker && {
          ...pointerDiagnostics.blocker,
          bounds: roundedBox(pointerDiagnostics.blocker.bounds),
        },
      },
    };
  } catch (error) {
    throw new Error(
      `${bounded(error.message)}; pointer_interception: ` +
      formatPointerInterceptionDiagnostics(pointerDiagnostics),
    );
  } finally {
    await restoreTemporaryTabIndex(canvas, canvasFocus);
    await restoreTemporaryTabIndex(frame.locator("body"), bodyFocus);
    await keyboard.up(verificationKey).catch(() => {});
  }
}
