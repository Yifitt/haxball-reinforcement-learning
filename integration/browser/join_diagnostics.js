export function sanitizeUrl(value) {
  try {
    const url = new URL(value);
    for (const key of url.searchParams.keys()) url.searchParams.set(key, "[REDACTED]");
    if (url.hash) url.hash = "#[REDACTED]";
    return url.toString();
  } catch {
    return "[invalid-url]";
  }
}

export function redactDiagnosticText(value) {
  return String(value ?? "")
    .replace(/https?:\/\/[^\s"'<>]+/gi, (url) => sanitizeUrl(url))
    .replace(/\b(token|auth|key|password)=([^\s&]+)/gi, "$1=[REDACTED]");
}

export function attachPageDiagnostics(page, events, clientId) {
  page.on("console", (entry) => {
    if (entry.type() === "error") events.push({
      type: "console_error",
      clientId,
      text: redactDiagnosticText(entry.text()),
    });
  });
  page.on("pageerror", (error) =>
    events.push({ type: "page_error", clientId, text: redactDiagnosticText(error.message) }));
  page.on("requestfailed", (request) => events.push({
    type: "request_failed",
    clientId,
    url: sanitizeUrl(request.url()),
    text: redactDiagnosticText(request.failure()?.errorText ?? "unknown"),
  }));
  page.on("websocket", (socket) => {
    socket.on("close", () => events.push({
      type: "websocket_closed",
      clientId,
      url: sanitizeUrl(socket.url()),
    }));
  });
}

async function inspectFrame(frame, index) {
  try {
    return await frame.evaluate(({ frameIndex, sanitizedUrl }) => {
      const visible = (element) => {
        const style = getComputedStyle(element);
        const box = element.getBoundingClientRect();
        return style.display !== "none" && style.visibility !== "hidden" && box.width > 0 && box.height > 0;
      };
      const short = (text) => String(text ?? "").replace(/\s+/g, " ").trim().slice(0, 160);
      const text = short(document.body?.innerText);
      return {
        index: frameIndex,
        url: sanitizedUrl,
        loading: document.readyState !== "complete",
        canvas_count: document.querySelectorAll("canvas").length,
        nickname_input: Boolean(document.querySelector('.choose-nickname-view input[data-hook="input"]')),
        inputs: [...document.querySelectorAll("input")].filter(visible).slice(0, 8).map((input) => ({
          type: input.type,
          placeholder: short(input.placeholder),
          label: short(input.labels?.[0]?.innerText),
          data_hook: input.getAttribute("data-hook"),
        })),
        buttons: [...document.querySelectorAll("button")].filter(visible).slice(0, 12).map((button) =>
          short(button.innerText || button.getAttribute("aria-label"))),
        join_button: [...document.querySelectorAll("button")].filter(visible).some((button) =>
          /^(ok|join|play)$/i.test(short(button.innerText || button.getAttribute("aria-label")))),
        text_excerpt: text,
        connection_error: /connection (failed|closed)|room (closed|full)|disconnected|browser incompatib|webrtc/i.test(text)
          ? text
          : null,
      };
    }, { frameIndex: index, sanitizedUrl: sanitizeUrl(frame.url()) });
  } catch (error) {
    return { index, url: sanitizeUrl(frame.url()), inspection_error: error.message };
  }
}

export async function summarizeContext(context, clientId) {
  const pages = context.pages();
  return {
    client_id: clientId,
    page_count: pages.length,
    pages: await Promise.all(pages.map(async (page, index) => ({
      index,
      url: sanitizeUrl(page.url()),
      title: await page.title().catch(() => ""),
      main_frame_url: sanitizeUrl(page.mainFrame().url()),
      frames: await Promise.all(page.frames().map(inspectFrame)),
    }))),
  };
}

export async function captureJoinDiagnostics(
  context,
  {
    clientId,
    stage,
    error,
    events = [],
  },
) {
  const failures = [];
  let summary = null;
  try {
    summary = await summarizeContext(context, clientId);
  } catch (inspectionError) {
    failures.push(`inspection: ${redactDiagnosticText(inspectionError.message)}`);
  }
  return {
    stage,
    error: redactDiagnosticText(error?.message ?? String(error)),
    summary,
    events: events.slice(-50),
    failures,
  };
}
