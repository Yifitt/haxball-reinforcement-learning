import { createHash } from "node:crypto";

function roomCodeHash(roomCode) {
  return createHash("sha256").update(roomCode, "utf8").digest("hex");
}

function parseRoomUrl(value) {
  if (typeof value !== "string" || value.length === 0 || value.trim() !== value) {
    throw new Error("room URL must be a non-empty string without surrounding whitespace");
  }

  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("room URL is not a valid URL");
  }
  if (parsed.protocol !== "https:") throw new Error("room URL protocol must be HTTPS");
  if (parsed.hostname !== "www.haxball.com") {
    throw new Error("room URL host must be www.haxball.com");
  }
  if (parsed.pathname !== "/play") throw new Error("room URL path must be /play");
  if (parsed.username || parsed.password || parsed.hash) {
    throw new Error("room URL must not contain credentials or a fragment");
  }

  const roomCodes = parsed.searchParams.getAll("c");
  if (roomCodes.length !== 1 || roomCodes[0].length === 0) {
    throw new Error("room URL must contain exactly one non-empty room code");
  }
  const roomCode = roomCodes[0];
  if (/\s|["']/.test(roomCode)) {
    throw new Error("room code must not contain whitespace or quotation marks");
  }
  if (/%[0-9a-f]{2}/i.test(roomCode)) {
    throw new Error("room code appears to be URL-encoded more than once");
  }
  return { parsed, roomCode, roomCodeHash: roomCodeHash(roomCode) };
}

export function validateRoomUrl(roomUrl, hostRoomUrl = roomUrl) {
  const navigation = parseRoomUrl(roomUrl);
  const host = parseRoomUrl(hostRoomUrl);
  return {
    host: navigation.parsed.hostname,
    path: navigation.parsed.pathname,
    roomCodePresent: true,
    roomCodeLength: navigation.roomCode.length,
    roomCodeHashMatch: navigation.roomCodeHash === host.roomCodeHash,
  };
}

