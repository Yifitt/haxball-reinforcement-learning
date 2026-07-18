(() => {
  "use strict";

  const QueueStates = Object.freeze({
    WAITING_FOR_HUMAN: "WAITING_FOR_HUMAN",
    READY: "READY",
    MATCH_RUNNING: "MATCH_RUNNING",
    MATCH_ENDING: "MATCH_ENDING",
    ROTATING: "ROTATING",
    WAITING_FOR_BOT: "WAITING_FOR_BOT",
  });

  // Keep user-facing text in one place so queue wording can change without
  // touching lifecycle transitions.
  const QueueAnnouncements = Object.freeze({
    position(position) {
      return position === 1
        ? "You are next."
        : `You are #${position} in the queue.`;
    },
    turnStarted: () => "Your turn has started.",
    promoted: () => "A queued player's turn has started.",
    waiting(count) {
      return `${count} ${count === 1 ? "player is" : "players are"} waiting.`;
    },
  });

  class PublicPlayerQueue {
    constructor({ enabled = true, matchesPerTurn = 1 } = {}) {
      if (!Number.isInteger(matchesPerTurn) || matchesPerTurn < 1) {
        throw new Error("matchesPerTurn must be a positive integer");
      }
      this.enabled = enabled === true;
      this.matchesPerTurn = matchesPerTurn;
      this.connectedHumanIds = new Set();
      this.waitingIds = [];
      this.activeHumanId = null;
      this.botReady = false;
      this.gameRunning = false;
      this.completionObserved = false;
      this.interruptionPending = false;
      this.rotationPending = false;
      this.matchesCompletedThisTurn = 0;
      this.state = QueueStates.WAITING_FOR_BOT;
    }

    _updateState() {
      if (!this.botReady) {
        this.state = QueueStates.WAITING_FOR_BOT;
      } else if (this.gameRunning) {
        this.state = (this.completionObserved || this.interruptionPending)
          ? QueueStates.MATCH_ENDING
          : QueueStates.MATCH_RUNNING;
      } else if (this.activeHumanId === null) {
        this.state = QueueStates.WAITING_FOR_HUMAN;
      } else {
        this.state = QueueStates.READY;
      }
    }

    setBotReady(ready) {
      this.botReady = ready === true;
      this._updateState();
      return this.snapshot();
    }

    reconcilePlayerQueue({ promote = !this.gameRunning } = {}) {
      if (this.activeHumanId !== null && !this.connectedHumanIds.has(this.activeHumanId)) {
        this.activeHumanId = null;
      }
      const seen = new Set();
      this.waitingIds = this.waitingIds.filter((id) => {
        if (!this.connectedHumanIds.has(id) || id === this.activeHumanId || seen.has(id)) {
          return false;
        }
        seen.add(id);
        return true;
      });
      let promoted = null;
      if (promote && this.activeHumanId === null) {
        promoted = this.promoteNextHuman();
      }
      this._updateState();
      return { promoted, snapshot: this.snapshot() };
    }

    promoteNextHuman() {
      if (this.activeHumanId !== null || this.gameRunning) return null;
      while (this.waitingIds.length) {
        const candidate = this.waitingIds.shift();
        if (!this.connectedHumanIds.has(candidate)) continue;
        this.activeHumanId = candidate;
        this.matchesCompletedThisTurn = 0;
        this._updateState();
        return candidate;
      }
      this._updateState();
      return null;
    }

    addHuman(id) {
      if (!Number.isInteger(id) || id < 0) throw new Error("human id must be a non-negative integer");
      if (this.connectedHumanIds.has(id)) {
        return { added: false, duplicate: true, active: this.activeHumanId === id, position: this.positionOf(id) };
      }
      this.connectedHumanIds.add(id);
      let promoted = null;
      if (this.activeHumanId === null && !this.gameRunning) {
        this.activeHumanId = id;
        this.matchesCompletedThisTurn = 0;
        promoted = id;
      } else if (id !== this.activeHumanId) {
        this.waitingIds.push(id);
      }
      this.reconcilePlayerQueue({ promote: !this.gameRunning });
      return {
        added: true,
        duplicate: false,
        active: this.activeHumanId === id,
        promoted,
        position: this.positionOf(id),
      };
    }

    removeHuman(id) {
      if (!this.connectedHumanIds.has(id)) {
        return { removed: false, wasActive: false, wasQueued: false, promoted: null };
      }
      const wasActive = this.activeHumanId === id;
      const wasQueued = this.waitingIds.includes(id);
      this.connectedHumanIds.delete(id);
      this.waitingIds = this.waitingIds.filter((candidate) => candidate !== id);
      if (wasActive) {
        this.activeHumanId = null;
        this.matchesCompletedThisTurn = 0;
        this.rotationPending = false;
        this.interruptionPending = this.gameRunning;
      }
      const { promoted } = this.reconcilePlayerQueue({ promote: !this.gameRunning });
      return { removed: true, wasActive, wasQueued, promoted };
    }

    beginMatch() {
      if (this.gameRunning) return false;
      this.reconcilePlayerQueue();
      if (!this.botReady || this.activeHumanId === null) return false;
      this.gameRunning = true;
      this.completionObserved = false;
      this.interruptionPending = false;
      this.rotationPending = false;
      this._updateState();
      return true;
    }

    completeMatch() {
      if (!this.gameRunning || this.completionObserved) return false;
      this.completionObserved = true;
      this.matchesCompletedThisTurn += 1;
      this.reconcilePlayerQueue({ promote: false });
      this.rotationPending = this.enabled && this.waitingIds.length > 0 &&
        this.matchesCompletedThisTurn >= this.matchesPerTurn;
      this._updateState();
      return true;
    }

    stopMatch() {
      if (!this.gameRunning && !this.completionObserved && !this.interruptionPending) {
        this.reconcilePlayerQueue();
        return { stopped: false, rotated: false, outgoing: null, promoted: null };
      }
      const outgoing = this.activeHumanId;
      let rotated = false;
      let promoted = null;
      this.gameRunning = false;
      if (this.completionObserved && this.rotationPending && this.waitingIds.length > 0) {
        this.state = QueueStates.ROTATING;
        if (outgoing !== null && this.connectedHumanIds.has(outgoing) &&
            !this.waitingIds.includes(outgoing)) {
          this.waitingIds.push(outgoing);
        }
        this.activeHumanId = null;
        this.matchesCompletedThisTurn = 0;
        promoted = this.promoteNextHuman();
        rotated = promoted !== null && promoted !== outgoing;
      } else if (this.interruptionPending || this.activeHumanId === null) {
        this.activeHumanId = null;
        this.matchesCompletedThisTurn = 0;
        promoted = this.promoteNextHuman();
      } else if (this.completionObserved && this.waitingIds.length === 0) {
        // Nobody was waiting at official match completion: grant an automatic
        // rematch without a spectator round-trip.
        this.matchesCompletedThisTurn = 0;
      }
      this.completionObserved = false;
      this.interruptionPending = false;
      this.rotationPending = false;
      this.reconcilePlayerQueue();
      return { stopped: true, rotated, outgoing, promoted };
    }

    positionOf(id) {
      const index = this.waitingIds.indexOf(id);
      return index < 0 ? null : index + 1;
    }

    snapshot() {
      return {
        enabled: this.enabled,
        matchesPerTurn: this.matchesPerTurn,
        state: this.state,
        botReady: this.botReady,
        gameRunning: this.gameRunning,
        activeHumanId: this.activeHumanId,
        waitingIds: [...this.waitingIds],
        connectedHumanIds: [...this.connectedHumanIds],
        matchesCompletedThisTurn: this.matchesCompletedThisTurn,
      };
    }

    assertInvariants() {
      const waiting = new Set(this.waitingIds);
      if (waiting.size !== this.waitingIds.length) throw new Error("queue contains duplicates");
      if (this.activeHumanId !== null && waiting.has(this.activeHumanId)) {
        throw new Error("active human is also queued");
      }
      if (this.activeHumanId !== null && !this.connectedHumanIds.has(this.activeHumanId)) {
        throw new Error("active human is disconnected");
      }
      for (const id of waiting) {
        if (!this.connectedHumanIds.has(id)) throw new Error("queued human is disconnected");
      }
      return true;
    }
  }

  window.HaxballPlayerQueue = Object.freeze({
    PublicPlayerQueue,
    QueueAnnouncements,
    QueueStates,
  });
})();
