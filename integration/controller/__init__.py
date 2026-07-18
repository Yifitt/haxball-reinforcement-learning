"""Python side of the local state/action bridge."""

from .protocol import StateMessage, validate_state_message

__all__ = ["BridgeClient", "StateMessage", "validate_state_message"]


def __getattr__(name: str):
    if name == "BridgeClient":
        from .client import BridgeClient

        return BridgeClient
    raise AttributeError(name)
