## \file observations.py
# \brief Fresh INDI definitions scoped to transport and device lifetimes.
"""Use the client's public transport callbacks; never promote an old cache on reconnect."""

import threading
import time
import typing
from copy import deepcopy
from purepyindi2 import constants, messages


class Observations:
    """Track fresh definitions independently of PurePyINDI2's retained value cache."""

    def __init__(self, client, devices, timeout=5.0, heartbeat_device=None):
        """Subscribe before discovery and expire silent devices even while TCP stays connected."""
        self.client, self.devices, self.timeout = client, tuple(devices), timeout
        self.lock = threading.RLock()
        self.defined = {}
        self.values = {}
        self.heartbeat_device = heartbeat_device
        self.heartbeat = None
        self.heartbeat_at = -float("inf")
        self.epoch = 0
        self.last_request = -float("inf")
        client.register_callback(self.observe)
        for event in (
            constants.TransportEvent.connection,
            constants.TransportEvent.disconnection,
        ):
            client.connection.add_callback(event, self.connection_changed)

    def connection_changed(self, status):
        """Invalidate all observations for every transport lifetime boundary."""
        with self.lock:
            self.epoch += 1
            self.defined.clear()
            self.values.clear()
            self.heartbeat = None
            self.heartbeat_at = -float("inf")
            self.last_request = -float("inf")

    def observe(self, message):
        """Only definitions establish capability; updates refresh already defined properties."""
        with self.lock:
            if isinstance(message, messages.DelProperty):
                if message.device in (None, self.heartbeat_device) and message.name in (
                    None,
                    "model_status",
                    "reload_engine",
                ):
                    self.epoch += 1
                for key in list(self.defined):
                    if (message.device is None or key[0] == message.device) and (
                        message.name is None or key[1] == message.name
                    ):
                        del self.defined[key]
                        self.values.pop(key, None)
                if (self.heartbeat_device, "model_status") not in self.defined:
                    self.heartbeat = None
                    self.heartbeat_at = -float("inf")
                return
            if (
                message.device not in self.devices
                or self.client.status != constants.ConnectionStatus.CONNECTED
            ):
                return
            key = (message.device, message.name)
            if isinstance(message, typing.get_args(messages.IndiDefMessage)):
                self.defined[key] = time.monotonic()
            elif key in self.defined and isinstance(
                message, typing.get_args(messages.IndiSetMessage)
            ):
                self.defined[key] = time.monotonic()
            if key in self.defined:
                # Publish a completed message snapshot, never the client's next in-place mutation.
                self.values[key] = deepcopy(
                    self.client[f"{message.device}.{message.name}"]
                )
            if key == (self.heartbeat_device, "model_status") and key in self.defined:
                # Repeated definitions prove transport liveness, but cannot hide a stalled appLogic.
                prop = self.values[key]
                if (
                    "instance" in prop
                    and "sequence" in prop
                    and prop["instance"]
                    and prop["sequence"]
                ):
                    beat = (prop["instance"], prop["sequence"])
                    if beat != self.heartbeat:
                        self.heartbeat, self.heartbeat_at = beat, time.monotonic()

    def refresh(self):
        """Request definitions periodically so unchanged live devices remain distinguishable from silence."""
        if self.client.status != constants.ConnectionStatus.CONNECTED:
            return
        with self.lock:
            now = time.monotonic()
            if now - self.last_request < min(1.0, self.timeout / 2):
                return
            self.last_request = now
        for device in self.devices:
            self.client.get_properties(device)

    def vector(self, device, prop):
        """Reject undefined, expired or prior-connection observations before returning values."""
        with self.lock:
            if (
                self.client.status != constants.ConnectionStatus.CONNECTED
                or time.monotonic() - self.defined.get((device, prop), -float("inf"))
                > self.timeout
                or (
                    device == self.heartbeat_device
                    and time.monotonic() - self.heartbeat_at > self.timeout
                )
            ):
                raise ConnectionError(
                    f"Fresh definition/readback unavailable: {device}.{prop}"
                )
            return self.values[device, prop]

    def close(self):
        """Detach callbacks when the owning app/CLI finishes."""
        self.client.unregister_callback(self.observe)
        for event in (
            constants.TransportEvent.connection,
            constants.TransportEvent.disconnection,
        ):
            self.client.connection.remove_callback(event, self.connection_changed)
