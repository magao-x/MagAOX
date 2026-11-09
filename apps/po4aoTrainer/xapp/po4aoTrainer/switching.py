## \file switching.py
# \brief Fresh INDI observations and completion-aware PO4AO runtime control.
"""NN/CACAO switching belongs exclusively to nnReconstructor."""

from copy import deepcopy
import threading
import time
import typing

from purepyindi2 import constants, messages
from purepyindi2.constants import PropertyState, SwitchState


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


class Runtime:
    """Narrow transport adapter shared by supervisor operations and offline fixtures."""

    def __init__(self, client, po4ao, timeout=30, *, freshness_timeout=5.0):
        """Observe only the PO4AO runtime; reconstruction has its own C++ owner."""
        self.client, self.po4ao, self.timeout = client, po4ao, timeout
        self.observations = Observations(
            client, (po4ao,), freshness_timeout, heartbeat_device=po4ao
        )
        self.reload_ticket = None
        self.observations.refresh()

    def get(self, device, prop, element):
        """Read a current-lifetime, non-expired INDI observation."""
        return self.observations.vector(device, prop)[element]

    def send(self, device, prop, values):
        """Send through a freshly discovered typed property."""
        vector = self.observations.vector(device, prop)
        self.client.connection.send(vector.make_new_property(**values))

    def close(self):
        """Release observation/reload callbacks without changing device state."""
        if self.reload_ticket is not None:
            self.client.unregister_callback(
                self.reload_ticket["callback"], self.po4ao, "reload_engine"
            )
        self.observations.close()

    def wait(self, predicate, cancel, label, timeout=None):
        """Wait boundedly for observed completion, with cooperative cancellation."""
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            self.observations.refresh()
            if predicate():
                break
            if cancel.wait(0.05):
                raise InterruptedError(label + " cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(label + " did not complete")
        if cancel.is_set():
            raise InterruptedError(label + " cancelled")

    def runtime_state(self):
        """Read the identity and requested/applied control generations used for reconciliation."""
        vector = self.observations.vector(self.po4ao, "model_status")
        state = {
            name: vector[name]
            for name in (
                "instance",
                "reset_request",
                "reset_applied",
                "control_request",
                "control_target",
                "active_id",
                "reload",
            )
        }
        if (
            not state["instance"]
            or state["control_target"] not in ("on", "off")
            or any(
                not str(state[name]).isdigit()
                for name in ("reset_request", "reset_applied", "control_request")
            )
            or len(state["active_id"]) != 16
        ):
            raise ConnectionError("Runtime status is not fully initialized")
        state["run_phase"] = vector["run_phase"] if "run_phase" in vector else "unverified"
        return state

    def request_control(self, enabled):
        """Queue controller selection without waiting for an input frame."""
        self.send(
            self.po4ao,
            "control_enable",
            {"toggle": SwitchState.ON if enabled else SwitchState.OFF},
        )

    def control_complete(self, enabled):
        """Require the requested selection to have reached an emitted frame."""
        return (
            self.get(self.po4ao, "control_enable", "toggle")
            == (SwitchState.ON if enabled else SwitchState.OFF)
            and self.observations.vector(self.po4ao, "control_enable").state
            == PropertyState.OK
        )

    def control(self, enabled, cancel):
        """Request selection and wait for post-frame acknowledgement."""
        if cancel.is_set():
            raise InterruptedError("PO4AO control cancelled")
        self.request_control(enabled)
        self.wait(lambda: self.control_complete(enabled), cancel, "PO4AO control")

    def request_reset(self):
        """Send fallback and generation invalidation before waiting for any frame or worker."""
        ticket = {
            "instance": self.get(self.po4ao, "model_status", "instance"),
            "after": int(self.get(self.po4ao, "model_status", "reset_request")),
        }
        self.request_control(False)
        self.send(self.po4ao, "reset_model", {"toggle": SwitchState.ON})
        return ticket

    def reset_complete(self, ticket, external=False):
        """Require post-frame reset completion in the intended runtime lifetime."""
        same = self.get(self.po4ao, "model_status", "instance") == ticket["instance"]
        reset = int(self.get(self.po4ao, "model_status", "reset_applied"))
        return (
            same
            and reset > ticket["after"]
            and (
                external
                or (
                    self.get(self.po4ao, "control_enable", "toggle") == SwitchState.OFF
                    and self.observations.vector(self.po4ao, "control_enable").state
                    == PropertyState.OK
                )
            )
        )

    def begin_reload(self, model_id, issued=None):
        """Keep a ticket after issuance; cancelling a caller's wait cannot cancel C++ loading."""
        if self.reload_ticket is not None:
            if self.reload_outcome() is None:
                raise RuntimeError("A runtime reload is still unresolved")
            self.client.unregister_callback(
                self.reload_ticket["callback"], self.po4ao, "reload_engine"
            )
        ticket = dict(
            model_id=model_id,
            instance=self.get(self.po4ao, "model_status", "instance"),
            reset=int(self.get(self.po4ao, "model_status", "reset_request")),
            epoch=self.observations.epoch,
            issued=False,
            busy=False,
            rejected=False,
            terminal=False,
            outcome=None,
        )

        def acknowledge(message):
            """Only post-issuance set vectors can acknowledge this observation-lifetime ticket."""
            if (
                not ticket["issued"]
                or self.reload_ticket is not ticket
                or ticket["outcome"] is not None
                or self.observations.epoch != ticket["epoch"]
                or not isinstance(message, messages.SetSwitchVector)
            ):
                return
            state = message.state
            if state == PropertyState.BUSY:
                ticket["busy"] = True
            elif state == PropertyState.ALERT:
                ticket["rejected"] = not ticket["busy"]
                ticket["terminal"] = True
            elif ticket["busy"] and state == PropertyState.OK:
                ticket["terminal"] = True

        ticket["callback"] = acknowledge
        self.reload_ticket = ticket
        self.client.register_callback(acknowledge, self.po4ao, "reload_engine")
        if issued:
            issued(ticket)
        ticket["issued"] = True
        self.send(self.po4ao, "reload_engine", {"toggle": SwitchState.ON})

    def reload_outcome(self, ticket=None):
        """Return a settled load/reset outcome, retaining unresolved tickets through timeouts."""
        ticket = self.reload_ticket if ticket is None else ticket
        if ticket is None:
            return None
        if ticket["outcome"] is not None:
            return ticket["outcome"]
        if (
            self.get(self.po4ao, "model_status", "instance") != ticket["instance"]
            or int(self.get(self.po4ao, "model_status", "reset_request"))
            != ticket["reset"]
        ):
            ticket["outcome"] = "invalidated"
        elif ticket["rejected"] and not ticket["busy"]:
            # A direct Alert rejects admission, even for an already-active model.
            # Do not discard an existing load while its status is still loading/ready.
            if (
                self.observations.epoch == ticket["epoch"]
                and self.get(self.po4ao, "model_status", "reload") in ("idle", "failed")
            ):
                ticket["outcome"] = "failed"
        elif ticket["terminal"] or (
            ticket["busy"] and self.observations.epoch != ticket["epoch"]
        ):
            state = self.get(self.po4ao, "model_status", "reload")
            if state == "failed":
                ticket["outcome"] = "failed"
            elif (
                state == "idle"
                and self.get(self.po4ao, "model_status", "active_id")
                == ticket["model_id"]
            ):
                ticket["outcome"] = "loaded"
        return ticket["outcome"]

    def finish_reload(self, cancel, ticket=None):
        """Wait on the captured load even if a later request replaces the runtime's current ticket."""
        ticket = self.reload_ticket if ticket is None else ticket
        self.wait(
            lambda: self.reload_outcome(ticket) is not None,
            cancel,
            "PO4AO engine reload",
        )
        return ticket["outcome"]

    def reload(self, model_id, cancel):
        """Convenience operation for callers that do not own publication bookkeeping."""
        self.begin_reload(model_id)
        if self.finish_reload(cancel) != "loaded":
            raise RuntimeError("PO4AO reload failed or was invalidated")

    def recollect(self, cancel, timeout):
        """Require a newer complete frozen snapshot; interrupted collection never resumes learning."""
        old = int(self.get(self.po4ao, "collection_status", "requested_generation"))
        self.send(self.po4ao, "recollect_warmup", {"toggle": SwitchState.ON})

        def complete():
            """Pair a complete snapshot with this newer requested collection generation."""
            generation = int(self.get(self.po4ao, "collection_status", "generation"))
            state = self.get(self.po4ao, "collection_status", "state")
            if generation > old and state == "interrupted":
                raise RuntimeError(
                    "Warmup collection interrupted; learning remains paused"
                )
            return (
                generation > old
                and state == "ready"
                and generation
                == int(
                    self.get(self.po4ao, "collection_status", "requested_generation")
                )
            )

        self.wait(complete, cancel, "warmup snapshot", timeout)
