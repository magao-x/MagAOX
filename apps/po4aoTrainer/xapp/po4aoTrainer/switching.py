## \file switching.py
# \brief Completion-aware PO4AO runtime control through the existing INDI interface.
"""NN/CACAO switching belongs exclusively to nnReconstructor."""

import time
from purepyindi2.constants import PropertyState, SwitchState
from .observations import Observations


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
            busy=False,
            terminal=False,
            outcome=None,
        )

        def acknowledge(message):
            """Remember Busy/terminal updates even when they occur between polls."""
            state = getattr(message, "state", None)
            if state == PropertyState.BUSY:
                ticket["busy"] = True
            elif ticket["busy"] and state in (PropertyState.OK, PropertyState.ALERT):
                ticket["terminal"] = True

        ticket["callback"] = acknowledge
        self.reload_ticket = ticket
        self.client.register_callback(acknowledge, self.po4ao, "reload_engine")
        if issued:
            issued(ticket)
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
