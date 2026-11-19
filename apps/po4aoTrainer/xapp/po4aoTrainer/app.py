## \file app.py
# \brief Standard XDevice entry point for the optional PO4AO training companion.
"""Only the numerical child imports torch or opens training replay streams."""

import math
import threading
import xconf
from magaox.indi.device import XDevice, BaseConfig
from purepyindi2 import properties, constants
from purepyindi2.messages import DefText, DefSwitch, DefNumber
from .switching import Runtime
from .supervisor import Supervisor
from .configuration import PARAMETERS


@xconf.config
class TrainerConfig(BaseConfig):
    """Shared device mappings, installed configuration paths and bounded operation deadlines."""

    po4ao_device: str = xconf.field(default="po4ao", help="C++ runtime INDI identity")
    training_config: str = xconf.field(
        default="/opt/MagAOX/config/po4ao_training.conf",
        help="Hot TOML training settings",
    )
    runtime_config: str = xconf.field(
        default="/opt/MagAOX/config/po4ao.conf",
        help="C++ configuration supplying runtime contract",
    )
    artifact_root: str = xconf.field(
        help="Provisioned shared artifact directory; C++ reads current/policy.trt beneath it"
    )
    saved_engines: str = xconf.field(
        default="",
        help="Shared saved model library; empty uses artifact_root/saved_engines",
    )
    source_revision: str = xconf.field(
        default="unrecorded",
        help="Installed trainer branch@commit for experiment provenance",
    )
    trtexec: str = xconf.field(
        default="/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec",
        help="Installed engine builder",
    )
    freshness_timeout: float = xconf.field(
        default=5.0, help="Maximum age of current-lifetime INDI observations"
    )
    operation_timeout: float = xconf.field(
        default=30.0, help="PO4AO runtime acknowledgement deadline"
    )
    worker_timeout: float = xconf.field(
        default=120.0, help="Worker start/pause deadline"
    )
    collection_timeout: float = xconf.field(
        default=120.0, help="Complete warmup publication deadline"
    )
    build_timeout: float = xconf.field(
        default=300.0, help="Maximum candidate export time"
    )
    heartbeat_timeout: float = xconf.field(
        default=30.0, help="Worker stops if supervisor is absent this long"
    )


class po4aoTrainer(XDevice):
    """Serve the native GUI contract using existing XDevice and PurePyINDI2 conventions."""

    config: TrainerConfig

    def setup(self):
        """Register capabilities and subscriptions without starting training or switching controllers."""
        self._property_lock = threading.RLock()
        for name in (
            "operation_timeout",
            "freshness_timeout",
            "worker_timeout",
            "collection_timeout",
            "build_timeout",
            "heartbeat_timeout",
        ):
            value = getattr(self.config, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        runtime = Runtime(
            self.client,
            self.config.po4ao_device,
            self.config.operation_timeout,
            freshness_timeout=self.config.freshness_timeout,
        )
        self.supervisor = Supervisor(runtime, self.config)
        self._heartbeat = 0
        self.log.info("PO4AO trainer source revision: %s", self.config.source_revision)
        self.client.get_properties(runtime.po4ao)
        self.text("fsm", {"state": "READY"})
        self.text("devices", {"po4ao": runtime.po4ao})
        self.text("status", self.supervisor.status)
        self.text(
            "training_config",
            {
                "path": self.config.training_config,
                "desired": "",
                "applied": "",
                "pending_reset": "",
                "error": "",
            },
        )
        for name, elements in {
            "po4ao_enable": ["toggle"],
            "reset": ["request"],
            "recollect": ["request"],
            "learning": ["run", "pause"],
        }.items():
            prop = properties.SwitchVector(
                name=name,
                perm=constants.PropertyPerm.READ_WRITE,
                rule=constants.SwitchRule.AT_MOST_ONE,
                state=constants.PropertyState.IDLE,
            )
            for element in elements:
                prop.add_element(
                    DefSwitch(name=element, _value=constants.SwitchState.OFF)
                )
            self.add_property(prop, callback=self.handle_request)
        for name in PARAMETERS:
            prop = properties.NumberVector(
                name=name,
                perm=constants.PropertyPerm.READ_WRITE,
                state=constants.PropertyState.IDLE,
            )
            for element in ("current", "target"):
                prop.add_element(
                    DefNumber(
                        name=element, _value=0, min=0, max=1e30, step=0, format="%.9g"
                    )
                )
            self.add_property(prop, callback=self.handle_request)
        self.text("set_policy", {"id": ""}, writable=True)
        self.text("save_policy", {"label": ""}, writable=True)
        self._catalog = None
        self.publish_parameters()
        self.publish_catalog()
        progress = properties.NumberVector(
            name="progress",
            perm=constants.PropertyPerm.READ_ONLY,
            state=constants.PropertyState.OK,
        )
        progress.add_element(
            DefNumber(name="iteration", _value=0, min=0, max=1e12, step=1, format="%g")
        )
        self.add_property(progress)

    def text(self, name, values, writable=False):
        """Create a normal text vector with the GUI's fixed element names."""
        prop = properties.TextVector(
            name=name,
            perm=(
                constants.PropertyPerm.READ_WRITE
                if writable
                else constants.PropertyPerm.READ_ONLY
            ),
            state=constants.PropertyState.OK,
        )
        for key, value in values.items():
            prop.add_element(DefText(name=key, _value=value))
        self.add_property(prop, callback=self.handle_request if writable else None)

    def handle_request(self, prop, message):
        """Acknowledge admission immediately and leave bounded operations to the supervisor."""
        self.log.info("Received %s request: %s", prop.name, dict(message))
        with self._property_lock:
            try:
                values = {key: message[key] for key in message}
                if prop.name in PARAMETERS:
                    if set(values) != {"target"}:
                        raise ValueError("Set only the target element")
                    self.supervisor.set_parameter(prop.name, values["target"])
                    prop["target"] = float(values["target"])
                    prop.state = constants.PropertyState.BUSY
                    self.update_property(prop)
                    self.publish_parameters()
                    return
                if prop.name in ("set_policy", "save_policy"):
                    element = "id" if prop.name == "set_policy" else "label"
                    if set(values) != {element}:
                        raise ValueError("Request elements do not match " + prop.name)
                    self.supervisor.submit(prop.name, values)
                    prop[element] = values[element]
                else:
                    required = {
                        "po4ao_enable": {"toggle"},
                        "reset": {"request"},
                        "recollect": {"request"},
                        "learning": {"run", "pause"},
                    }[prop.name]
                    if set(values) != required:
                        raise ValueError("Request elements do not match " + prop.name)
                    if any(
                        value
                        not in (constants.SwitchState.ON, constants.SwitchState.OFF)
                        for value in values.values()
                    ):
                        raise ValueError("Expected explicit On/Off switches")
                    values = {
                        key: value == constants.SwitchState.ON
                        for key, value in values.items()
                    }
                    if prop.name in ("reset", "recollect") and not values["request"]:
                        return
                    if prop.name == "learning" and values["run"] == values["pause"]:
                        raise ValueError("Select exactly one learning action")
                    self.supervisor.submit(prop.name, values)
                prop.state = constants.PropertyState.BUSY
                self.log.info("Admitted %s request", prop.name)
            except Exception as error:
                prop.state = constants.PropertyState.ALERT
                if prop.name in PARAMETERS:
                    self.supervisor.parameter_errors[prop.name] = str(error)
                self.supervisor.status["error"] = str(error)
                self.log.error("Rejected %s request: %s", prop.name, error)
            self.update_property(prop)

    def publish_parameters(self):
        """Publish desired and acknowledged values without pretending paused edits have applied."""
        for name, values in self.supervisor.parameters().items():
            prop = self.properties[name]
            prop["current"], prop["target"] = values["current"], values["target"]
            prop.state = constants.PropertyState(values["state"])
            self.update_property(prop)

    def publish_catalog(self):
        """Redefine the text catalog only when completed immutable entries change."""
        catalog = {
            identity: f"{entry['label']} | {entry['created_utc'][:16].replace('T', ' ')} UTC | iteration {entry['iteration']}"
            for identity, entry in self.supervisor.catalog.items()
        }
        # INDI clients also need an explicit empty definition to distinguish no entries from unavailable.
        if not catalog:
            catalog = {"none": "No saved policies"}
        if catalog != self._catalog:
            if self._catalog is not None:
                self.delete_property(self.properties["policy_catalog"])
            self.text("policy_catalog", catalog)
            self._catalog = catalog

    def loop(self):
        """Publish effective readback before terminal operation acknowledgements."""
        self.supervisor.runtime.observations.refresh()
        events = self.supervisor.tick()
        self._heartbeat += 1
        self.supervisor.status["heartbeat"] = str(self._heartbeat)
        from .configuration import revision

        with self._property_lock:
            self.publish_parameters()
            self.publish_catalog()
            status = self.properties["status"]
            for key, value in self.supervisor.status.items():
                status[key] = value
            self.update_property(status)
            config = self.properties["training_config"]
            for key, value in dict(
                desired=(
                    revision(self.supervisor.desired) if self.supervisor.desired else ""
                ),
                applied=(
                    self.supervisor.worker.revision if self.supervisor.worker else ""
                ),
                pending_reset=self.supervisor.pending_reset,
                error=self.supervisor.config_error,
            ).items():
                config[key] = value
            self.update_property(config)
            worker = self.supervisor.worker
            self.properties["progress"]["iteration"] = worker.iteration if worker else 0
            self.update_property(self.properties["progress"])
            for command, state, error in events:
                self.log.info(
                    "Completed %s: %s%s", command, state, ": " + error if error else ""
                )
                prop = self.properties[command]
                prop.state = constants.PropertyState(state)
                self.update_property(prop)

    def teardown(self):
        """Leave real-time controller selection alone while stopping owned numerical work."""
        if hasattr(self, "supervisor"):
            self.supervisor.close()
            self.supervisor.runtime.close()


main = po4aoTrainer.console_app
