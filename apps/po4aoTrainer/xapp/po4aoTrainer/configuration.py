## \file configuration.py
# \brief Typed training configuration and atomic iteration-boundary revisions.
"""Runtime dimensions, gains, noise and limits remain owned by the C++ configuration."""

from copy import deepcopy
from dataclasses import asdict
from configparser import ConfigParser
from pathlib import Path
from typing import Union
import hashlib
import json
import math
import tomllib
import xconf


@xconf.config
class MdpConfig:
    """MDP training settings; all values are explicit in the TOML file."""

    n_history: int = xconf.field(help="n history")
    control_delay: int = xconf.field(help="control delay")
    planning_horizon: int = xconf.field(help="planning horizon")


@xconf.config
class RlConfig:
    """RL training settings; all values are explicit in the TOML file."""

    loss_function_penalty: float = xconf.field(help="loss function penalty")


@xconf.config
class NnModelsConfig:
    """NN_models training settings; all values are explicit in the TOML file."""

    training_batch: int = xconf.field(help="training batch")
    policy_architecture: str = xconf.field(help="policy architecture")


@xconf.config
class NonlinearConfig:
    """nonlinear training settings; all values are explicit in the TOML file."""

    architecture: str = xconf.field(help="architecture")
    channels: int = xconf.field(help="channels")
    shared_layers: int = xconf.field(help="shared layers")
    negative_slope: float = xconf.field(help="negative slope")
    low_order_modes: int = xconf.field(help="low order modes")
    global_hidden: int = xconf.field(help="global hidden")
    global_features: int = xconf.field(help="global features")


@xconf.config
class TrainingLoopConfig:
    """Optimization, validation and publication settings for the training loop."""

    dynamics_pretrain_steps: int = xconf.field(help="dynamics pretrain steps")
    dynamics_grad_steps: int = xconf.field(help="dynamics grad steps")
    policy_grad_steps: int = xconf.field(help="policy grad steps")
    dynamics_learning_rate: float = xconf.field(help="dynamics learning rate")
    policy_learning_rate: float = xconf.field(help="policy learning rate")
    dynamics_gradient_clip_norm: float = xconf.field(help="dynamics gradient clip norm")
    policy_gradient_clip_norm: float = xconf.field(help="policy gradient clip norm")
    validation_episodes: int = xconf.field(help="validation episodes")
    diagnostics_interval: int = xconf.field(help="diagnostics interval")
    model_publish_interval: int = xconf.field(help="model publish interval")
    first_model_publish_iteration: int = xconf.field(
        help="first model publish iteration"
    )
    max_policy_clip_fraction: float = xconf.field(help="max policy clip fraction")
    max_candidate_action_rms_ratio: float = xconf.field(
        help="max candidate action rms ratio"
    )
    training_iterations: int = xconf.field(help="training iterations")


@xconf.config
class NormalizationConfig:
    """normalization training settings; all values are explicit in the TOML file."""

    rms_floor_fraction: float = xconf.field(help="rms floor fraction")


@xconf.config
class ReplayBuffersConfig:
    """replay_buffers training settings; all values are explicit in the TOML file."""

    historical_seconds: int = xconf.field(help="historical seconds")
    historical_episodes: int = xconf.field(help="historical episodes")
    train_warmup_percent: float = xconf.field(help="train warmup percent")
    train_historical_percent: float = xconf.field(help="train historical percent")


@xconf.config
class PolicyUpdateConfig:
    """policy_update training settings; all values are explicit in the TOML file."""

    penalty: Union[float, list[float]] = xconf.field(
        help="Penalty on normalized action changes from the acknowledged policy"
    )
    anchor_samples: int = xconf.field(help="anchor samples")


@xconf.config
class DiagnosticsConfig:
    """diagnostics training settings; all values are explicit in the TOML file."""

    sample_rate_hz: int = xconf.field(help="sample rate hz")


@xconf.config
class ComputeConfig:
    """compute training settings; all values are explicit in the TOML file."""

    training_device: int = xconf.field(help="training device")
    build_device: int = xconf.field(help="build device")


@xconf.config
class TrainingConfig:
    """Complete supported training schema; unknown keys are rejected by xconf."""

    MDP: MdpConfig = xconf.field(help="MDP settings")
    RL: RlConfig = xconf.field(help="RL settings")
    NN_models: NnModelsConfig = xconf.field(help="NN_models settings")
    nonlinear: NonlinearConfig = xconf.field(help="nonlinear settings")
    training: TrainingLoopConfig = xconf.field(help="training settings")
    normalization: NormalizationConfig = xconf.field(help="normalization settings")
    replay_buffers: ReplayBuffersConfig = xconf.field(help="replay_buffers settings")
    policy_update: PolicyUpdateConfig = xconf.field(help="policy_update settings")
    diagnostics: DiagnosticsConfig = xconf.field(help="diagnostics settings")
    compute: ComputeConfig = xconf.field(help="compute settings")


# Hot fields use resolved notebook keys, after normalization of the TOML names.
HOT = {
    "RL": {"loss_function_penalty"},
    "NN_models": {"training_batch"},
    "training": {
        "dynamics_grad_steps",
        "policy_grad_steps",
        "dynamics_learning_rate",
        "policy_learning_rate",
        "dynamics_gradient_clip_norm",
        "policy_gradient_clip_norm",
        "diagnostics_interval",
        "model_publish_interval",
        "first_model_publish_iteration",
        "max_policy_clip_fraction",
        "max_candidate_action_rms_ratio",
        "training_iterations",
    },
    "replay_buffers": {"train_warmup_percent", "train_historical_percent"},
    "policy_update": {"weight", "anchor_samples"},
    "diagnostics": {"sample_rate_hz"},
}


# INDI property names map to resolved training keys, never to config-file writes.
PARAMETERS = {
    "command_regularization": ("RL", "loss_function_penalty"),
    "delta_policy_regularization": ("policy_update", "weight"),
    "policy_learning_rate": ("training", "policy_learning_rate"),
}


def parameter_value(config, name):
    """Return a scalar control value, or None for a nonuniform per-mode penalty."""
    group, key = PARAMETERS[name]
    value = config[group][key]
    if isinstance(value, list):
        return value[0] if value and all(x == value[0] for x in value) else None
    return value


def revision(config):
    """Hash the resolved numerical snapshot, identically to TrainingSession."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def needs_reset(previous, candidate):
    """Stage the entire revision if any changed field is not declared hot."""
    return any(
        previous.get(group, {}).get(key) != value and key not in HOT.get(group, ())
        for group, values in candidate.items()
        for key, value in values.items()
    )


def load_training(path, runtime_path):
    """Resolve TOML names to notebook keys and merge the C++ runtime contract.

    Accept the previous policy_update.weight spelling for deployed files, but
    reject both spellings together instead of silently choosing a value.
    """
    settings = tomllib.loads(Path(path).read_text())
    update = settings.get("policy_update", {})
    if "weight" in update:
        if "penalty" in update:
            raise ValueError("Use policy_update.penalty or legacy weight, not both")
        update["penalty"] = update.pop("weight")
    config = asdict(xconf.from_dict(TrainingConfig, settings))
    # Preserve the numerical notebook API and config identity for a spelling-only edit.
    config["policy_update"]["weight"] = config["policy_update"].pop("penalty")
    parser = ConfigParser(inline_comment_prefixes=("#", ";"))
    if not parser.read(runtime_path):
        raise FileNotFoundError(runtime_path)
    p = parser["parameters"]
    if p.getint("config_version") != 2:
        raise ValueError("Expected PO4AO runtime config_version=2")
    config["runtime"] = {
        "version": 2,
        "stream_n_modes": p.getint("Nmodes"),
        "n_modes_controlled": p.getint("Nmodes_controlled"),
    }
    config["MDP"]["history_capacity"] = p.getint("max_Nhist")
    config["RL"]["episode_length"] = p.getint("iterations_per_ep")
    config["replay_buffers"]["capacity"] = p.getint("replay_buffer_size")
    config["integrator"] = {"gain": p.getfloat("integrator_gain")}
    config["safety"] = {"max_command_abs": p.getfloat("max_command_abs")}
    config["compute"]["inference_device"] = p.getint("cuda_device")
    if (
        p.getint("warmup_episodes") * config["RL"]["episode_length"]
        != config["replay_buffers"]["capacity"]
    ):
        raise ValueError(
            "Shared session requires warmup_episodes * iterations_per_ep == replay_buffer_size"
        )
    return validate_training(config)


def validate_training(config):
    """Validate resolved settings from a file or a merged live parameter request."""
    for group, values in config.items():
        for key, value in values.items():
            if isinstance(value, (int, float)) and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{group}.{key} must be finite and nonnegative")
    positive = {
        "MDP": ("n_history", "planning_horizon"),
        "RL": ("episode_length",),
        "NN_models": ("training_batch",),
        "nonlinear": (
            "channels",
            "shared_layers",
            "global_hidden",
            "global_features",
            "low_order_modes",
        ),
        "normalization": ("rms_floor_fraction",),
        "policy_update": ("anchor_samples",),
        "replay_buffers": ("capacity", "historical_seconds", "historical_episodes"),
        "diagnostics": ("sample_rate_hz",),
        "safety": ("max_command_abs",),
    }
    positive["training"] = tuple(
        k
        for k in config["training"]
        if k not in ("first_model_publish_iteration", "max_policy_clip_fraction")
    )
    for group, keys in positive.items():
        if any(config[group][k] <= 0 for k in keys):
            raise ValueError(f"{group} requires positive sizes/rates/limits")
    m, r, t = config["MDP"], config["replay_buffers"], config["training"]
    ep = config["RL"]["episode_length"]
    if (
        not 0
        < config["runtime"]["n_modes_controlled"]
        <= config["runtime"]["stream_n_modes"]
    ):
        raise ValueError("Invalid runtime modal prefix")
    if (
        not 0 <= m["control_delay"] < m["n_history"] <= m["history_capacity"]
        or m["planning_horizon"] <= m["control_delay"]
    ):
        raise ValueError("Invalid policy history/delay/horizon")
    if (
        ep < m["n_history"] + m["planning_horizon"]
        or r["capacity"] % ep
        or r["capacity"] // ep <= t["validation_episodes"]
    ):
        raise ValueError(
            "Need complete training and validation episodes with sufficient history"
        )
    if r["train_warmup_percent"] + r["train_historical_percent"] >= 1:
        raise ValueError("Replay fractions must sum to less than one")
    if t["max_policy_clip_fraction"] > 1:
        raise ValueError("Clip fraction must not exceed one")
    nonlinear = config["nonlinear"]
    if nonlinear["channels"] < 2 or nonlinear["shared_layers"] < 2:
        raise ValueError("nonlinear channels and shared_layers must be at least two")
    if not 0 < nonlinear["negative_slope"] < 1:
        raise ValueError("negative_slope must be between zero and one")
    if not 0 < config["normalization"]["rms_floor_fraction"] <= 1:
        raise ValueError("rms_floor_fraction must be in (0, 1]")
    architecture = config["NN_models"]["policy_architecture"]
    if (
        architecture not in ("nonlinear_local", "nonlinear_global")
        or config["nonlinear"]["architecture"] != architecture
    ):
        raise ValueError("Choose matching supported nonlinear architectures")
    weights = config["policy_update"]["weight"]
    weights = weights if isinstance(weights, list) else [weights]
    if len(weights) not in (
        1,
        (
            config["runtime"]["n_modes_controlled"]
            if architecture == "nonlinear_local"
            else 1
        ),
    ) or any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError("Invalid policy update weights")
    c = config["compute"]
    if (
        c["training_device"] != c["build_device"]
        or c["inference_device"] == c["training_device"]
    ):
        raise ValueError("Training/build must share a GPU separate from inference")
    return config


def apply_hot(session, candidate):
    """Apply one validated snapshot at an iteration boundary, preserving Adam and normalization."""
    if needs_reset(session.config, candidate):
        raise ValueError("Configuration requires Reset")
    session._config = deepcopy(candidate)
    session.config_revision = revision(candidate)
    for name in ("dynamics", "policy"):
        for group in getattr(session, name + "_optimizer").param_groups:
            group["lr"] = candidate["training"][name + "_learning_rate"]
    if session.policy_update is not None:
        import torch

        session.policy_update.weight = torch.as_tensor(
            candidate["policy_update"]["weight"],
            dtype=torch.float32,
            device=session.device,
        ).reshape(-1)
        session.policy_update.anchors = session.policy_update.reference_actions = None
