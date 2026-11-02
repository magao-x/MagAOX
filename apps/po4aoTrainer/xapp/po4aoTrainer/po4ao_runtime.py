## \file po4ao_runtime.py
# \brief Packaged runtime from the nonlinear notebook.
"""Safe replay snapshots, compute isolation, and guarded policy publication."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

LINEAR_POLICY_MAGIC = b"PO4AOLIN"
LINEAR_POLICY_VERSION = 1
CPU_LINEAR_ARTIFACT_MAGIC = b"PO4AOCPU"
SYSTEM_CPUSET_FILE = Path("/opt/MagAOX/cpuset/system/cpuset.cpus")


@dataclass(frozen=True)
class ReplaySnapshot:
    """One internally consistent replay-buffer snapshot."""

    data: np.ndarray
    counter: int
    valid_length: int
    episode_id: int
    model_id: str


def model_publication_feedback(
    *,
    iteration,
    first_iteration,
    last_publish_iteration,
    publish_interval,
    candidate_clip_fraction,
    max_clip_fraction,
    candidate_action_rms_ratio,
    max_action_rms_ratio,
    nonfinite_samples,
    policy_update_finite,
    pending_model_id,
    active_model_id,
    reference_model_id,
):
    """Explain every existing publication gate; only ``ready`` permits export.

    Timing and pending activation are waiting states. Safety failures and an
    unexpected active model are blocked states, reported before timing reasons.
    Keep failed non-finite comparisons blocked, just like the original checks.
    This runs only at notebook diagnostic cadence, never in real-time control.
    """
    waiting = []
    blocked = []
    if iteration < first_iteration:
        waiting.append(f"first publication: iteration {iteration}/{first_iteration}")
    if last_publish_iteration is not None:
        elapsed = iteration - last_publish_iteration
        if elapsed < publish_interval:
            waiting.append(f"{elapsed}/{publish_interval} iterations since last export")
    if not candidate_clip_fraction <= max_clip_fraction:
        blocked.append(
            f"candidate clipping {candidate_clip_fraction:.2%} > {max_clip_fraction:.2%}"
            if np.isfinite(candidate_clip_fraction)
            else "candidate clipping is non-finite"
        )
    if not candidate_action_rms_ratio <= max_action_rms_ratio:
        blocked.append(
            f"candidate RMS ratio {candidate_action_rms_ratio:.3f} > {max_action_rms_ratio:.3f}"
            if np.isfinite(candidate_action_rms_ratio)
            else "candidate RMS ratio is non-finite"
        )
    if nonfinite_samples != 0:
        blocked.append(f"replay contains {nonfinite_samples} non-finite samples")
    if not policy_update_finite:
        blocked.append("policy-update checks are non-finite")
    if pending_model_id is not None:
        waiting.append(
            f"C++ activation of model {pending_model_id}; active={active_model_id}"
        )
    if active_model_id != reference_model_id:
        blocked.append(
            f"active model {active_model_id} != reference {reference_model_id}"
        )
    return {
        "publication_status": (
            "blocked" if blocked else "waiting" if waiting else "ready"
        ),
        "publication_reasons": blocked + waiting,
    }


def select_controlled_modes(values, controlled_modes):
    """Return a contiguous view-copy containing only the controlled modal prefix."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Replay values must have shape [samples, modes]")
    if not 0 < controlled_modes <= values.shape[1]:
        raise ValueError("controlled_modes must fit within the replay mode dimension")
    return np.ascontiguousarray(values[:, :controlled_modes])


def modal_rms_scales(observations, actions, *, floor_fraction=0.01):
    """Return fixed per-mode RMS scales for raw observations and commands."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must have shape [samples, modes]")
    if observations.shape != actions.shape or not observations.size:
        raise ValueError("observations and actions must have the same non-empty shape")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("normalization data must be finite")
    if not 0 < floor_fraction <= 1:
        raise ValueError("floor_fraction must be in (0, 1]")

    def floored_rms(values, label):
        """Floor per-mode RMS by the median positive modal scale."""
        rms = np.sqrt(np.mean(np.square(values), axis=0))
        positive = rms[rms > 0]
        if not positive.size:
            raise ValueError(f"warm-up {label} contain no non-zero values")
        floor = floor_fraction * np.median(positive)
        return np.ascontiguousarray(np.maximum(rms, floor), dtype=np.float32)

    return floored_rms(observations, "observations"), floored_rms(actions, "actions")


def _parse_cpu_list(value):
    """Parse a Linux CPU-list string such as ``0-3,8,10-11``."""
    cpus = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", maxsplit=1))
            if first > last:
                raise ValueError(f"Invalid CPU range: {item}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def configure_compute_resources(inference_device, training_device, build_device):
    """Validate GPU isolation and keep notebook CPU work off real-time cores."""
    import torch

    devices = (inference_device, training_device, build_device)
    if any(not isinstance(device, int) or device < 0 for device in devices):
        raise ValueError("CUDA device indices must be non-negative integers")
    if training_device == inference_device or build_device == inference_device:
        raise ValueError("Training and building must not use the inference GPU")
    if build_device != training_device:
        raise ValueError("TensorRT building must use the training GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(devices):
        raise RuntimeError("Configured CUDA devices are not available")
    torch.cuda.set_device(training_device)

    inference_properties = torch.cuda.get_device_properties(inference_device)
    training_properties = torch.cuda.get_device_properties(training_device)
    inference_contract = (
        inference_properties.name,
        inference_properties.major,
        inference_properties.minor,
    )
    training_contract = (
        training_properties.name,
        training_properties.major,
        training_properties.minor,
    )
    if inference_contract != training_contract:
        raise RuntimeError(
            "Inference and training GPUs must have matching model and compute capability"
        )

    assigned_cpus = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        if SYSTEM_CPUSET_FILE.is_file():
            system_cpus = _parse_cpu_list(
                SYSTEM_CPUSET_FILE.read_text(encoding="utf-8")
            )
            assigned_cpus = system_cpus & set(os.sched_getaffinity(0))
            if not assigned_cpus:
                raise RuntimeError("No allowed CPUs overlap the MagAO-X system CPU set")
            task_directory = Path("/proc/self/task")
            thread_ids = (
                [int(path.name) for path in task_directory.iterdir()]
                if task_directory.is_dir()
                else [0]
            )
            for thread_id in thread_ids:
                try:
                    os.sched_setaffinity(thread_id, assigned_cpus)
                except ProcessLookupError:
                    pass

    return {
        "inference_gpu": inference_contract,
        "training_gpu": training_contract,
        "system_cpus": sorted(assigned_cpus) if assigned_cpus is not None else None,
    }


def _engine_id(path):
    """Return the same 64-bit FNV-1a model ID used by the C++ app."""
    value = 14695981039346656037
    with open(path, "rb") as engine:
        for chunk in iter(lambda: engine.read(1024 * 1024), b""):
            for byte in chunk:
                value ^= byte
                value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def stable_replay_snapshot(image, metadata_image, *, attempts=100, poll_seconds=0.001):
    """Copy a replay shmim and paired metadata from one stable publication."""
    return stable_replay_prefix_snapshot(
        image,
        metadata_image,
        modes=None,
        attempts=attempts,
        poll_seconds=poll_seconds,
    )


def stable_replay_prefix_snapshot(
    image,
    metadata_image,
    *,
    modes,
    attempts=100,
    poll_seconds=0.001,
):
    """Copy only a leading modal prefix from one stable replay publication."""
    for _ in range(attempts):
        if bool(image.md.write) or bool(metadata_image.md.write):
            time.sleep(poll_seconds)
            continue

        counter_before = int(image.md.cnt0)
        metadata_counter_before = int(metadata_image.md.cnt0)
        metadata = np.array(metadata_image, dtype=np.uint32, copy=True).reshape(-1)
        image_view = np.asarray(image, dtype=np.float32)
        if image_view.ndim != 3 or image_view.shape[2] != 2:
            raise RuntimeError(
                f"Replay data must have shape [modes, samples, 2], got {image_view.shape}"
            )
        mode_count = image_view.shape[0] if modes is None else int(modes)
        if not 0 < mode_count <= image_view.shape[0]:
            raise ValueError("modes must select a non-empty replay modal prefix")
        if metadata.size != 4:
            raise RuntimeError(
                f"Replay metadata must contain four uint32 values, got {metadata.size}"
            )
        valid_length = int(metadata[0])
        capacity = image_view.shape[1]
        if not 0 <= valid_length <= capacity:
            raise RuntimeError(
                f"Invalid replay valid length {valid_length}; capacity is {capacity}"
            )
        data = np.array(
            image_view[:mode_count, :valid_length, :], dtype=np.float32, copy=True
        )
        counter_after = int(image.md.cnt0)
        metadata_counter_after = int(metadata_image.md.cnt0)
        if (
            not bool(image.md.write)
            and not bool(metadata_image.md.write)
            and counter_before == counter_after
            and metadata_counter_before == metadata_counter_after
            and counter_after == metadata_counter_after
        ):
            episode_id = int(metadata[1])
            model_id_value = int(metadata[2]) | (int(metadata[3]) << 32)
            return ReplaySnapshot(
                data=data,
                counter=counter_after,
                valid_length=valid_length,
                episode_id=episode_id,
                model_id=f"{model_id_value:016x}",
            )
        time.sleep(poll_seconds)

    raise TimeoutError("Replay buffer stayed busy or changed during every copy attempt")


def save_training_snapshot(path, policy, dynamics, model_id, metadata=None):
    """Atomically save compact policy and dynamics state for later analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    candidate = path.with_name(f".{path.name}.{token}")
    snapshot_metadata = dict(metadata or {})
    snapshot_metadata.update(
        {
            "model_id": model_id,
            "saved_unix_ns": time.time_ns(),
        }
    )
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(snapshot_metadata, allow_nan=False, sort_keys=True)
        )
    }
    if hasattr(policy, "raw_linear_weights"):
        arrays["policy_raw_weights"] = (
            policy.raw_linear_weights().detach().cpu().numpy()
        )
    else:
        for name, value in policy.state_dict().items():
            arrays[f"policy_state__{name}"] = value.detach().cpu().numpy()
    for index, model in enumerate(dynamics.models):
        if hasattr(model, "raw_linear_weights"):
            arrays[f"dynamics_{index}_raw_weights"] = (
                model.raw_linear_weights().detach().cpu().numpy()
            )
        else:
            for name, value in model.state_dict().items():
                arrays[f"dynamics_{index}_state__{name}"] = value.detach().cpu().numpy()
    try:
        with candidate.open("wb") as output:
            np.savez_compressed(output, **arrays)
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)


def _trtexec_command(trtexec, onnx_path, candidate_path, timing_cache, build_device):
    """Return the TensorRT build command with an explicit non-real-time GPU."""
    if not isinstance(build_device, int) or build_device < 0:
        raise ValueError("build_device must be a non-negative CUDA device index")
    return [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={candidate_path}",
        f"--timingCacheFile={timing_cache}",
        f"--device={build_device}",
        "--profilingVerbosity=none",
        "--skipInference",
        "--fp16",
    ]


def _write_linear_policy(path, weights, model_id):
    """Write a versioned CPU linear-policy sidecar tied to one engine hash."""
    weights = np.asarray(weights, dtype="<f4")
    if weights.ndim != 2 or not weights.size or not np.isfinite(weights).all():
        raise ValueError(
            "Linear-policy weights must be a finite feature-by-mode matrix"
        )
    features, modes = weights.shape
    header = struct.pack(
        "<8sIIIQ",
        LINEAR_POLICY_MAGIC,
        LINEAR_POLICY_VERSION,
        features,
        modes,
        int(model_id, 16),
    )
    with open(path, "wb") as output:
        output.write(header)
        output.write(np.ascontiguousarray(weights).tobytes())


def _write_cpu_linear_artifact(path, weights):
    """Write the hashable primary artifact for a CPU-only linear policy."""
    weights = np.asarray(weights, dtype="<f4")
    if weights.ndim != 2 or not weights.size or not np.isfinite(weights).all():
        raise ValueError(
            "Linear-policy weights must be a finite feature-by-mode matrix"
        )
    features, modes = weights.shape
    header = struct.pack(
        "<8sIII",
        CPU_LINEAR_ARTIFACT_MAGIC,
        LINEAR_POLICY_VERSION,
        features,
        modes,
    )
    with open(path, "wb") as output:
        output.write(header)
        output.write(np.ascontiguousarray(weights).tobytes())


def publish_tensorrt_engine(
    policy,
    example_input,
    engine_path,
    *,
    build_device,
    trtexec="/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec",
    timing_cache="timing.cache",
    log_path="trt_outputs",
    metadata=None,
    dynamics=None,
):
    """Atomically publish a CPU-linear artifact or TensorRT engine."""
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    onnx_path = engine_path.with_name(f".{engine_path.name}.{token}.onnx")
    candidate_path = engine_path.with_name(f".{engine_path.name}.{token}.candidate")
    linear_path = engine_path.with_suffix(engine_path.suffix + ".linear")
    linear_candidate = linear_path.with_name(f".{linear_path.name}.{token}")

    try:
        model_metadata = dict(metadata or {})
        linear_policy = model_metadata.get("policy_architecture") == "linear_local"
        if linear_policy:
            if getattr(policy.output, "bias", None) is not None:
                raise ValueError("CPU linear inference requires a bias-free policy")
            try:
                weights = policy.raw_linear_weights().detach().cpu().numpy()
            except (AttributeError, IndexError) as error:
                raise ValueError(
                    "linear_local policy does not expose the expected output weights"
                ) from error
            _write_cpu_linear_artifact(candidate_path, weights)
            model_id = _engine_id(candidate_path)
            _write_linear_policy(linear_candidate, weights, model_id)
            os.replace(linear_candidate, linear_path)
        else:
            import torch

            torch.onnx.export(policy, example_input, onnx_path, input_names=["input"])
            command = _trtexec_command(
                trtexec, onnx_path, candidate_path, timing_cache, build_device
            )
            with open(log_path, "w", encoding="utf-8") as output:
                subprocess.run(
                    command, check=True, stdout=output, stderr=subprocess.STDOUT
                )
            if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
                raise RuntimeError(
                    "TensorRT reported success without creating an engine"
                )
            model_id = _engine_id(candidate_path)
            linear_path.unlink(missing_ok=True)
        os.replace(candidate_path, engine_path)
        model_metadata.update(
            {
                "active_model_id": model_id,
                "artifact_type": "cpu_linear" if linear_policy else "tensorrt",
                "engine": str(engine_path),
                "published_unix_ns": time.time_ns(),
            }
        )
        if not linear_policy:
            model_metadata["engine_build_cuda_device"] = build_device
        if dynamics is not None:
            save_training_snapshot(
                engine_path.with_suffix(engine_path.suffix + ".training.npz"),
                policy,
                dynamics,
                model_id,
                model_metadata,
            )
        metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
        temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{token}")
        temporary_metadata.write_text(
            json.dumps(model_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
        return model_id
    finally:
        onnx_path.unlink(missing_ok=True)
        candidate_path.unlink(missing_ok=True)
        linear_candidate.unlink(missing_ok=True)
