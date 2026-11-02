## \file po4ao_diagnostics.py
# \brief Packaged diagnostics from the nonlinear notebook.
"""Lightweight training and replay diagnostics for modal PO4AO."""

import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import welch

_PLOT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _representative_modes(n_modes):
    """Return up to three useful modal indices without assuming 101 modes exist."""
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")
    preferred = [mode for mode in (0, 10, 100) if mode < n_modes]
    for mode in (n_modes // 2, n_modes - 1):
        if mode not in preferred:
            preferred.append(mode)
        if len(preferred) == 3:
            break
    return preferred[:3]


def read_runtime_telemetry(client):
    """Read the low-rate PO4AO INDI timing and failure telemetry."""
    fields = (
        "p50_us",
        "p95_us",
        "p99_us",
        "max_us",
        "missed_frames",
        "deadline_misses",
        "inference_failures",
        "replay_drops",
    )
    telemetry = {
        f"runtime_{field}": float(client[f"po4ao.runtime_stats.{field}"])
        for field in fields
    }
    telemetry["runtime_active_model_id"] = str(client["po4ao.model_status.active_id"])
    return telemetry


def replay_metrics(
    observations,
    actions,
    command_limit,
    *,
    delay_max_lag=16,
    baseline_gain=None,
    assumed_control_delay=None,
):
    """Summarize replay health, command saturation, and apparent delay."""
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if observations.shape != actions.shape or observations.ndim != 2:
        raise ValueError(
            "observations and actions must be matching (time, modes) arrays"
        )
    if command_limit <= 0:
        raise ValueError("command_limit must be positive")

    finite_observations = np.isfinite(observations)
    finite_actions = np.isfinite(actions)
    finite_observation_values = observations[finite_observations]
    finite_action_values = actions[finite_actions]
    if assumed_control_delay is not None:
        if (
            int(assumed_control_delay) != assumed_control_delay
            or assumed_control_delay < 0
        ):
            raise ValueError("assumed_control_delay must be a non-negative integer")
        delay = int(assumed_control_delay)
        response_lag = None
        delay_scores = np.empty(0)
        delay_source = "assumed"
    else:
        delay_actions = actions
        if baseline_gain is not None:
            if not np.isfinite(baseline_gain):
                raise ValueError("baseline_gain must be finite")
            delay_actions = actions - baseline_gain * observations
        response_lag, delay_scores = estimate_delay_frames(
            observations, delay_actions, max_lag=delay_max_lag
        )
        delay = max(0, response_lag - 1) if baseline_gain is not None else response_lag
        delay_source = "estimated"
    return {
        "observation_rms": (
            float(np.sqrt(np.mean(np.square(finite_observation_values))))
            if finite_observation_values.size
            else 0.0
        ),
        "action_rms": (
            float(np.sqrt(np.mean(np.square(finite_action_values))))
            if finite_action_values.size
            else 0.0
        ),
        "action_abs_max": (
            float(np.max(np.abs(finite_action_values)))
            if finite_action_values.size
            else float("nan")
        ),
        "command_clip_fraction": (
            float(np.mean(np.abs(finite_action_values) >= 0.99 * command_limit))
            if finite_action_values.size
            else float("nan")
        ),
        "nonfinite_samples": int(
            finite_observations.size
            + finite_actions.size
            - finite_observations.sum()
            - finite_actions.sum()
        ),
        "estimated_delay_frames": delay,
        "estimated_response_lag_frames": response_lag,
        "delay_scores": delay_scores.tolist(),
        "delay_source": delay_source,
    }


def estimate_delay_frames(observations, actions, *, max_lag=16, mode_indices=None):
    """Estimate the action-to-observation lag from normalized correlation."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if observations.shape != actions.shape or observations.ndim != 2:
        raise ValueError(
            "observations and actions must be matching (time, modes) arrays"
        )
    if max_lag < 0 or observations.shape[0] <= max_lag:
        raise ValueError("max_lag must be shorter than the time axis")

    if mode_indices is None:
        mode_indices = np.unique(
            [0, observations.shape[1] // 2, observations.shape[1] - 1]
        )
    selected_observations = np.nan_to_num(
        observations[:, mode_indices], nan=0.0, posinf=0.0, neginf=0.0
    )
    selected_actions = np.nan_to_num(
        actions[:, mode_indices], nan=0.0, posinf=0.0, neginf=0.0
    )
    scores = []
    for lag in range(max_lag + 1):
        command = selected_actions[: selected_actions.shape[0] - lag or None]
        measurement = selected_observations[lag:]
        command = command - command.mean(axis=0, keepdims=True)
        measurement = measurement - measurement.mean(axis=0, keepdims=True)
        denominator = np.sqrt(
            np.sum(command * command, axis=0)
            * np.sum(measurement * measurement, axis=0)
        )
        correlation = np.divide(
            np.sum(command * measurement, axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        scores.append(float(np.mean(np.abs(correlation))))
    return int(np.argmax(scores)), np.asarray(scores)


def lag_correlation(reference, response, *, max_lag=16, mode_indices=None):
    """Return signed per-mode correlations of reference[t] with response[t + lag]."""
    reference = np.asarray(reference, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    if reference.shape != response.shape or reference.ndim != 2:
        raise ValueError("reference and response must be matching (time, modes) arrays")
    if max_lag < 0 or reference.shape[0] <= max_lag:
        raise ValueError("max_lag must be shorter than the time axis")

    if mode_indices is None:
        mode_indices = np.arange(reference.shape[1])
    mode_indices = np.asarray(mode_indices, dtype=int)
    if mode_indices.ndim != 1 or mode_indices.size == 0:
        raise ValueError("mode_indices must contain at least one mode")
    if np.any(mode_indices < 0) or np.any(mode_indices >= reference.shape[1]):
        raise ValueError("mode_indices must fit within the mode dimension")

    correlations = np.zeros((max_lag + 1, mode_indices.size), dtype=np.float64)
    selected_reference = reference[:, mode_indices]
    selected_response = response[:, mode_indices]
    for lag in range(max_lag + 1):
        left = selected_reference[: selected_reference.shape[0] - lag or None]
        right = selected_response[lag:]
        valid = np.isfinite(left) & np.isfinite(right)
        counts = valid.sum(axis=0)
        safe_counts = np.maximum(counts, 1)
        left_mean = np.where(valid, left, 0.0).sum(axis=0) / safe_counts
        right_mean = np.where(valid, right, 0.0).sum(axis=0) / safe_counts
        x = np.where(valid, left - left_mean, 0.0)
        y = np.where(valid, right - right_mean, 0.0)
        denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
        correlations[lag] = np.divide(
            np.sum(x * y, axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=(counts >= 2) & (denominator > 0),
        )
    return correlations


def _longest_repeated_run(repeated):
    """Return the longest number of identical consecutive frames."""
    longest = current = 0
    for value in repeated:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest + 1 if longest else 1


def replay_buffer_health(
    observations,
    actions,
    *,
    baseline_gain=1.0,
    max_lag=16,
    mode_indices=None,
    repeat_atol=0.0,
):
    """Measure replay repetition, scale, autocorrelation, and probe-response delay."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if observations.shape != actions.shape or observations.ndim != 2:
        raise ValueError(
            "observations and actions must be matching (time, modes) arrays"
        )
    if observations.shape[0] < 2:
        raise ValueError("at least two replay samples are required")
    if repeat_atol < 0:
        raise ValueError("repeat_atol must be non-negative")

    if mode_indices is None:
        mode_indices = np.arange(observations.shape[1])
    mode_indices = np.asarray(mode_indices, dtype=int)
    selected_observations = observations[:, mode_indices]
    selected_actions = actions[:, mode_indices]
    probe = selected_actions - baseline_gain * selected_observations

    observation_repeats = np.all(
        np.isclose(
            selected_observations[1:],
            selected_observations[:-1],
            rtol=0.0,
            atol=repeat_atol,
            equal_nan=True,
        ),
        axis=1,
    )
    action_repeats = np.all(
        np.isclose(
            selected_actions[1:],
            selected_actions[:-1],
            rtol=0.0,
            atol=repeat_atol,
            equal_nan=True,
        ),
        axis=1,
    )
    observation_acf = lag_correlation(
        selected_observations, selected_observations, max_lag=max_lag
    )
    probe_acf = lag_correlation(probe, probe, max_lag=max_lag)
    probe_response = lag_correlation(probe, selected_observations, max_lag=max_lag)
    delay_scores = np.mean(np.abs(probe_response), axis=1)
    response_lag = int(np.argmax(delay_scores))

    finite_observations = selected_observations[np.isfinite(selected_observations)]
    finite_actions = selected_actions[np.isfinite(selected_actions)]
    finite_probe = probe[np.isfinite(probe)]
    return {
        "samples": int(observations.shape[0]),
        "modes_analyzed": mode_indices,
        "nonfinite_values": int(
            selected_observations.size
            + selected_actions.size
            - np.isfinite(selected_observations).sum()
            - np.isfinite(selected_actions).sum()
        ),
        "observation_rms": (
            float(np.sqrt(np.mean(np.square(finite_observations))))
            if finite_observations.size
            else float("nan")
        ),
        "action_rms": (
            float(np.sqrt(np.mean(np.square(finite_actions))))
            if finite_actions.size
            else float("nan")
        ),
        "probe_rms": (
            float(np.sqrt(np.mean(np.square(finite_probe))))
            if finite_probe.size
            else float("nan")
        ),
        "repeated_observation_fraction": float(np.mean(observation_repeats)),
        "repeated_action_fraction": float(np.mean(action_repeats)),
        "longest_repeated_observation_run": _longest_repeated_run(observation_repeats),
        "longest_repeated_action_run": _longest_repeated_run(action_repeats),
        "response_lag_frames": response_lag,
        "control_delay_frames": max(0, response_lag - 1),
        "delay_scores": delay_scores,
        "observation_acf": observation_acf,
        "probe_acf": probe_acf,
        "probe_response_correlation": probe_response,
        "probe": probe,
    }


def plot_replay_buffer_health(observations, actions, health, *, sample_rate_hz):
    """Plot replay traces, repetition, correlations, and per-mode signal scale."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401 Registers optional plot styles.
    except ImportError:
        scienceplots = None
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    modes = np.asarray(health["modes_analyzed"], dtype=int)
    selected_observations = observations[:, modes]
    selected_actions = actions[:, modes]
    probe = np.asarray(health["probe"])
    display_columns = np.unique([0, min(10, modes.size - 1), min(100, modes.size - 1)])
    samples = min(1000, observations.shape[0])
    time_ms = np.arange(samples) * 1000.0 / sample_rate_hz
    lags = np.arange(health["delay_scores"].size)

    if scienceplots is not None:
        plt.style.use(["science", "no-latex", "grid"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    for column in display_columns:
        mode = modes[column]
        axes[0, 0].plot(
            time_ms,
            selected_observations[-samples:, column],
            label=f"measurement {mode}",
        )
        axes[0, 0].plot(
            time_ms,
            selected_actions[-samples:, column],
            linestyle="--",
            alpha=0.7,
            label=f"command {mode}",
        )
    axes[0, 0].set(
        title="Recent replay values", xlabel="Time [ms]", ylabel="Modal value"
    )
    axes[0, 0].legend(fontsize="small", ncol=2)

    observation_step = np.linalg.norm(np.diff(selected_observations, axis=0), axis=1)
    action_step = np.linalg.norm(np.diff(selected_actions, axis=0), axis=1)
    axes[0, 1].semilogy(
        observation_step + np.finfo(float).tiny, label="measurement change"
    )
    axes[0, 1].semilogy(action_step + np.finfo(float).tiny, label="command change")
    axes[0, 1].set(
        title="Consecutive-frame change", xlabel="Replay index", ylabel="L2 difference"
    )
    axes[0, 1].legend()

    axes[0, 2].plot(
        lags, np.mean(np.abs(health["observation_acf"]), axis=1), marker="o"
    )
    axes[0, 2].set(
        title="Measurement autocorrelation",
        xlabel="Lag [frames]",
        ylabel="Mean |correlation|",
    )

    axes[1, 0].plot(lags, np.mean(np.abs(health["probe_acf"]), axis=1), marker="o")
    axes[1, 0].set(
        title="Probe autocorrelation",
        xlabel="Lag [frames]",
        ylabel="Mean |correlation|",
    )

    axes[1, 1].plot(lags, health["delay_scores"], marker="o")
    axes[1, 1].axvline(health["response_lag_frames"], color="C1", linestyle="--")
    axes[1, 1].set(
        title="Probe to measurement response",
        xlabel="Response lag [frames]",
        ylabel="Mean |correlation|",
    )

    axes[1, 2].semilogy(
        modes,
        np.sqrt(np.nanmean(np.square(selected_observations), axis=0)),
        label="measurement RMS",
    )
    axes[1, 2].semilogy(
        modes,
        np.sqrt(np.nanmean(np.square(selected_actions), axis=0)),
        label="command RMS",
    )
    axes[1, 2].semilogy(
        modes,
        np.sqrt(np.nanmean(np.square(probe), axis=0)),
        label="probe RMS",
    )
    axes[1, 2].set(title="Per-mode scale", xlabel="Mode", ylabel="RMS")
    axes[1, 2].legend()

    figure.suptitle(
        f"Replay health | {health['samples']} samples | "
        f"response lag ~ {health['response_lag_frames']} frames | "
        f"assumed control delay ~ {health['control_delay_frames']} frames",
        fontweight="bold",
    )
    return figure


def evaluate_dynamics_rollout(
    dynamics,
    observations,
    actions,
    *,
    n_history,
    episode_length,
    horizon,
    batch_size,
    device,
):
    """Measure one- through multi-step dynamics error on the newest episode."""
    return evaluate_dynamics_rollout_details(
        dynamics,
        observations,
        actions,
        n_history=n_history,
        episode_length=episode_length,
        horizon=horizon,
        batch_size=batch_size,
        device=device,
    )["horizon_mse"]


def evaluate_dynamics_rollout_details(
    dynamics,
    observations,
    actions,
    *,
    n_history,
    episode_length,
    horizon,
    batch_size,
    device,
    mode_indices=None,
):
    """Return rollout errors plus one representative measured/predicted trace."""
    import torch

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if observations.shape != actions.shape or observations.ndim != 2:
        raise ValueError(
            "observations and actions must be matching (time, modes) arrays"
        )
    if horizon <= 0 or batch_size <= 0:
        raise ValueError("horizon and batch_size must be positive")
    if mode_indices is None:
        mode_indices = _representative_modes(observations.shape[1])
    mode_indices = np.asarray(mode_indices, dtype=int)
    if mode_indices.ndim != 1 or mode_indices.size == 0:
        raise ValueError("mode_indices must contain at least one mode")
    if np.any(mode_indices < 0) or np.any(mode_indices >= observations.shape[1]):
        raise ValueError("mode_indices must fit within the mode dimension")

    first = observations.shape[0] - episode_length + n_history - 1
    last = observations.shape[0] - horizon - 1
    if first > last:
        raise ValueError(
            "validation episode is too short for history and rollout horizon"
        )
    starts = np.unique(
        np.linspace(first, last, min(batch_size, last - first + 1), dtype=int)
    )

    past_observations = torch.as_tensor(
        np.stack([observations[t - n_history + 1 : t] for t in starts]),
        device=device,
    ).unsqueeze(2)
    past_actions = torch.as_tensor(
        np.stack([actions[t - n_history + 1 : t] for t in starts]),
        device=device,
    ).unsqueeze(2)
    state = (
        torch.as_tensor(observations[starts], device=device).unsqueeze(1).unsqueeze(1)
    )

    was_training = dynamics.training
    dynamics.eval()
    losses = []
    predictions = []
    targets = []
    sample_squared_error = torch.zeros(starts.size, device=device)
    modal_squared_error = torch.zeros(observations.shape[1], device=device)
    try:
        with torch.no_grad():
            for step in range(horizon):
                action = (
                    torch.as_tensor(actions[starts + step], device=device)
                    .unsqueeze(1)
                    .unsqueeze(1)
                )
                history = torch.cat([past_observations, past_actions], dim=1)
                prediction = dynamics(torch.cat([history, state, action], dim=1)).mean(
                    dim=1, keepdim=True
                )
                target = (
                    torch.as_tensor(observations[starts + step + 1], device=device)
                    .unsqueeze(1)
                    .unsqueeze(1)
                )
                residual = dynamics.normalize_observation(target - prediction)
                squared_error = residual.pow(2).mean(dim=(1, 2))
                losses.append(float(squared_error.mean().item()))
                sample_squared_error += squared_error.mean(dim=1)
                modal_squared_error += squared_error.sum(dim=0)
                predictions.append(prediction[:, 0, 0].detach().cpu().numpy())
                targets.append(target[:, 0, 0].detach().cpu().numpy())
                past_observations = torch.cat([past_observations[:, 1:], state], dim=1)
                past_actions = torch.cat([past_actions[:, 1:], action], dim=1)
                state = prediction
    finally:
        dynamics.train(was_training)

    representative = int(torch.argsort(sample_squared_error)[starts.size // 2].item())
    initial = observations[starts[representative]]
    observed_trace = np.vstack(
        [initial, *[target[representative] for target in targets]]
    )[:, mode_indices]
    predicted_trace = np.vstack(
        [initial, *[prediction[representative] for prediction in predictions]]
    )[:, mode_indices]
    observation_scale = (
        dynamics.models[0]
        .normalizer.observation_scale.detach()
        .cpu()
        .numpy()
        .reshape(-1)[mode_indices]
    )
    return {
        "horizon_mse": losses,
        "per_mode_nrmse": torch.sqrt(modal_squared_error / (starts.size * horizon))
        .detach()
        .cpu()
        .numpy()
        .tolist(),
        "trace_modes": mode_indices.tolist(),
        "trace_observed_normalized": (observed_trace / observation_scale).tolist(),
        "trace_predicted_normalized": (predicted_trace / observation_scale).tolist(),
        "trace_start_index": int(starts[representative]),
    }


def measured_replay_cost(
    observations,
    actions,
    *,
    observation_scale,
    action_scale,
    loss_penalty,
    episode_length,
):
    """Measure the training objective per frame on the newest replay episode."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    observation_scale = np.asarray(observation_scale, dtype=np.float64).reshape(-1)
    action_scale = np.asarray(action_scale, dtype=np.float64).reshape(-1)
    if observations.shape != actions.shape or observations.ndim != 2:
        raise ValueError(
            "observations and actions must be matching (time, modes) arrays"
        )
    if observation_scale.size != observations.shape[1]:
        raise ValueError("observation_scale must contain one value per mode")
    if action_scale.size != actions.shape[1]:
        raise ValueError("action_scale must contain one value per mode")
    if np.any(observation_scale <= 0) or np.any(action_scale <= 0):
        raise ValueError("normalization scales must be positive")
    if not 0 < episode_length <= observations.shape[0]:
        raise ValueError("episode_length must fit within the replay time axis")
    if loss_penalty < 0 or not np.isfinite(loss_penalty):
        raise ValueError("loss_penalty must be finite and non-negative")

    normalized_observations = observations[-episode_length:] / observation_scale
    normalized_actions = actions[-episode_length:] / action_scale
    observation_cost = float(np.mean(np.square(normalized_observations)))
    action_cost = float(loss_penalty * np.mean(np.square(normalized_actions)))
    return {
        "total": observation_cost + action_cost,
        "observation": observation_cost,
        "action_penalty": action_cost,
    }


def modal_scale_ratios(
    observations,
    actions,
    *,
    observation_scale,
    action_scale,
    episode_length,
):
    """Compare recent per-mode RMS values with the fixed warm-up scales."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    observation_scale = np.asarray(observation_scale, dtype=np.float64).reshape(-1)
    action_scale = np.asarray(action_scale, dtype=np.float64).reshape(-1)
    if observations.shape != actions.shape or observations.ndim != 2:
        raise ValueError(
            "observations and actions must be matching (time, modes) arrays"
        )
    if observation_scale.size != observations.shape[1]:
        raise ValueError("observation_scale must contain one value per mode")
    if action_scale.size != actions.shape[1]:
        raise ValueError("action_scale must contain one value per mode")
    if np.any(observation_scale <= 0) or np.any(action_scale <= 0):
        raise ValueError("normalization scales must be positive")
    if not 0 < episode_length <= observations.shape[0]:
        raise ValueError("episode_length must fit within the replay time axis")

    recent_observations = observations[-episode_length:]
    recent_actions = actions[-episode_length:]
    return {
        "observation": (
            np.sqrt(np.mean(np.square(recent_observations), axis=0)) / observation_scale
        ),
        "action": (np.sqrt(np.mean(np.square(recent_actions), axis=0)) / action_scale),
    }


def evaluate_policy_actions(
    policy,
    observations,
    actions,
    *,
    n_history,
    episode_length,
    batch_size,
    command_limit,
    device,
):
    """Measure candidate-policy command scale on states from the newest episode."""
    import torch

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    first = observations.shape[0] - episode_length + n_history - 1
    last = observations.shape[0] - 1
    if first > last:
        raise ValueError("validation episode is too short for the requested history")
    starts = np.unique(
        np.linspace(first, last, min(batch_size, last - first + 1), dtype=int)
    )
    past_observations = torch.as_tensor(
        np.stack([observations[t - n_history + 1 : t] for t in starts]),
        device=device,
    ).unsqueeze(2)
    past_actions = torch.as_tensor(
        np.stack([actions[t - n_history + 1 : t] for t in starts]),
        device=device,
    ).unsqueeze(2)
    state = (
        torch.as_tensor(observations[starts], device=device).unsqueeze(1).unsqueeze(1)
    )

    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            policy_actions = (
                policy(torch.cat([state, past_observations, past_actions], dim=1))
                .detach()
                .cpu()
                .numpy()
            )
    finally:
        policy.train(was_training)
    absolute = np.abs(policy_actions)
    return {
        "candidate_action_rms": float(np.sqrt(np.mean(np.square(policy_actions)))),
        "candidate_action_abs_max": float(np.max(absolute)),
        "candidate_clip_fraction": float(np.mean(absolute >= 0.99 * command_limit)),
    }


def evaluate_policy_rollout(
    policy,
    dynamics,
    observations,
    actions,
    *,
    n_history,
    episode_length,
    horizon,
    batch_size,
    loss_penalty,
    device,
    fallback_gain=None,
):
    """Evaluate candidate or proportional-fallback cost on held-back states."""
    import torch

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if fallback_gain is not None and not np.isfinite(fallback_gain):
        raise ValueError("fallback_gain must be finite")
    first = observations.shape[0] - episode_length + n_history - 1
    last = observations.shape[0] - horizon - 1
    if first > last:
        raise ValueError(
            "validation episode is too short for history and rollout horizon"
        )
    starts = np.unique(
        np.linspace(first, last, min(batch_size, last - first + 1), dtype=int)
    )
    past_observations = torch.as_tensor(
        np.stack([observations[t - n_history + 1 : t] for t in starts]),
        device=device,
    ).unsqueeze(2)
    past_actions = torch.as_tensor(
        np.stack([actions[t - n_history + 1 : t] for t in starts]),
        device=device,
    ).unsqueeze(2)
    state = (
        torch.as_tensor(observations[starts], device=device).unsqueeze(1).unsqueeze(1)
    )

    policy_was_training = policy.training
    dynamics_was_training = dynamics.training
    policy.eval()
    dynamics.eval()
    total = torch.zeros(starts.size, device=device)
    try:
        with torch.no_grad():
            for _ in range(horizon):
                history = torch.cat([past_observations, past_actions], dim=1)
                if fallback_gain is None:
                    action = policy(torch.cat([state, history], dim=1)).reshape(
                        starts.size, 1, 1, observations.shape[1]
                    )
                else:
                    action = fallback_gain * state
                prediction = dynamics(torch.cat([history, state, action], dim=1)).mean(
                    dim=1, keepdim=True
                )
                normalized_prediction = dynamics.normalize_observation(prediction)
                normalized_action = policy.normalize_action(action)
                total += normalized_prediction.pow(2).mean(dim=(1, 2, 3))
                total += loss_penalty * normalized_action.pow(2).mean(dim=(1, 2, 3))
                past_observations = torch.cat([past_observations[:, 1:], state], dim=1)
                past_actions = torch.cat([past_actions[:, 1:], action], dim=1)
                state = prediction
    finally:
        policy.train(policy_was_training)
        dynamics.train(dynamics_was_training)
    return float(total.mean().item())


def append_metrics(path, record):
    """Append one JSON record so plots remain reproducible after the notebook exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def _counter_rates(history, field):
    """Return non-negative per-second rates for a cumulative runtime counter."""
    timestamps = np.asarray(
        [entry["timestamp_unix_ns"] for entry in history], dtype=np.int64
    )
    values = np.asarray(
        [entry.get(field, np.nan) for entry in history], dtype=np.float64
    )
    rates = np.zeros(values.shape, dtype=np.float64)
    if values.size < 2:
        return rates
    elapsed = np.diff(timestamps).astype(np.float64) * 1e-9
    increments = np.diff(values)
    valid = (elapsed > 0) & (increments >= 0)
    rates[1:] = np.divide(
        increments,
        elapsed,
        out=np.zeros_like(increments),
        where=valid,
    )
    return rates


def plot_training_diagnostics(
    history, observations, actions, *, sample_rate_hz, policy=None
):
    """Build the compact per-episode diagnostic dashboard used by the notebook."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401 Registers optional plot styles.
    except ImportError:
        scienceplots = None
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    modes = _representative_modes(observations.shape[1])
    iterations = [entry["training_iteration"] for entry in history]
    latest = history[-1]

    if scienceplots is not None:
        plt.style.use(["science", "no-latex", "grid"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=_PLOT_COLORS),
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "grid.alpha": 0.25,
            "legend.frameon": True,
            "lines.linewidth": 1.8,
        }
    )

    figure, axes = plt.subplots(2, 5, figsize=(23, 8.5), constrained_layout=True)
    axes[0, 0].semilogy(
        iterations,
        [entry["dynamics_loss"] for entry in history],
        label="dynamics train",
    )
    axes[0, 0].semilogy(
        iterations,
        [entry["rollout_loss"][0] for entry in history],
        label="dynamics validation",
    )
    axes[0, 0].semilogy(
        iterations,
        [entry["policy_loss"] for entry in history],
        label="policy train",
    )
    axes[0, 0].semilogy(
        iterations,
        [entry["policy_validation_loss"] for entry in history],
        label="policy validation",
    )
    axes[0, 0].set(title="Training losses", xlabel="Training iteration", ylabel="Loss")
    axes[0, 0].legend()
    for index, entry in enumerate(history[1:], start=1):
        if (
            entry["runtime_active_model_id"]
            != history[index - 1]["runtime_active_model_id"]
        ):
            axes[0, 0].axvline(entry["training_iteration"], color="0.7", linewidth=0.8)

    axes[0, 1].semilogy(
        np.arange(1, len(latest["rollout_loss"]) + 1),
        latest["rollout_loss"],
        marker="o",
    )
    axes[0, 1].set(
        title="Dynamics rollout validation", xlabel="Horizon [frames]", ylabel="MSE"
    )

    axes[0, 2].semilogy(np.nanvar(observations, axis=0))
    axes[0, 2].set(title="Residual variance", xlabel="KL mode", ylabel="Variance")

    if policy is not None and hasattr(policy, "raw_linear_weights"):
        weights = policy.raw_linear_weights().detach().cpu().numpy()
        n_history = (weights.shape[0] + 1) // 2
        observation_lags = np.arange(-(n_history - 1), 1)
        command_lags = np.arange(-(n_history - 1), 0)
        for mode in range(min(2, weights.shape[1])):
            observation_weights = np.concatenate(
                [weights[1:n_history, mode], weights[:1, mode]]
            )
            axes[0, 3].plot(
                observation_lags,
                observation_weights,
                marker="o",
                label=f"mode {mode} measurement",
            )
            axes[0, 3].plot(
                command_lags,
                weights[n_history:, mode],
                marker="o",
                linestyle="--",
                label=f"mode {mode} command",
            )
        axes[0, 3].axhline(0, color="0.4", linewidth=0.8)
        axes[0, 3].set(
            title="Linear policy coefficients",
            xlabel="Lag [frames]",
            ylabel="Raw-unit weight",
        )
        axes[0, 3].legend(fontsize="small")
    else:
        axes[0, 3].set_axis_off()

    for field, label in (
        ("runtime_p50_us", "p50"),
        ("runtime_p95_us", "p95"),
        ("runtime_p99_us", "p99"),
        ("runtime_max_us", "maximum"),
    ):
        if not any(field in entry for entry in history):
            continue
        axes[0, 4].semilogy(
            iterations, [entry.get(field, np.nan) for entry in history], label=label
        )
    axes[0, 4].set(
        title="Real-time callback", xlabel="Training iteration", ylabel="Latency [µs]"
    )
    if axes[0, 4].lines:
        axes[0, 4].legend()
    else:
        axes[0, 4].text(
            0.5,
            0.5,
            "No runtime telemetry",
            ha="center",
            transform=axes[0, 4].transAxes,
        )

    samples = min(1000, observations.shape[0])
    time_axis = np.arange(samples) * 1000.0 / sample_rate_hz
    for color, mode in zip(_PLOT_COLORS, modes):
        axes[1, 0].plot(
            time_axis,
            observations[-samples:, mode],
            color=color,
            label=f"mode {mode}",
        )
        axes[1, 0].plot(
            time_axis,
            actions[-samples:, mode],
            color=color,
            linestyle="--",
            alpha=0.55,
        )
    axes[1, 0].set(
        title="Recent residuals (solid) and commands (dashed)",
        xlabel="Time [ms]",
        ylabel="Modal value",
    )
    axes[1, 0].legend()

    for mode in modes:
        frequency, density = welch(
            observations[:, mode],
            fs=sample_rate_hz,
            nperseg=min(1024, observations.shape[0]),
        )
        axes[1, 1].loglog(frequency[1:], density[1:], label=f"mode {mode}")
    axes[1, 1].set(title="Residual PSD", xlabel="Frequency [Hz]", ylabel="PSD")
    axes[1, 1].legend()

    axes[1, 2].plot(
        iterations,
        [entry["observation_rms"] for entry in history],
        label="residual RMS",
    )
    axes[1, 2].plot(
        iterations,
        [entry["action_rms"] for entry in history],
        label="applied action RMS",
    )
    axes[1, 2].plot(
        iterations,
        [entry["candidate_action_rms"] for entry in history],
        label="candidate action RMS",
    )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set(title="Control scale", xlabel="Training iteration", ylabel="RMS")
    axes[1, 2].legend()

    axes[1, 3].semilogy(
        iterations,
        [entry["candidate_cost_ratio"] for entry in history],
        label="predicted cost / fallback",
    )
    axes[1, 3].semilogy(
        iterations,
        [entry["candidate_action_rms_ratio"] for entry in history],
        label="candidate / applied RMS",
    )
    axes[1, 3].axhline(1, color="0.4", linestyle="--", linewidth=0.8)
    axes[1, 3].set(
        title="Candidate ratios",
        xlabel="Training iteration",
        ylabel="Ratio",
    )
    axes[1, 3].legend()

    for field, label in (
        ("runtime_missed_frames", "missed frames"),
        ("runtime_deadline_misses", "deadline misses"),
        ("runtime_inference_failures", "inference failures"),
        ("runtime_replay_drops", "replay drops"),
    ):
        if any(field in entry for entry in history):
            axes[1, 4].plot(iterations, _counter_rates(history, field), label=label)
    axes[1, 4].set(
        title="Runtime event rates",
        xlabel="Training iteration",
        ylabel="Events [s⁻¹]",
    )
    axes[1, 4].set_ylim(bottom=0)
    if axes[1, 4].lines:
        axes[1, 4].legend()
    else:
        axes[1, 4].text(
            0.5,
            0.5,
            "No runtime telemetry",
            ha="center",
            transform=axes[1, 4].transAxes,
        )

    figure.suptitle(
        f"PO4AO online training  •  episode {latest['episode_id']}  •  "
        f"model {latest['runtime_active_model_id']}  •  "
        f"{latest['delay_source']} delay {latest['estimated_delay_frames']} frames  •  "
        f"clipped {latest['candidate_clip_fraction']:.2%}",
        fontsize=14,
        fontweight="bold",
    )
    figure.align_labels()
    return figure


def _plot_coefficient_map(axis, weights, lags, title):
    """Plot signed raw-unit coefficients with mode vertically and lag horizontally."""
    if weights.shape[0] == 0:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "No command history (Nhist = 1)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return
    limit = max(float(np.max(np.abs(weights))), np.finfo(float).eps)
    image = axis.imshow(
        weights.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[lags[0] - 0.5, lags[-1] + 0.5, -0.5, weights.shape[1] - 0.5],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.grid(False)
    axis.set(
        title=title,
        xlabel="Input lag [frames]",
        ylabel="Controlled KL mode",
        xticks=lags,
    )
    axis.figure.colorbar(image, ax=axis, label="Raw-unit coefficient")


def _dynamics_raw_weights(dynamics):
    """Get the ensemble-mean linear dynamics weights, including delay masking."""
    members = []
    for model in dynamics.models:
        if hasattr(model, "raw_linear_weights"):
            weights = model.raw_linear_weights().detach().cpu().numpy()
        elif (
            hasattr(model, "normalizer")
            and hasattr(getattr(model, "net", None), "weight")
            and model.net.weight.ndim == 4
        ):
            # Older deployed models do not yet expose raw_linear_weights().
            normalizer = model.normalizer
            weights = (
                (
                    model.net.weight[0, :, 0]
                    * normalizer.observation_scale.reshape(-1)
                    / normalizer.dynamics_feature_scale()[0, :, 0]
                )
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            if model.control_delay > 0:
                weights[-1] = 0
                if model.control_delay > 1:
                    first = 2 * model.n_history - model.control_delay - 1
                    weights[first : 2 * model.n_history - 2] = 0
        else:
            return None
        members.append(weights)
    return np.mean(members, axis=0) if members else None


def plot_model_debug_diagnostics(history, *, policy=None, dynamics=None):
    """Plot model fidelity, realized cost, replay shift, and optional linear maps."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401 Registers optional plot styles.
    except ImportError:
        scienceplots = None
    latest = history[-1]
    iterations = np.asarray(
        [entry["training_iteration"] for entry in history], dtype=int
    )

    if scienceplots is not None:
        plt.style.use(["science", "no-latex", "grid"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=_PLOT_COLORS),
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "grid.alpha": 0.25,
            "legend.frameon": True,
            "lines.linewidth": 1.8,
        }
    )

    include_maps = policy is not None or dynamics is not None
    figure, axes = plt.subplots(
        4 if include_maps else 2,
        2,
        figsize=(13.5, 16.5 if include_maps else 8.5),
        constrained_layout=True,
    )

    observed = np.asarray(latest["dynamics_trace_observed_normalized"])
    predicted = np.asarray(latest["dynamics_trace_predicted_normalized"])
    trace_modes = latest["dynamics_trace_modes"]
    horizon = np.arange(observed.shape[0])
    for color, column, mode in zip(_PLOT_COLORS, range(observed.shape[1]), trace_modes):
        axes[0, 0].plot(
            horizon,
            observed[:, column],
            color=color,
            marker="o",
            label=f"mode {mode} measured",
        )
        axes[0, 0].plot(
            horizon,
            predicted[:, column],
            color=color,
            marker="s",
            linestyle="--",
            label=f"mode {mode} predicted",
        )
    axes[0, 0].set(
        title="Representative dynamics unroll",
        xlabel="Rollout horizon [frames]",
        ylabel="Modal value / warm-up RMS",
    )
    axes[0, 0].legend(fontsize="small", ncol=2)

    candidate_cost = np.asarray(
        [entry["predicted_candidate_cost_per_step"] for entry in history]
    )
    realized_cost = np.asarray([entry["measured_cost_per_step"] for entry in history])
    predicted_active_cost = np.asarray(
        [
            (
                np.nan
                if entry["predicted_active_cost_per_step"] is None
                else entry["predicted_active_cost_per_step"]
            )
            for entry in history
        ],
        dtype=float,
    )
    axes[0, 1].semilogy(
        iterations,
        candidate_cost,
        color="0.55",
        label="current candidate prediction",
    )
    axes[0, 1].semilogy(
        iterations,
        predicted_active_cost,
        marker="o",
        label="active model prediction",
    )
    axes[0, 1].semilogy(
        iterations,
        realized_cost,
        marker="o",
        label="realized replay cost",
    )
    axes[0, 1].set(
        title="Predicted vs realized cost (reward = -cost)",
        xlabel="Training iteration",
        ylabel="Normalized cost per frame",
    )
    axes[0, 1].legend(fontsize="small")

    per_mode_nrmse = np.asarray(latest["dynamics_per_mode_nrmse"])
    axes[1, 0].semilogy(
        np.arange(per_mode_nrmse.size),
        np.maximum(per_mode_nrmse, np.finfo(float).tiny),
    )
    axes[1, 0].axhline(1, color="0.4", linestyle="--", linewidth=0.8)
    axes[1, 0].set(
        title="Dynamics error by mode",
        xlabel="Controlled mode",
        ylabel="Rollout NRMSE [warm-up RMS]",
    )

    observation_ratio = np.asarray(latest["observation_scale_ratio"])
    action_ratio = np.asarray(latest["action_scale_ratio"])
    modes = np.arange(observation_ratio.size)
    axes[1, 1].semilogy(modes, observation_ratio, label="residual RMS / warm-up")
    axes[1, 1].semilogy(modes, action_ratio, label="command RMS / warm-up")
    axes[1, 1].axhline(1, color="0.4", linestyle="--", linewidth=0.8)
    axes[1, 1].set(
        title="Replay distribution shift",
        xlabel="Controlled mode",
        ylabel="RMS ratio",
    )
    axes[1, 1].legend(fontsize="small")

    if include_maps:
        if policy is not None and hasattr(policy, "raw_linear_weights"):
            weights = policy.raw_linear_weights().detach().cpu().numpy()
            n_history = (weights.shape[0] + 1) // 2
            _plot_coefficient_map(
                axes[2, 0],
                np.vstack([weights[1:n_history], weights[:1]]),
                np.arange(-(n_history - 1), 1),
                "Candidate policy: measurements",
            )
            _plot_coefficient_map(
                axes[2, 1],
                weights[n_history:],
                np.arange(-(n_history - 1), 0),
                "Candidate policy: commands",
            )
        else:
            for axis in axes[2]:
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    "Policy has no linear coefficient map",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
        weights = _dynamics_raw_weights(dynamics) if dynamics is not None else None
        if weights is not None:
            n_history = weights.shape[0] // 2
            lags = np.arange(-(n_history - 1), 1)
            _plot_coefficient_map(
                axes[3, 0],
                np.vstack([weights[: n_history - 1], weights[-2:-1]]),
                lags,
                "Dynamics: measurement history",
            )
            _plot_coefficient_map(
                axes[3, 1],
                np.vstack([weights[n_history - 1 : -2], weights[-1:]]),
                lags,
                "Dynamics: command history",
            )
        else:
            for axis in axes[3]:
                axis.set_axis_off()
                axis.text(
                    0.5,
                    0.5,
                    "Dynamics have no linear coefficient map",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

    matched_costs = np.isfinite(predicted_active_cost).sum()
    figure.suptitle(
        f"PO4AO model diagnostics  •  episode {latest['episode_id']}  •  "
        f"replay model {latest['active_model_id']}  •  "
        f"{matched_costs} matched predicted/realized points",
        fontsize=14,
        fontweight="bold",
    )
    figure.align_labels()
    return figure


def new_metrics_record(**values):
    """Return a JSON-safe timestamped metrics record."""
    record = {"timestamp_unix_ns": time.time_ns()}
    record.update(values)
    return record
