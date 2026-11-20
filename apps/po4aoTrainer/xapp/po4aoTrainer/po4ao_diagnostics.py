## \file po4ao_diagnostics.py
# \brief Packaged diagnostics from the nonlinear notebook.
"""Lightweight training and replay diagnostics for modal PO4AO."""

import json
from pathlib import Path
import time

import numpy as np


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


def new_metrics_record(**values):
    """Return a JSON-safe timestamped metrics record."""
    record = {"timestamp_unix_ns": time.time_ns()}
    record.update(values)
    return record
