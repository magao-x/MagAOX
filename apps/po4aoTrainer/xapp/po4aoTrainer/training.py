## \file training.py
# \brief Stateful notebook and worker training without instrument I/O.
"""Keep numerical session state across calls; accept copied replay snapshots explicitly.

This module never opens streams, changes control state, or publishes engines.
A caller may pass a cooperative checkpoint which blocks for pause/resume or
raises to terminate the session. It runs before each optimizer update and at
iteration boundaries. Aborting an iteration is terminal, not transactional:
updates already completed remain in memory. Reset discards that session.
"""

from copy import deepcopy
import hashlib
import json
import time
import uuid

import numpy as np
import torch

from .po4ao_nonlinear_models import (
    build_nonlinear_models,
    train_nonlinear_dynamics,
    train_nonlinear_policy,
)
from .po4ao_policy_update import PolicyUpdatePenalty, policy_history_batch
from .po4ao_replay import HistoricalReplay
from .po4ao_runtime import (
    modal_rms_scales,
    model_publication_feedback,
    select_controlled_modes,
)
from .po4ao_diagnostics import (
    evaluate_dynamics_rollout_details,
    evaluate_policy_rollout,
    replay_metrics,
    evaluate_policy_actions,
    measured_replay_cost,
    modal_scale_ratios,
    new_metrics_record,
)


class TrainingSession:
    """Own models, Adam state, fixed normalization and publication bookkeeping.

    Use one caller at a time. Configuration is a deep-copied resolved notebook
    mapping, not an import-time configuration module. Models and optimizers are
    public for notebook inspection. File watching and managed export belong to
    the companion and are not provided by this initial numerical layer.
    """

    def __init__(self, config, *, device="cpu"):
        """Allocate one session on an explicit device without changing process affinity."""
        self.session_id = uuid.uuid4().hex
        self._config = deepcopy(config)
        self.config_revision = hashlib.sha256(
            json.dumps(self._config, sort_keys=True).encode()
        ).hexdigest()
        self.device = torch.device(device)
        self.n_history = config["MDP"]["n_history"]
        self.controlled_nmodes = config["runtime"]["n_modes_controlled"]
        self.episode_length = config["RL"]["episode_length"]
        self.required_samples = config["replay_buffers"]["capacity"]
        self.training_samples = self.required_samples - (
            config["training"]["validation_episodes"] * self.episode_length
        )
        if (
            config["training"]["validation_episodes"] < 1
            or self.episode_length < self.n_history + config["MDP"]["planning_horizon"]
            or self.required_samples % self.episode_length
            or self.training_samples < self.episode_length
        ):
            raise ValueError(
                "Reserve complete validation and training episodes with sufficient history"
            )
        positive = (
            "dynamics_pretrain_steps",
            "dynamics_grad_steps",
            "policy_grad_steps",
            "dynamics_learning_rate",
            "policy_learning_rate",
            "diagnostics_interval",
            "model_publish_interval",
            "training_iterations",
            "dynamics_gradient_clip_norm",
            "policy_gradient_clip_norm",
        )
        if any(
            not np.isfinite(config["training"][key]) or config["training"][key] <= 0
            for key in positive
        ):
            raise ValueError(
                "Training steps, learning rates, limits and intervals must be positive"
            )
        self.dynamics, self.policy = build_nonlinear_models(self._config)
        self.dynamics.to(self.device)
        self.policy.to(self.device)
        self.policy_copy = deepcopy(self.policy).eval()
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics.parameters(), lr=config["training"]["dynamics_learning_rate"]
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["training"]["policy_learning_rate"]
        )
        self.historical = HistoricalReplay(
            self.episode_length,
            seconds=config["replay_buffers"]["historical_seconds"],
            max_episodes=config["replay_buffers"]["historical_episodes"],
        )
        self.iteration = 0
        self.dynamics_pretrained = False
        self.normalization_metadata = None
        self.policy_update = None
        self.last_publish_iteration = None
        self.published_cost_by_model = {}
        self.metrics_history = []
        self.last_result = None
        self.last_replay_identity = None
        self.warmup_generation = None
        self.invalidated_warmup = None
        self.state = "waiting"

    @property
    def config(self):
        """Return an independent configuration snapshot for diagnostics and export."""
        return deepcopy(self._config)

    def invalidate_replay(self):
        """Discard old history/anchors at recollection start without changing the policy.

        The stream adapter must call this when recollection begins, before a new
        warmup snapshot exists. Training waits until a complete newer snapshot.
        Frozen normalization and optimizer/reference state remain untouched.
        """
        self.invalidated_warmup = self.warmup_generation
        self.last_result = None
        self.last_replay_identity = None
        self.historical.clear()
        if self.policy_update is not None:
            self.policy_update.anchors = self.policy_update.reference_actions = None
        self.state = "waiting"

    def _replay_pair(self, snapshot):
        """Validate and copy complete newest-first stream data into chronological arrays."""
        expected = (self._config["runtime"]["stream_n_modes"], self.required_samples, 2)
        if (
            snapshot.data.shape != expected
            or snapshot.valid_length != self.required_samples
        ):
            raise ValueError(
                f"Expected a complete replay snapshot with shape {expected}"
            )
        pair = tuple(
            select_controlled_modes(
                snapshot.data[:, :, plane].T[::-1], self.controlled_nmodes
            ).copy()
            for plane in (0, 1)
        )
        if not all(np.isfinite(array).all() for array in pair):
            raise ValueError("Replay contains non-finite values")
        return pair

    def _initialize_normalization(self, warmup):
        """Freeze warmup scales once before the first gradient update."""
        floor = self._config["normalization"]["rms_floor_fraction"]
        gain = self._config["integrator"]["gain"]
        self.observation_scale, self.action_scale = modal_rms_scales(
            *warmup, floor_fraction=floor
        )
        for model in (self.dynamics, self.policy, self.policy_copy):
            model.set_modal_scales(self.observation_scale, self.action_scale)
        self.normalization_metadata = {
            "method": "fixed_warmup_per_mode_rms",
            "rms_floor_fraction": floor,
            "observation_scale": self.observation_scale.tolist(),
            "action_scale": self.action_scale.tolist(),
        }
        self.update_scale = np.maximum(
            abs(gain) * self.observation_scale, floor * np.median(self.action_scale)
        )
        self.policy_update = PolicyUpdatePenalty(
            architecture=self._config["NN_models"]["policy_architecture"],
            action_scale=self.update_scale,
            baseline_gain=gain,
            weight=self._config["policy_update"]["weight"],
        )

    def step(
        self,
        replay,
        warmup,
        *,
        active_model_id,
        telemetry=None,
        checkpoint=None,
    ):
        """Complete one update on fresh complete data, or return None while waiting.

        ``replay`` and ``warmup`` are ReplaySnapshot objects captured by the
        caller. No counters advance on incomplete/duplicate data. The optional
        checkpoint can wait for resume while retaining this exact Python stack;
        returning resumes the same update, without recreating Adam or models.
        """
        if self.state == "failed":
            raise RuntimeError("Discard the failed training session before continuing")
        try:
            return self._step(replay, warmup, active_model_id, telemetry, checkpoint)
        except BaseException:
            self.state = "failed"
            self.last_result = None
            raise

    def _step(self, replay, warmup, active_model_id, telemetry, checkpoint):
        """Run one numerical iteration; the public wrapper invalidates failed candidates."""
        check = checkpoint if checkpoint is not None else lambda: None
        check()
        if self.iteration >= self._config["training"]["training_iterations"]:
            self.state = "completed"
            return None
        if warmup.counter != self.warmup_generation:
            self.last_result = None
            self.historical.clear()
            if self.policy_update is not None:
                self.policy_update.anchors = self.policy_update.reference_actions = None
            self.warmup_generation = warmup.counter
        identity = (replay.counter, replay.episode_id, warmup.counter)
        if (
            identity == self.last_replay_identity
            or warmup.counter == self.invalidated_warmup
            or min(replay.valid_length, warmup.valid_length) < self.required_samples
        ):
            self.state = "waiting"
            return None
        self.obs_replay, self.act_replay = self._replay_pair(replay)
        self.obs_replay_warmup, self.act_replay_warmup = self._replay_pair(warmup)
        self.buffer_snapshot = replay
        self.last_result = None
        config = self._config
        start = time.monotonic()
        self.historical.observe(
            self.obs_replay,
            self.act_replay,
            episode_id=replay.episode_id,
            validation_episodes=config["training"]["validation_episodes"],
        )
        if self.normalization_metadata is None:
            self._initialize_normalization(
                (self.obs_replay_warmup, self.act_replay_warmup)
            )
        self.policy_update.observe_active_model(active_model_id)
        if self.policy_update.anchors is None:
            self.policy_update.set_anchors(
                torch.cat(
                    [
                        policy_history_batch(
                            obs[: self.training_samples],
                            act[: self.training_samples],
                            n_history=self.n_history,
                            episode_length=self.episode_length,
                            samples=config["policy_update"]["anchor_samples"],
                            device=self.device,
                        )
                        for obs, act in (
                            (self.obs_replay, self.act_replay),
                            (self.obs_replay_warmup, self.act_replay_warmup),
                        )
                    ],
                    dim=0,
                )
            )
        recent = (
            self.obs_replay[: self.training_samples],
            self.act_replay[: self.training_samples],
        )
        initial = (
            self.obs_replay_warmup[: self.training_samples],
            self.act_replay_warmup[: self.training_samples],
        )
        dynamics_steps = config["training"][
            (
                "dynamics_grad_steps"
                if self.dynamics_pretrained
                else "dynamics_pretrain_steps"
            )
        ]
        self.state = "running"
        dyn_loss = train_nonlinear_dynamics(
            self.dynamics_optimizer,
            self.dynamics,
            recent,
            initial,
            settings=config,
            steps=dynamics_steps,
            device=self.device,
            historical=self.historical,
            checkpoint=check,
        )
        self.dynamics_pretrained = True
        pol_loss = train_nonlinear_policy(
            self.policy_optimizer,
            self.policy,
            self.dynamics,
            recent,
            initial,
            settings=config,
            steps=config["training"]["policy_grad_steps"],
            device=self.device,
            policy_update=self.policy_update,
            historical=self.historical,
            checkpoint=check,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        check()
        self.policy_copy.load_state_dict(self.policy.state_dict())
        self.iteration += 1
        self.last_replay_identity = identity
        result = {
            "training_iteration": self.iteration,
            "dynamics_loss": dyn_loss,
            "policy_loss": pol_loss,
            "training_seconds": time.monotonic() - start,
            "config_revision": self.config_revision,
            "record": None,
        }
        if self.iteration % config["training"]["diagnostics_interval"] == 0:
            result["record"] = self._diagnostics(
                dyn_loss,
                pol_loss,
                dynamics_steps,
                result["training_seconds"],
                PolicyUpdatePenalty._model_id(active_model_id),
                telemetry or {},
            )
            self.metrics_history.append(result["record"])
        check()
        self.last_result = result
        self.state = (
            "completed"
            if self.iteration >= config["training"]["training_iterations"]
            else "ready"
        )
        return result

    def _diagnostics(
        self,
        dyn_loss,
        pol_loss,
        dynamics_steps,
        training_seconds,
        active_model_id,
        telemetry,
    ):
        """Evaluate the notebook's held-out diagnostics and unchanged publication gates."""
        config = self._config
        planning_horizon = config["MDP"]["planning_horizon"]
        control_delay = config["MDP"]["control_delay"]
        batch_size = config["NN_models"]["training_batch"]
        loss_penalty = config["RL"]["loss_function_penalty"]
        gain = config["integrator"]["gain"]
        command_limit = config["safety"]["max_command_abs"]
        training_device = self.device
        update_settings = config["policy_update"]
        first_publish_iteration = config["training"]["first_model_publish_iteration"]
        publish_interval = config["training"]["model_publish_interval"]
        max_clip_fraction = config["training"]["max_policy_clip_fraction"]
        max_candidate_action_rms_ratio = config["training"][
            "max_candidate_action_rms_ratio"
        ]
        dynamics_diagnostics = evaluate_dynamics_rollout_details(
            self.dynamics,
            self.obs_replay,
            self.act_replay,
            n_history=self.n_history,
            episode_length=self.episode_length,
            horizon=planning_horizon,
            batch_size=batch_size,
            device=training_device,
        )
        rollout_loss = dynamics_diagnostics["horizon_mse"]
        policy_validation_loss = evaluate_policy_rollout(
            self.policy_copy,
            self.dynamics,
            self.obs_replay,
            self.act_replay,
            n_history=self.n_history,
            episode_length=self.episode_length,
            horizon=planning_horizon,
            batch_size=batch_size,
            loss_penalty=loss_penalty,
            device=training_device,
        )
        fallback_validation_loss = evaluate_policy_rollout(
            self.policy_copy,
            self.dynamics,
            self.obs_replay,
            self.act_replay,
            n_history=self.n_history,
            episode_length=self.episode_length,
            horizon=planning_horizon,
            batch_size=batch_size,
            loss_penalty=loss_penalty,
            device=training_device,
            fallback_gain=gain,
        )
        health = replay_metrics(
            self.obs_replay,
            self.act_replay,
            command_limit,
            assumed_control_delay=control_delay,
        )
        candidate_health = evaluate_policy_actions(
            self.policy_copy,
            self.obs_replay,
            self.act_replay,
            n_history=self.n_history,
            episode_length=self.episode_length,
            batch_size=batch_size,
            command_limit=command_limit,
            device=training_device,
        )
        candidate_cost_ratio = float(
            policy_validation_loss / max(fallback_validation_loss, np.finfo(float).tiny)
        )
        candidate_action_rms_ratio = float(
            candidate_health["candidate_action_rms"]
            / max(health["action_rms"], np.finfo(float).tiny)
        )
        measured_cost = measured_replay_cost(
            self.obs_replay,
            self.act_replay,
            observation_scale=self.observation_scale,
            action_scale=self.action_scale,
            loss_penalty=loss_penalty,
            episode_length=self.episode_length,
        )
        scale_ratios = modal_scale_ratios(
            self.obs_replay,
            self.act_replay,
            observation_scale=self.observation_scale,
            action_scale=self.action_scale,
            episode_length=self.episode_length,
        )
        predicted_candidate_cost_per_step = float(
            policy_validation_loss / planning_horizon
        )
        predicted_active_cost_per_step = self.published_cost_by_model.get(
            self.buffer_snapshot.model_id
        )
        if (
            not np.isfinite(rollout_loss).all()
            or not np.isfinite(policy_validation_loss)
            or (not np.isfinite(fallback_validation_loss))
            or (not np.isfinite(candidate_cost_ratio))
            or (not np.isfinite(candidate_action_rms_ratio))
        ):
            raise RuntimeError("non-finite validation metric; model was not published")
        record = new_metrics_record(
            training_iteration=self.iteration,
            training_seconds=training_seconds,
            dynamics_loss=float(dyn_loss),
            policy_loss=float(pol_loss),
            policy_validation_loss=policy_validation_loss,
            fallback_policy_validation_loss=fallback_validation_loss,
            candidate_cost_ratio=candidate_cost_ratio,
            candidate_action_rms_ratio=candidate_action_rms_ratio,
            dynamics_gradient_steps=dynamics_steps,
            rollout_loss=rollout_loss,
            dynamics_per_mode_nrmse=dynamics_diagnostics["per_mode_nrmse"],
            dynamics_trace_modes=dynamics_diagnostics["trace_modes"],
            dynamics_trace_observed_normalized=dynamics_diagnostics[
                "trace_observed_normalized"
            ],
            dynamics_trace_predicted_normalized=dynamics_diagnostics[
                "trace_predicted_normalized"
            ],
            dynamics_trace_start_index=dynamics_diagnostics["trace_start_index"],
            predicted_candidate_cost_per_step=predicted_candidate_cost_per_step,
            predicted_active_cost_per_step=predicted_active_cost_per_step,
            measured_cost_per_step=measured_cost["total"],
            measured_observation_cost_per_step=measured_cost["observation"],
            measured_action_cost_per_step=measured_cost["action_penalty"],
            observation_scale_ratio=scale_ratios["observation"].tolist(),
            action_scale_ratio=scale_ratios["action"].tolist(),
            episode_id=self.buffer_snapshot.episode_id,
            replay_counter=self.buffer_snapshot.counter,
            active_model_id=self.buffer_snapshot.model_id,
            published_model_id=None,
        )
        record.update(self.historical.metrics())
        record.update(health)
        record.update(candidate_health)
        record.update(
            {
                key: value
                for key, value in telemetry.items()
                if key.startswith("runtime_")
            }
        )
        record["runtime_active_model_id"] = active_model_id
        update_checks = [
            self.policy_update.metrics(
                self.policy_copy,
                policy_history_batch(
                    observations[-self.episode_length :],
                    actions[-self.episode_length :],
                    n_history=self.n_history,
                    episode_length=self.episode_length,
                    samples=update_settings["anchor_samples"],
                    device=training_device,
                ),
            )
            for (observations, actions) in (
                (self.obs_replay, self.act_replay),
                (self.obs_replay_warmup, self.act_replay_warmup),
            )
        ]
        record["policy_update"] = update_checks
        record.update(
            model_publication_feedback(
                iteration=self.iteration,
                first_iteration=first_publish_iteration,
                last_publish_iteration=self.last_publish_iteration,
                publish_interval=publish_interval,
                candidate_clip_fraction=candidate_health["candidate_clip_fraction"],
                max_clip_fraction=max_clip_fraction,
                candidate_action_rms_ratio=candidate_action_rms_ratio,
                max_action_rms_ratio=max_candidate_action_rms_ratio,
                nonfinite_samples=health["nonfinite_samples"],
                policy_update_finite=all(
                    (check["policy_update_finite"] for check in update_checks)
                ),
                pending_model_id=(
                    None
                    if self.policy_update.pending is None
                    else self.policy_update.pending[0]
                ),
                active_model_id=record["runtime_active_model_id"],
                reference_model_id=self.policy_update.reference_id,
            )
        )
        record["config_revision"] = self.config_revision
        record["session_id"] = self.session_id
        record["warmup_generation"] = self.warmup_generation
        return record

    def stage_publication(self, model_id):
        """Record an explicitly exported current candidate, awaiting runtime acknowledgement.

        This is numerical bookkeeping only. The future managed supervisor must
        check its session generation and atomically commit artifacts before
        requesting reload. Calling this method never performs export or INDI I/O.
        """
        record = None if self.last_result is None else self.last_result["record"]
        if record is None or record["publication_status"] != "ready":
            raise RuntimeError("The current iteration has no eligible publication")
        self.policy_update.stage_publication(self.policy_copy, model_id)
        model_id = PolicyUpdatePenalty._model_id(model_id)
        record["published_model_id"] = model_id
        record["publication_status"] = "published (awaiting activation)"
        self.published_cost_by_model[model_id] = record[
            "predicted_candidate_cost_per_step"
        ]
        self.last_publish_iteration = self.iteration
