"""BBFAgent subclass that captures training metrics for external logging."""

import os
import time

from absl import logging
import gin
import jax
import numpy as onp
import tensorflow as tf

from bigger_better_faster.bbf.agents.spr_agent import BBFAgent
from bigger_better_faster.bbf.agents.spr_agent import jit_split
from bigger_better_faster.bbf.agents.spr_agent import tree_norm


@gin.configurable
class MetricBBFAgent(BBFAgent):
    """BBFAgent that exposes per-step training metrics.

    The published BBFAgent writes metrics to a TF summary writer and
    discards them. This subclass stores the most recent metrics dict
    in self._last_metrics so that an external training loop can read
    and log them without intercepting the summary writer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_metrics = {}
        self._reset_log = []

    def reset_weights(self):
        """Call parent reset_weights and log the event for JSON serialization."""
        step_before = self.training_steps
        resets_before = self.cumulative_resets
        super().reset_weights()
        # If the reset was skipped (too late in training), cumulative_resets
        # still incremented but next_reset was not updated and the method
        # returned early after logging. Detect actual resets by checking
        # whether cycle_grad_steps was zeroed.
        if self.cycle_grad_steps == 0:
            self._reset_log.append({
                "training_step": step_before,
                "reset_index": resets_before,
                "cumulative_resets": self.cumulative_resets,
                "shrink_factor": self.shrink_factor,
                "perturb_factor": self.perturb_factor,
                "keys_shrink_perturbed": list(self.shrink_perturb_keys),
                "next_reset": self.next_reset,
            })

    # ------------------------------------------------------------------
    # Verbatim copy of BBFAgent._training_step_update with one addition:
    #   self._last_metrics = metrics
    # inserted before the TF summary write block.
    # ------------------------------------------------------------------
    def _training_step_update(self, step_index, offline=False):
        """Gradient update during every training step."""
        should_log = (
            self.training_steps % self.log_every == 0 and not offline and
            step_index == 0)
        interbatch_time = time.time() - self.start
        self.start = time.time()
        train_start = time.time()

        if not hasattr(self, "replay_elements"):
            self._sample_from_replay_buffer()
        if self._replay_scheme == "prioritized":
            probs = self.replay_elements["sampling_probabilities"]
            loss_weights = 1.0 / onp.sqrt(probs + 1e-10)
            loss_weights /= onp.max(loss_weights)
            indices = self.replay_elements["indices"]
        else:
            loss_weights = onp.ones(self.replay_elements["state"].shape[0:1])

        if self.log_churn and should_log:
            eval_batch = self.sample_eval_batch(256)
            eval_states = eval_batch["state"].reshape(
                -1, *eval_batch["state"].shape[-3:])
            eval_actions = eval_batch["action"].reshape(-1,)
            self._rng, eval_rng = jax.random.split(self._rng, 2)
            og_actions = self.select_action(
                eval_states,
                self.online_params,
                eval_mode=True,
                force_zero_eps=True,
                rng=eval_rng,
                use_noise=False,
            )
            og_target_actions = self.select_action(
                eval_states,
                self.target_network_params,
                eval_mode=True,
                force_zero_eps=True,
                rng=eval_rng,
                use_noise=False,
            )

        self._rng, train_rng = jit_split(self._rng, num=2)
        (
            new_online_params,
            new_target_params,
            new_optimizer_state,
            new_dynamic_scale,
            aux_losses,
        ) = self.train_fn(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            self.replay_elements["state"],
            self.replay_elements["action"],
            self.replay_elements["next_state"],
            self.replay_elements["return"],
            self.replay_elements["terminal"],
            self.replay_elements["same_trajectory"],
            loss_weights,
            self._support,
            self.replay_elements["discount"],
            self._double_dqn,
            self._distributional,
            train_rng,
            self.spr_weight,
            self.dynamic_scale,
            self._data_augmentation,
            self.dtype,
            self._batch_size,
            self.use_target_network,
            self.target_update_tau_scheduler(self.cycle_grad_steps),
            self.target_update_period,
            self.grad_steps,
            self.match_online_target_rngs,
            self.target_eval_mode,
        )
        self.grad_steps += self._batches_to_group
        self.cycle_grad_steps += self._batches_to_group

        sample_start = time.time()
        self._sample_from_replay_buffer()
        sample_time = time.time() - sample_start

        prio_set_start = time.time()
        if self._replay_scheme == "prioritized":
            indices = onp.reshape(onp.asarray(indices), (-1,))
            dqn_loss = onp.reshape(onp.asarray(aux_losses["DQNLoss"]), (-1))
            priorities = onp.sqrt(dqn_loss + 1e-10)
            self._replay.set_priority(indices, priorities)
        prio_set_time = time.time() - prio_set_start

        training_time = time.time() - train_start
        if (self.training_steps % self.log_every == 0 and not offline and
                step_index == 0):
            metrics = {
                **{k: onp.mean(v) for k, v in aux_losses.items()},
                "PNorm": float(tree_norm(new_online_params)),
                "Inter-batch time":
                    float(interbatch_time) / self._batches_to_group,
                "Training time":
                    float(training_time) / self._batches_to_group,
                "Sampling time":
                    float(sample_time) / self._batches_to_group,
                "Set priority time":
                    float(prio_set_time) / self._batches_to_group,
            }

            if self.log_churn:
                new_actions = self.select_action(
                    eval_states,
                    new_online_params,
                    eval_mode=True,
                    force_zero_eps=True,
                    rng=eval_rng,
                    use_noise=False,
                )
                new_target_actions = self.select_action(
                    eval_states,
                    new_target_params,
                    eval_mode=True,
                    force_zero_eps=True,
                    rng=eval_rng,
                    use_noise=False,
                )
                online_churn = onp.mean(new_actions != og_actions)
                target_churn = onp.mean(
                    new_target_actions != og_target_actions)
                online_off_policy_frac = onp.mean(
                    new_actions != eval_actions)
                target_off_policy_frac = onp.mean(
                    new_target_actions != eval_actions)
                online_target_agreement = onp.mean(
                    new_actions == new_target_actions)
                churn_metrics = {
                    "Online Churn": online_churn,
                    "Target Churn": target_churn,
                    "Online-Target Agreement": online_target_agreement,
                    "Online Off-Policy Rate": online_off_policy_frac,
                    "Target Off-Policy Rate": target_off_policy_frac,
                }
                metrics.update(**churn_metrics)

            if self.dynamic_scale:
                metrics["Dynamic Scale"] = self.dynamic_scale.scale

            diff_tree = jax.tree_util.tree_map(
                lambda a, b: a - b, new_online_params, new_target_params)
            metrics["TargetDivergence"] = float(tree_norm(diff_tree))

            self._last_metrics = metrics

            if self.summary_writer is not None:
                with self.summary_writer.as_default():
                    for k, v in metrics.items():
                        tf.summary.scalar(k, v, step=self.training_steps)
            if self.verbose:
                logging.info(str(metrics))

        self.target_network_params = new_target_params
        self.online_params = new_online_params
        self.optimizer_state = new_optimizer_state
        self.dynamic_scale = new_dynamic_scale

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Save standard checkpoint then dump replay buffer contents."""
        bundle = super().bundle_and_checkpoint(checkpoint_dir, iteration_number)
        if bundle is None:
            return None

        replay = self._replay
        n_valid = min(int(replay.add_count), replay._replay_capacity)
        arrays = {}
        for name, arr in replay._store.items():
            arrays[name] = arr[:n_valid]

        if hasattr(replay, 'sum_tree'):
            priorities = onp.array([
                replay.sum_tree.get(i) for i in range(n_valid)])
            arrays['priority'] = priorities

        path = os.path.join(
            checkpoint_dir,
            f'replay_buffer_{iteration_number}.npz')
        onp.savez_compressed(path, **arrays)
        logging.info('Saved replay buffer snapshot (%d entries) to %s',
                     n_valid, path)
        return bundle
