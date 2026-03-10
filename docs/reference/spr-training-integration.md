# SPR Training Integration

> **Quick links:** [SPR Architecture](spr-architecture.md) . [DQN Training](dqn-training.md) . [Training Loop](training-loop-runtime.md)

<br><br>

## Overview

This document describes how SPR integrates into the DQN training
pipeline: how losses combine, when the EMA encoder updates, how
sequence data is sampled, what config options control SPR behavior,
how checkpoints change, and how to toggle SPR on or off.

SPR adds a self-supervised auxiliary loss alongside the standard TD
loss. The training loop samples two batches per optimization step
when SPR is active: one standard `(s, a, r, s', done)` batch for
Q-learning and one consecutive-sequence batch for SPR predictions.

**Source files:**

| File | Role |
|------|------|
| `src/training/metrics.py` | `perform_update_step()` -- combined loss, EMA update |
| `src/training/training_loop.py` | `training_step()` -- sequence sampling, augmentation |
| `src/training/spr_loss.py` | `compute_spr_loss()`, `compute_spr_forward()` |
| `src/training/metrics_logger.py` | SPR metric keys and `log_step()` integration |
| `src/training/logging.py` | Checkpoint save/load with SPR state |
| `src/replay/replay_buffer.py` | `sample_sequences()` for SPR data |
| `experiments/dqn_atari/configs/base.yaml` | Default SPR/EMA/dropout config |

<br><br>

## Combined Loss Formula

The total training loss is the sum of the temporal-difference loss
and the weighted SPR loss:

```text
L_total = L_TD + lambda * L_SPR
```

where `lambda = 2.0` (config key `spr.loss_weight`, from Schwarzer
et al. 2021, Table 3).

### TD loss

Standard DQN mean-squared error (or Huber) between the online
Q-value for the taken action and the target:

```text
L_TD = (1/B) * sum_b (Q(s_b, a_b) - y_b)^2
y_b  = r_b + gamma * max_a Q_target(s'_b, a)
```

### SPR loss

Negative cosine similarity between online predictions and EMA
targets, averaged over valid (unmasked) entries across K prediction
steps:

```text
L_SPR = -(1/N) * sum_{k,b} mask_{k,b} * cos_sim(y_hat_{k,b}, y_tilde_{k,b})
```

The mask is a cumulative product of `(1 - done)` flags that zeroes
out all steps at and after the first episode boundary in each
sample's sequence. `N` is the count of valid entries, clamped to a
minimum of 1 to avoid division by zero.

### Loss combination

`compute_combined_loss()` in `src/training/metrics.py` handles the
combination. When `spr_loss_tensor` is `None` (SPR disabled), the
total loss equals the TD loss with no overhead.

```python
total_loss = td_loss + spr_weight * spr_loss   # SPR enabled
total_loss = td_loss                            # SPR disabled
```

Gradients from the combined loss flow through the online encoder
(shared with Q-learning), transition model, projection head, and
prediction head. The target side receives no gradients.

<br><br>

## EMA Update Schedule

The EMA encoder updates **once per optimizer step**, immediately
after `optimizer.step()`. This happens inside `perform_update_step()`
in `src/training/metrics.py`:

```python
# Update EMA after gradient step (SPR only)
if spr_components is not None:
    spr_components["target_encoder"].update(online_net)
    spr_components["target_projection"].update(
        spr_components["projection_head"]
    )
```

The update rule for each parameter:

```text
theta_m <- tau * theta_m + (1 - tau) * theta_o
```

| Condition | `tau` (config `ema.momentum`) | Behavior |
|-----------|------|----------|
| SPR without augmentation | `0.99` | Smooth averaging over ~100 steps |
| SPR with augmentation | `0.0` | Direct copy each step (augmentation regularizes) |

Buffers (e.g., BatchNorm running stats) are copied directly, not
EMA-averaged.

The EMA encoder is independent of the DQN target network. The DQN
target network continues to use hard parameter copies every
`target_network.update_interval` steps (default 10K in base config,
2K under Atari-100K) for Q-learning stability.

<br><br>

## Sequence Sampling API

SPR requires consecutive transition sequences from the replay
buffer. The `sample_sequences()` method provides this data.

### Method signature

```python
ReplayBuffer.sample_sequences(batch_size: int, seq_len: int)
    -> Dict[str, Union[np.ndarray, torch.Tensor]]
```

### Parameters

- `batch_size` -- number of sequences to sample
- `seq_len` -- number of transitions per sequence (use `spr.prediction_steps`, default 5)

### Return value

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `states` | `(B, K+1, C, H, W)` | float32 | Observations s_t through s_{t+K} |
| `actions` | `(B, K)` | int64 | Actions a_t through a_{t+K-1} |
| `rewards` | `(B, K)` | float32 | Rewards r_t through r_{t+K-1} |
| `dones` | `(B, K)` | bool | Done flags for each transition |

### Valid starting indices

`sample_sequences()` identifies valid starting indices via
`_get_valid_sequence_starts()`. A start index `i` is valid when all
`seq_len` consecutive transitions exist in the buffer without
crossing the write pointer. If the buffer wraps around, sequences
that span the write pointer boundary are excluded.

Episode boundaries are **not** filtered at the sampling level.
Instead, the `dones` flags are returned so `compute_spr_loss()` can
apply its cumulative mask (`cumprod(1 - done)`) to zero out
predictions that cross resets.

### Usage in the training loop

The training loop samples the SPR batch alongside the standard DQN
batch inside `training_step()`:

```python
if spr_components is not None:
    seq_batch = replay_buffer.sample_sequences(
        batch_size, spr_prediction_steps
    )
    spr_batch_device = {
        "states": to_tensor(seq_batch["states"]),
        "actions": to_tensor(seq_batch["actions"]),
        "dones": to_tensor(seq_batch["dones"]),
    }
    # Augment all K+1 states in the sequence if augmentation is enabled
    if augment_fn is not None:
        s = spr_batch_device["states"]
        B, Kp1 = s.shape[0], s.shape[1]
        s_flat = augment_fn(s.reshape(B * Kp1, *s.shape[2:]))
        spr_batch_device["states"] = s_flat.reshape(B, Kp1, *s_flat.shape[1:])
```

Note that the SPR batch and the TD batch are sampled independently
-- they contain different transitions.

<br><br>

## Config Options

SPR behavior is controlled by three config sections in `base.yaml`.

### SPR section

```yaml
spr:
  enabled: false               # Toggle SPR auxiliary loss
  prediction_steps: 5          # Future steps K to predict
  loss_weight: 2.0             # Lambda for combined loss
  projection_dim: 512          # Projection head output dim
  transition_channels: 64      # Transition model conv channels
```

### EMA section

```yaml
ema:
  momentum: 0.99               # Decay coefficient tau
  # 0.99 without augmentation, 0.0 with augmentation
```

### Dropout (in network section)

```yaml
network:
  dropout: 0.0    # 0.5 without augmentation, 0.0 with augmentation
```

### Per-condition configs

Each experimental condition inherits from the per-game base config
and overrides the relevant settings:

| Condition | Config suffix | Key overrides |
|-----------|---------------|---------------|
| Vanilla DQN | (none) | defaults |
| DQN + augmentation | (none, aug in base) | `augmentation.enabled: true` |
| DQN + SPR | `_spr.yaml` | `spr.enabled: true`, `ema.momentum: 0.99`, `network.dropout: 0.5` |
| DQN + SPR + aug | `_both.yaml` | `spr.enabled: true`, `ema.momentum: 0.0`, `augmentation.enabled: true` |

Example `_spr.yaml`:

```yaml
base_config: "experiments/dqn_atari/configs/atari100k_pong.yaml"
experiment:
  name: "atari100k_pong_spr"
spr:
  enabled: true
ema:
  momentum: 0.99
network:
  dropout: 0.5
```

Example `_both.yaml`:

```yaml
base_config: "experiments/dqn_atari/configs/atari100k_pong.yaml"
experiment:
  name: "atari100k_pong_both"
spr:
  enabled: true
ema:
  momentum: 0.0
augmentation:
  enabled: true
```

<br><br>

## Checkpoint Schema Changes

When SPR is enabled, checkpoints include an additional `spr_state`
key alongside the standard fields.

### Standard checkpoint keys (always present)

```text
schema_version    "1.0.0"
timestamp         ISO-format datetime
commit_hash       Git commit hash
step              Environment step counter
episode           Episode counter
epsilon           Current exploration rate
online_model      Online network state_dict
target_model      Target network state_dict
optimizer         Optimizer state_dict
rng_states        RNG states (torch, numpy, random)
```

### SPR-specific key (present only when SPR is enabled)

```text
spr_state:
  transition_model     TransitionModel state_dict
  projection_head      ProjectionHead state_dict
  prediction_head      PredictionHead state_dict
  target_encoder       EMAEncoder state_dict (encoder + EMA copy)
  target_projection    EMAEncoder state_dict (projection + EMA copy)
```

### Backward compatibility

Loading a checkpoint without `spr_state` into an SPR-enabled run
leaves SPR components at their randomly initialized weights. The
`load_checkpoint()` return dict includes an `spr_restored` boolean
flag:

```python
result = manager.load_checkpoint(..., spr_components=components)
if not result["spr_restored"]:
    print("SPR weights not found in checkpoint; using fresh init")
```

Loading a checkpoint with `spr_state` into a vanilla DQN run (no
`spr_components` passed) silently ignores the SPR data -- no error
is raised.

<br><br>

## Toggling SPR On and Off

### Enable SPR

Set `spr.enabled: true` in the config file (or via CLI override).
The training entry point (`train_dqn.py`) reads this flag and
conditionally instantiates the SPR components:

1. Creates `TransitionModel`, `ProjectionHead`, `PredictionHead`
2. Wraps the online encoder and projection in `EMAEncoder` instances
3. Builds a single optimizer covering online network + SPR modules
4. Passes `spr_components` dict to `training_step()`

When `spr.enabled` is false (or absent), `spr_components` is `None`
throughout. All SPR code paths are guarded by `if spr_components is
not None` checks, so there is zero overhead for vanilla DQN or
DQN+augmentation conditions.

### Metrics when SPR is disabled

SPR metric fields (`spr_loss`, `cosine_similarity`, `ema_update_count`)
are `None` in `UpdateMetrics` and are omitted from logger output.
CSV columns exist but remain empty. TensorBoard and W&B skip
metrics with `None` values.

### Runtime cost

SPR adds one extra replay buffer sample (`sample_sequences`), one
forward pass through the online encoder for the initial state, K
forward passes through the transition model, K+1 forward passes
through the target encoder, and K+1 projection passes. The backward
pass runs through all online-side modules. EMA updates are O(params)
copies with no gradient computation.

<br><br>

## Logged Metrics

The `MetricsLogger.log_step()` method accepts three SPR-specific
parameters, all optional:

| Parameter | Metric key | Description |
|-----------|------------|-------------|
| `spr_loss` | `spr/loss` | SPR auxiliary loss value |
| `cosine_similarity` | `spr/cosine_similarity` | Mean cosine similarity between predictions and targets |
| `ema_update_count` | `spr/ema_update_count` | Cumulative EMA updates (equals optimizer step count) |

These appear in TensorBoard, W&B, and CSV backends when non-None.
The training loop passes them from the `UpdateMetrics` dataclass:

```python
metrics_logger.log_step(
    ...
    spr_loss=m.spr_loss if m else None,
    cosine_similarity=m.cosine_similarity if m else None,
    ema_update_count=m.update_count if m and m.spr_loss is not None else None,
)
```

<br><br>

## References

- Schwarzer et al. 2021, "Data-Efficient Reinforcement Learning with
  Self-Predictive Representations" (ICLR 2021)
- [SPR Architecture](spr-architecture.md) -- component details,
  tensor shapes, gradient flow
- [DQN Training](dqn-training.md) -- TD loss computation, target
  network updates
