# Tests

Unit and integration tests for the DQN/Rainbow/SPR implementation.
Run from the repository root with pytest.

## Running Tests

```bash
source .venv/bin/activate

# Full suite
pytest tests/ -x

# Specific file
pytest tests/test_rainbow_model.py -v

# By keyword
pytest tests/ -k "rainbow" -v
```

## Test Files

### Models

| File | Coverage |
|------|----------|
| `test_dqn_model.py` | DQN architecture, forward/backward pass, checkpointing |
| `test_rainbow_model.py` | RainbowDQN (distributional, dueling, noisy), forward pass |
| `test_noisy_linear.py` | NoisyLinear layer, noise reset, eval-mode determinism |
| `test_model_utils.py` | Model summary, parameter counting |

### Replay

| File | Coverage |
|------|----------|
| `test_replay_buffer.py` | Circular buffer, sampling, episode boundaries |
| `test_prioritized_buffer.py` | PER sampling, IS weights, beta annealing, priority updates |
| `test_sum_tree.py` | Sum-tree data structure, prefix-sum queries |
| `test_n_step.py` | N-step return computation, episode truncation |

### Training

| File | Coverage |
|------|----------|
| `test_dqn_trainer.py` | Training loop, schedulers, loss, target sync, logging |
| `test_distributional.py` | C51 categorical projection, distributional loss |
| `test_rainbow_update.py` | Rainbow update step (IS weights, priorities, C51) |
| `test_rainbow_integration.py` | Rainbow components working together |
| `test_rainbow_train_integration.py` | End-to-end Rainbow training through train_dqn.py |
| `test_rainbow_backward_compat.py` | Vanilla DQN unaffected by Rainbow additions |
| `test_evaluation.py` | Evaluation loop, metrics aggregation |
| `test_metrics_logger.py` | TensorBoard/W&B/CSV logging, Rainbow metrics |
| `test_plot_results.py` | Plotting script, CSV loading, smoothing |
| `test_video_recorder.py` | Video capture during evaluation |

### SPR

| File | Coverage |
|------|----------|
| `test_spr_components.py` | Transition model, projection/prediction heads, EMA |
| `test_spr_integration.py` | SPR loss, sequence sampling, training integration |
| `test_augmentation.py` | DrQ random-shift augmentation |

### Infrastructure

| File | Coverage |
|------|----------|
| `test_config_loader.py` | Config loading, merging, CLI overrides, Rainbow config parsing |
| `test_schema_validator.py` | Config schema validation |
| `test_cli.py` | Command-line argument parsing |
| `test_run_manager.py` | Run directory creation and metadata |
| `test_checkpoint.py` | Checkpoint save/load, atomic writes, best model |
| `test_resume.py` | Resume from checkpoint, config validation, RNG restoration |
| `test_seeding.py` | Deterministic seeding across Python/NumPy/PyTorch |
| `test_determinism.py` | Determinism flags, cuDNN configuration |
| `test_save_resume_determinism.py` | End-to-end save/resume determinism smoke test |

### Scripts

| File | Coverage |
|------|----------|
| `test_eval_checkpoints.py` | Checkpoint re-evaluation: discovery, model creation, CSV output |
