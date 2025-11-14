# Design Notes

Use this directory to capture architectural sketches, module dependency diagrams, and experiment orchestration plans. As the DQN scaffold solidifies, document:

- Shared module responsibilities (`src/agents`, `src/envs`, etc.).
- Configuration strategy (Hydra/OmegaConf layouts, default overrides).
- Experiment lifecycle (training workflow, evaluation workflow, reporting pipeline).
- Foundational setup references (e.g., `dqn_setup.md` for Subtask 1 outputs).

Keeping design decisions explicit here makes it easier to evolve the codebase while maintaining reproducibility across experiments.

## Available Design Documents

- [DQN Setup](dqn_setup.md) - Environment setup, game selection, and reproducibility
- [Atari Environment Wrapper](atari_env_wrapper.md) - Preprocessing, frame stacking, reward clipping
- [DQN Model](dqn_model.md) - CNN architecture and weight initialization
- [Replay Buffer](replay_buffer.md) - Circular buffer with uniform sampling
- [DQN Training](dqn_training.md) - TD targets, loss computation, and optimization
- [Training Loop Runtime](training_loop_runtime.md) - Main training loop orchestration
- [Episode Handling](episode_handling.md) - Life-loss vs full-episode termination
- [Checkpointing](checkpointing.md) - Save/resume and deterministic seeding
- [Config & CLI](config_cli.md) - Configuration system and command-line interface
- [Evaluation Harness](eval_harness.md) - Periodic evaluation, metrics, and video capture
- [Logging & Plotting Pipeline](logging_pipeline.md) - Multi-backend logging, plotting, and results analysis
