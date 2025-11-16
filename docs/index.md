# Documentation Index

Complete reference for DQN Atari implementation. Use this index to navigate design specifications, implementation guides, and configuration references.

## Quick Navigation

- **New to the project?** Start with [Architecture Overview](architecture.md) → [Quick Start Guide](quick_start.md) → [Roadmap](roadmap.md)
- **Need to do something?** Check [Workflows](workflows.md) for task-oriented guides
- **Stuck on an issue?** See [Troubleshooting](troubleshooting.md) for quick problem resolution
- **Setting up environment?** See [DQN Setup](design/dqn_setup.md)
- **Implementing a component?** Jump to relevant design doc below
- **Running experiments?** See [Quick Start Guide](quick_start.md) or [Experiment README](../experiments/dqn_atari/README.md)
- **Tracking progress?** Check [Changelog](changelog.md) and [Roadmap](roadmap.md)

---

## Getting Started

### [Architecture Overview](architecture.md)
High-level system design and component interactions (READ THIS FIRST).

**What's inside:**
- System component diagram (Environment, Q-Network, Replay, Training Loop, Checkpointing)
- Complete data flow walkthrough for single training step
- Component details with memory layout and tensor shapes
- Key algorithms (Q-learning update, epsilon-greedy exploration)
- Directory structure and file organization
- Execution flow summary

**When to read:** First stop for new contributors, before diving into detailed design docs

---

### [Common Workflows](workflows.md)
Task-oriented guides for frequent operations.

**What's inside:**
- First-time setup and verification
- Train from scratch and resume from checkpoint
- Monitor training progress in real-time
- Debug unstable training and verify determinism
- Run unit tests and smoke tests
- Inspect checkpoint contents

**When to read:** When you need to DO something (setup, train, debug, test)

---

### [Troubleshooting Guide](troubleshooting.md)
Quick reference for diagnosing and fixing common issues.

**What's inside:**
- Setup issues (ImportError, ROM not found, CUDA unavailable)
- Training issues (NaN loss, not decreasing, divergence, non-determinism)
- Performance issues (too slow, out of memory, high CPU)
- Checkpoint/resume issues (config mismatch, commit warnings, RNG states)
- Environment issues (frame shapes, action space, termination)

**When to read:** When something is broken or not working as expected

---

## Design Documentation

Core implementation specifications for DQN components.

### [DQN Setup](design/dqn_setup.md)
Environment setup, dependencies, and ROM installation guide.

**What's inside:**
- Python environment creation and package installation
- Atari ROM download via AutoROM
- System requirements and compatibility notes
- Troubleshooting common setup issues

**When to read:** First-time setup, dependency issues, new machine setup

---

### [Atari Environment Wrapper](design/atari_env_wrapper.md)
Comprehensive wrapper chain specification for Atari preprocessing.

**What's inside:**
- Complete wrapper chain (NoopReset → MaxAndSkip → EpisodeLife → RewardClipper → AtariPreprocessing → FrameStack)
- Expected tensor shapes throughout pipeline
- Episode life termination policy (training vs evaluation)
- Debug artifacts and dry-run validation
- Configuration toggles and troubleshooting

**When to read:** Implementing preprocessing, debugging wrapper issues, validating frame shapes

---

### [DQN Model](design/dqn_model.md)
Q-network architecture and forward pass specification.

**What's inside:**
- DQN 2013 CNN architecture (2 conv layers + 2 FC layers)
- Layer specifications (filters, kernels, strides, activations)
- Forward pass implementation and output format
- Weight initialization strategies
- Testing and validation procedures

**When to read:** Implementing Q-network, debugging model architecture, verifying output shapes

---

### [Replay Buffer](design/replay_buffer.md)
Experience replay storage and sampling specification.

**What's inside:**
- Circular buffer implementation with uint8 storage
- Sampling strategies (uniform random, no recency bias)
- Memory efficiency optimizations
- Batch preparation and tensor conversion
- Thread safety and performance considerations

**When to read:** Implementing replay buffer, optimizing memory usage, debugging sampling

---

### [DQN Training](design/dqn_training.md)
Q-learning update flow, loss functions, and training pipeline.

**What's inside:**
- Complete update pipeline (TD targets → loss → backward → clip → step)
- TD target computation with Bellman equation
- Loss functions (MSE and Huber) with selection guidance
- Optimizer configuration (RMSprop and Adam)
- Target network synchronization (2013 vs 2015 DQN)
- Training frequency scheduling
- Metrics logging and reference-state Q tracking
- Debugging unstable training (exploding TD error, stale targets, NaN/Inf)

**When to read:** Implementing training loop, debugging convergence issues, understanding Q-learning

---

### [Episode Handling](design/episode_handling.md)
Episode management, termination policies, and training vs. evaluation modes.

**What's inside:**
- Episode termination policies (terminated vs. truncated)
- Life-loss as terminal during training (EpisodicLifeEnv wrapper)
- Full episodes during evaluation (no life-loss wrapper)
- No-op starts for stochastic initial states
- Episode tracking and metrics (return, length, rolling averages)
- Training vs. evaluation environment differences
- Configuration and common pitfalls

**When to read:** Implementing training/eval loops, understanding episode resets, debugging episode metrics

---

### [Training Loop Runtime](design/training_loop_runtime.md)
Complete training loop orchestration, component integration, and runtime behavior.

**What's inside:**
- High-level control flow (action select → env step → replay → optimize → eval)
- Component orchestration (10+ schedulers, loggers, trackers)
- Logging schema (CSV formats, directory structure, plotting examples)
- Evaluation cadence (when/how to evaluate, interpreting results)
- Command reference (training, dry-run, smoke test, monitoring)
- Smoke test procedure (validation checklist, expected output)
- Troubleshooting guide (common issues, diagnosis, fixes)
- Configuration knobs (epsilon, intervals, logging frequency)

**When to read:** Running training, debugging training issues, understanding component interaction, monitoring progress

---

## Configuration Reference

### [Config Documentation](../experiments/dqn_atari/configs/README.md)
Complete guide to OmegaConf configuration system.

**What's inside:**
- OmegaConf basics (interpolation, hierarchical defaults, CLI overrides)
- All config sections documented (env, training, agent, exploration, eval)
- Toggle explanations (episode_life, target_update_interval, reward_clip)
- Common workflows and parameter override examples

**When to read:** Creating new configs, understanding config keys, overriding parameters

---

### [Script Documentation](../experiments/dqn_atari/scripts/README.md)
CLI scripts for training, setup, and debugging.

**What's inside:**
- `run_dqn.sh`: Training and dry-run with all flags
- `setup_roms.sh`: ROM installation and verification
- `capture_env.sh`: System info capture
- Common workflows (first-time setup, Subtasks 1-2 reproduction, debugging)

**When to read:** Running training, setting up ROMs, capturing environment state

---

## Project Management

### [Roadmap](roadmap.md)
Complete project plan with 21 subtasks and progress tracking.

**What's inside:**
- 6 major sections (Setup, Env, Model, Replay, Training, Loop)
- Detailed checklists for each subtask
- Current progress and completion status
- Verification steps for each component

**When to read:** Planning work, checking progress, understanding project scope

---

### [Changelog](changelog.md)
Timeline of major subtask completions and documentation updates.

**What's inside:**
- Chronological record of completed subtasks
- Design doc and config updates
- Major implementation milestones
- Git commit references

**When to read:** Understanding project history, tracking context shifts, onboarding

---

### [Quick Start Guide](quick_start.md)
End-to-end workflow from setup to training.

**What's inside:**
- Step-by-step setup instructions
- ROM download and environment validation
- Running dry runs and full training
- Evaluation and plotting commands
- Troubleshooting common issues

**When to read:** Getting started, running first experiment, end-to-end workflow

---

## Additional Resources

### Experiment-Specific Documentation
- [DQN Atari Experiment README](../experiments/dqn_atari/README.md) – Game selection, ROM setup, experiment layout
- [Notes](../experiments/dqn_atari/notes.md) – Chronological experiment log and observations

### Code Documentation
- [src/models/](../src/models/) – Neural network implementations
- [src/replay/](../src/replay/) – Replay buffer implementations
- [src/envs/](../src/envs/) – Atari wrapper implementations
- [src/training/](../src/training/) – DQN trainer and update logic
- [tests/](../tests/) – Unit tests for all modules

### External References
- [DQN Paper (2013)](https://arxiv.org/abs/1312.5602) – Mnih et al., "Playing Atari with Deep Reinforcement Learning"
- [Nature DQN (2015)](https://www.nature.com/articles/nature14236) – Mnih et al., "Human-level control through deep reinforcement learning"
- [Gymnasium ALE](https://gymnasium.farama.org/environments/atari/) – Atari Learning Environment documentation

---

## Document Maintenance

**Last updated:** 2025-11-16

**How to update this index:**
1. Add new design docs to appropriate section
2. Update one-line summaries when major changes occur
3. Keep "When to read" focused on user intent
4. Update changelog when adding new documents

**Contributing:**
- Keep summaries concise (1-2 sentences)
- Organize by user workflow (setup → implement → run → debug)
- Cross-reference related documents
- Update changelog when making major changes
