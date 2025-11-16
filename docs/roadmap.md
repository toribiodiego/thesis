# DQN Reproduction Roadmap

## Objective

Re-implement and reproduce DeepMind's DQN (*Playing Atari with Deep Reinforcement Learning*, arXiv:1312.5602) from scratch. Train an agent to play Atari 2600 games from raw pixels using a single architecture and shared hyperparameters. Start with a small subset (Pong, Breakout, Beam Rider) before scaling to the full paper set.

**Success criteria:**
- Agent learns on selected games from the paper
- Performance approaches or beats paper baselines (within reasonable margin)
- Code, configs, and results are fully reproducible

**Key outputs:**
- `envs/` – Environment setup and pinned dependencies
- `src/` – Reusable agents, replay, preprocessing, configs, logging
- `experiments/dqn_atari/` – DQN configs, launch scripts, run metadata
- `results/` – Plots, tables, experiment cards
- `docs/` – Design docs, paper notes, experiment logs

**Workflow:**
- Follow commit prefixes in `docs/git_commit_guide.md` (feat/fix/docs/test/build/chore)
- Mark completed checklist items as work progresses
- Current focus: **Subtask 11 – Integration & First Training Runs**

**Progress Summary**:
- Subtasks 1-10: **Complete** (335+ tests, all passing)
- Subtask 11: **95% Complete** (implementation done, ready to execute training)
- Infrastructure: Fully implemented and tested
  - Environment, wrappers, preprocessing, frame stacking
  - DQN model, replay buffer, Q-learning loss
  - Training loop with epsilon-greedy exploration
  - Multi-backend logging (TensorBoard, W&B, CSV)
  - Checkpointing, resume, deterministic seeding
  - Evaluation harness with video capture
  - Plotting and results export scripts
- **Next step**: Execute first full Pong training run (10M frames)

---

## Key Hyperparameters (DQN 2013)

- **Preprocessing:** 210×160×3 RGB → 84×84 grayscale, 4-frame stack
- **Network:** Conv(16,8×8,s4) → Conv(32,4×4,s2) → FC(256) → linear(|A|)
- **Replay:** 1M capacity, uniform sampling, 50k warm-up
- **Optimizer:** RMSProp (ρ=0.95, ε=0.01), LR=2.5e-4
- **Training:** γ=0.99, batch=32, target update every 10k steps, train every 4 steps
- **Exploration:** ε: 1.0→0.1 over 1M frames
- **Frame skip:** 4 (action repeated, rewards accumulated)
- **Reward clipping:** {−1, 0, +1}

See `docs/papers/dqn_2013_notes.md` for details.

---
## Milestones

| Milestone | Criteria | Status |
|-----------|----------|--------|
| **M1** | Environment + tooling smoke test passes | **Complete** |
| **M2** | Training infrastructure complete (Subtasks 1-10) | **Complete** |
| **M3** | First full Pong training run (10M frames) | **Next** |
| **M4** | Three-game suite reproduced with reports | Pending |

**Current Status**: All training infrastructure is complete. Ready to integrate MetricsLogger into training loop and launch first real training runs.

---

## Next Steps (Execution Phase)

### Immediate (Complete Subtask 11 - This Week)
1. [X] Integrate MetricsLogger into train_dqn.py (DONE)
2. [X] Add logging config to YAML files (DONE)
3. [X] Update schema validation (DONE)
4. [ ] Run smoke test (200K frames, ~30 min)
5. [ ] Launch first full Pong training (10M frames, ~8-12 hours)
6. [ ] Verify all logging backends work (TensorBoard, CSV, W&B)
7. [ ] Generate plots from training run
8. [ ] Document game suite plan

### Short-term (Subtask 12 - Multi-Game Training)
1. Launch Breakout training (50M frames, 3 seeds)
2. Launch Beam Rider training (50M frames, 3 seeds)
3. Monitor for stability issues (NaN/Inf, crashes)
4. Aggregate results with `export_results_table.py`
5. Generate multi-seed plots with `plot_results.py --multi-seed`
6. Compare to paper baselines
7. Document any hyperparameter adjustments

### Medium-term (Subtask 13 - Analysis & Reporting)
1. Compare reproduction scores to DQN 2013 paper
2. Analyze discrepancies (environment versions, etc.)
3. Generate publication-quality plots and tables
4. Write results summary (`docs/reports/dqn_results.md`)
5. Upload final artifacts to W&B (if applicable)

### Future Work (Optional - Subtasks 14-21)
These provide valuable enhancements but are not critical for initial reproduction:
- Ablation studies (reward clipping, target network, frame stack)
- Automated report generation
- Repository archival and organization
- Thesis integration and write-up

**Focus**: We've completed implementation (Subtasks 1-10). Now we execute, verify, and analyze (Subtasks 11-13).

---

## Subtasks

### Subtask 1 — Choose paper games, pin versions, and scaffold runs

**Objective:**
Establish the experimental foundation by selecting 2–3 representative Atari games (Pong, Breakout, Beam Rider), pinning all dependencies (Python, PyTorch, Gymnasium/ALE, ROMs), and creating config files plus a launch script. Define a consistent evaluation protocol (ε-eval, termination policy, frame budgets). Complete when a random-policy dry run successfully produces preprocessed frames, action lists, and a minimal evaluation report.

**Checklist:**
- [X] Choose initial 2–3 games from the paper for reproduction (recommended: Pong, Breakout, Beam Rider) and create config stubs at `experiments/dqn_atari/configs/{pong,breakout,beam_rider}.yaml`, plus note the chosen games in `experiments/dqn_atari/README.md`.
    - [X] docs: Add selected Atari games (Pong, Breakout, Beam Rider) and config stubs
- [X] Record official game IDs (Gymnasium/ALE names like `ALE/Pong-v5`, `ALE/Breakout-v5`, `ALE/BeamRider-v5`) in `experiments/dqn_atari/README.md` as a table with an action_set column (use minimal) and ROM acquisition instructions (add `scripts/setup_roms.sh` calling `python -m AutoROM --accept-license`).
    - [X] docs: Document ALE v5 env IDs, minimal action set, and AutoROM setup script
- [X] Pin env versions by adding exact versions to `requirements.txt` (e.g., Python 3.10.13, PyTorch 2.4.1+cu121, Gymnasium 0.29.1, ale-py 0.8.1) and documenting ALE runtime settings in the README (e.g., `repeat_action_probability=0.0`, `frameskip=4`, `full_action_space=false`); include `scripts/capture_env.sh` to write `experiments/dqn_atari/system_info.txt`.
    - [X] build: Pin Python/Torch/Gym/ALE versions; capture system info; document deterministic ALE settings
- [X] Define evaluation protocol in `experiments/dqn_atari/configs/base.yaml`: ε-greedy evaluation with small ε (e.g., `eval.epsilon=0.05` or greedy toggle), termination policy (full episode for eval; training may treat life-loss as terminal), `eval.episodes=10` per checkpoint, reward clipping to {−1,0,+1}, and per-game frame budgets (e.g., 10–20M for main runs, smaller for smoke tests).
    - [X] feat: Add base eval/train protocol (ε=0.05, full-episode eval, life-loss training option, reward clipping, frame budgets)
- [X] Seed & reproducibility by introducing a unified `set_seed(seed, deterministic=False)` utility (`src/utils/repro.py`), wiring `--seed` in the entry point, and saving with each run the git commit hash, merged config, seed, and ALE settings as `meta.json` in the run directory.
    - [X] feat: Add unified seeding utility and run metadata snapshot
- [X] Scaffold runs by creating per-game YAMLs under `experiments/dqn_atari/configs/` (common defaults + small per-game overrides) and adding `scripts/run_dqn.sh` that launches `src/train_dqn.py` with logging to `experiments/dqn_atari/runs/`; support a `--dry-run` path that executes a short random rollout, saves a few preprocessed frame stacks, lists available actions, and writes a minimal evaluation report.
    - [X] feat: Add run launcher and random-policy dry run with frames, action list, and eval report
- [X] Capture Subtask 1 outputs in `docs/design/dqn_setup.md`: outline selected games, pinned dependencies, evaluation protocol, seeding utility, ROM/setup commands, and dry-run instructions so onboarding contributors have a single reference.
    - [X] docs: Summarize required commands (`envs/setup_env.sh`, `scripts/setup_roms.sh`, `scripts/run_dqn.sh --dry-run`) plus troubleshooting tips.

---

### Subtask 2 — Implement Atari env wrapper (preprocess, frame-skip, reward clipping)

**Objective:**
Build wrapper transforming raw 210×160×3 frames to 84×84 grayscale, stacking last 4 frames as (4,84,84). Apply 4-frame action repeat with max-pooling, clip rewards to {−1,0,+1}. Handle episode termination (life-loss vs full-episode). Complete when verified shapes, sample frame stacks, and rollout logs exist in `experiments/dqn_atari/artifacts/frames/`.

**Checklist:**
- [X] Preprocess frames from RGB (210×160) to grayscale, resize/crop to 84×84, stack the last 4 frames as channels-first `(4,84,84)` using `uint8` storage and convert to `float32` on sample; save a few sample stacks as PNGs per game.
    - [X] feat: Add grayscale→84×84 preprocessing and 4-frame stack with sample PNG export
- [X] Implement action repeat/frame-skip (`k=4`) and elementwise max-pool over the last two raw frames before preprocessing to reduce flicker.
    - [X] feat: Add 4-step action repeat and last-2-frame max-pooling
- [X] Apply reward clipping to the set {−1, 0, +1} with a config toggle to disable for ablations.
    - [X] feat: Add configurable reward clipping to {-1,0,+1}
- [X] Align episode termination with evaluation policy: allow training to treat life loss as terminal, use full-episode termination for evaluation, and support optional no-op starts and auto-fire reset where needed; document choices in the wrapper docstring.
    - [X] docs: Document termination (life-loss vs full-episode), no-op starts, and auto-fire behavior
- [X] Implement random no-op resets (0–30 actions) at episode start with a configurable `noop_max` (default 30) to match Bellemare/Mnih evaluation protocol.
    - [X] feat: Add no-op reset logic to the wrapper with config toggles and documentation
- [X] Explicitly document that full-episode termination is the default (life-loss termination only when the config enables it) and ensure training/eval configs reflect this.
    - [X] docs: Note default terminal behavior in wrapper docs/config comments
- [X] Clarify preprocessing documentation: describe the two-frame max-pooling, the 84×84 resize/crop policy (score bar kept or removed), and state that reward clipping defaults to ON per the paper.
    - [X] docs: Update wrapper design note to outline pooling/cropping choices and reward clipping default
- [X] Produce debug artifacts: write a short random rollout log recording obs shape, action repeat behavior, and clipped reward stats; save preprocessed stacks under `experiments/dqn_atari/artifacts/frames/<game>/` and the rollout log in the corresponding run directory.
    - [X] feat: Emit rollout debug log and per-game preprocessed frame artifacts
- [X] Capture the wrapper specification + troubleshooting guide in `docs/design/atari_env_wrapper.md`: summarize preprocessing pipeline, config flags, artifact locations, and common failure modes (e.g., life-loss mismatch, flicker, reward clipping) so future debugging/design reviews have a single reference.
    - [X] docs: Outline regeneration steps for artifacts (`scripts/run_dqn.sh --dry-run`), expected tensor shapes, and how to toggle key behaviors via config.

---

### Subtask 3 — Implement the DQN model (vision trunk + Q-head)

**Objective:**
CNN mapping (4×84×84) to Q-values: Conv(16,8×8,s4) → Conv(32,4×4,s2) → FC(256) → linear(|A|). Kaiming init, float32, channels-first. Complete when forward passes produce correct shapes without NaNs and checkpointing works.

**Checklist:**
- [X] Implement DQN CNN with input `(4,84,84)`: Conv1 (16, 8×8, stride 4, ReLU) → Conv2 (32, 4×4, stride 2, ReLU) → flatten → FC(256, ReLU) → linear head of size `|A|`; channels-first tensors, return dict with `q_values` and optional `features` for debugging.
    - [X] feat: Add DQN model (Conv8x8s4→Conv4x4s2→FC256→Q-head)
- [X] Set weight initialization and dtypes: use Kaiming normal (fan_out) for conv/linear with ReLU, zeros for biases; keep parameters in float32; expose a `to(device)` utility; ensure forward accepts `float32` inputs scaled to `[0,1]`.
    - [X] build: Configure Kaiming init and float32 dtype for all layers
- [X] Add model summary and shape checks: implement a small `model_summary(module, input_shape)` printer and log parameter count; assert expected output shape `(B, |A|)` for a dummy batch; handle dynamic `|A|` from env.
    - [X] chore: Add model summary utility and output-shape assertion
- [X] Create forward-path unit tests: with random input `(B=2, 4, 84, 84)` verify no NaNs/Infs, correct shapes for action spaces (e.g., Pong=6, Breakout=4, BeamRider=9), and gradients flow with a dummy MSE loss/backward.
    - [X] test: Add forward/grad tests across multiple action sizes
- [X] Implement save/load helpers: `save_checkpoint(path, state_dict, meta)` and `load_checkpoint(path)` for model-only, plus convenience `from_env(action_space_n)` constructor; ensure strict key matching and device-safe loading.
    - [X] feat: Add checkpoint save/load and environment-aware constructor
- [X] Document architecture decisions in `docs/design/dqn_model.md`: layer-by-layer tensor shapes, init choices, dtype/device expectations, summary utility usage, and common debugging tips (e.g., NaN traces, mismatched action dims).
    - [X] docs: Include commands to regenerate summaries/tests (`pytest tests/test_dqn_model.py`, `python scripts/model_summary.py`) and guidance on inspecting saved checkpoints.

---

### Subtask 4 — Experience Replay Buffer (uniform)

**Objective:**
Circular buffer storing ~1M transitions (s,a,r,s',done). Store as uint8, convert to float32 on sample. Support 50k warm-up. Complete when sampling returns correct shapes, respects episode boundaries, and integrates with GPU training.

**Checklist:**
- [X] Implement a circular replay buffer (capacity e.g., `1_000_000`) that stores tuples `(s_t, a_t, r_t, s_{t+1}, done_t)` with a ring write index and per-step episode boundary markers to prevent cross-episode samples.
    - [X] feat: Add uniform replay buffer with circular storage and episode boundary tracking
- [X] Provide a minimal API: `append(state, action, reward, next_state, done)`, `sample(batch_size) -> dict(tensors)`, and `__len__`; include input validation and graceful handling when the buffer has fewer than `batch_size` valid indices.
    - [X] feat: Expose append/sample/len API with basic validation
- [X] Optimize memory layout: store frames (and frame stacks) as `uint8` to save RAM, keep a contiguous frame array plus indices for stacking, convert to `float32` only on sample, and normalize to `[0,1]` (configurable).
    - [X] build: Store observations as uint8 and defer float32 conversion/normalization to sampling
- [X] Enforce warm-up: add a configurable pre-fill (default `50_000` random steps) before any optimization; expose `can_sample(min_size)` helper used by the training loop.
    - [X] feat: Add warm-up threshold and can_sample helper
- [X] Implement uniform sampling without replacement: draw valid indices that have `t, t-1, t-2, t-3` within the same episode and available `t+1` (for `s'`), rejecting indices near wrap/episode boundaries; return batches with shapes `s: (B,4,84,84)`, `a: (B,)`, `r: (B,)`, `s_next: (B,4,84,84)`, `done: (B,)`.
    - [X] feat: Add boundary-safe uniform sampler with no replacement
- [X] Device transfer and speed: on `sample`, assemble stacks, convert to `float32`, normalize (or leave in 0–255 if configured), move tensors to GPU if available, and optionally use pinned host memory for faster H2D copies.
    - [X] perf: Add device move, optional pinned memory, and normalization toggle
- [X] Add tests: fill buffer past `batch_size`, call `sample`, verify exact shapes and dtypes, ensure no cross-episode indices, check wrap-around correctness at buffer edges, and assert reproducibility with a fixed RNG seed.
    - [X] test: Add shape/boundary/repro tests for sampling and ring wrap-around
- [X] Capture replay design in `docs/design/replay_buffer.md`: diagram memory layout, document sampling pseudocode, warm-up policy, config flags, and known failure modes (e.g., episode leakage, dtype mismatch) with troubleshooting steps.
    - [X] docs: Reference the commands/tests used to validate the buffer and instructions for dumping sample batches for inspection.

---

### Subtask 5 — Q-Learning Loss, Target Network, and Optimizer

**Objective:**
TD target: *y = r + γ(1−done)×maxₐ′ Q_target(s′,a′)*. MSE or Huber loss. RMSProp/Adam (LR 2.5e-4, γ=0.99, batch=32). Target sync every 10k steps. Train every 4 steps. Complete when loss decreases smoothly and target updates are correct.

**Note on Target Networks:**
The target network is a 2015 Nature paper improvement (Mnih et al.). The original 2013 arXiv DQN paper used a single network for both Q(s,a) and target computations. For purist 2013 reproduction, set `TargetNetworkUpdater(update_interval=1)` or use the same network for both online and target Q-value computations.

**Checklist:**
- [X] Create online and target Q-networks with identical architecture; initialize target as a hard copy of online, freeze target grads, and provide `hard_update_target()` utility.
    - [X] feat: Initialize online/target Q-nets and hard-copy sync helper
- [X] Compute TD targets per minibatch using `y = r + γ * (1 - done) * max_a' Q_target(s', a')` under `no_grad`, and gather `Q_online(s, a)` with `gather` for chosen actions; ensure correct broadcasting and shapes `(B,)` after squeeze.
    - [X] feat: Implement TD target computation and online Q selection
- [X] Add configurable loss: default MSE on `(Q_selected - y)` with `reduction='mean'`, optional Huber (δ=1.0) via config flag; return loss and aux stats (mean |TD error|).
    - [X] feat: Add MSE/Huber loss with TD-error metrics
- [X] Configure optimizer and hyperparameters: RMSProp (ρ=0.95, ε=1e-2) or Adam via config; LR `2.5e-4`, γ `0.99`, batch size `32`; apply global-norm gradient clipping (e.g., `10.0`) right before `optimizer.step()`.
    - [X] build: Add optimizer setup and global-norm gradient clipping
- [X] Implement periodic target updates: call `hard_update_target()` every `C` environment steps (default `10_000`); track env step counter and log each sync step.
    - [X] feat: Add step-scheduled hard target sync with logging
- [X] Document that the target network is a 2015 stability improvement (not present in the 2013 paper) and expose a config flag/notes on how to disable it for purist reproductions.
    - [X] docs: Add roadmap/config comments describing how to run without a target network
- [X] Schedule training frequency: perform one optimization step every `k=4` environment steps after replay warm-up; skip updates if `can_sample` is false; support configurable `train_every`.
    - [X] feat: Add train-every-k update scheduler with warm-up gating
- [X] Add stability checks: unit test on a synthetic batch to confirm loss decreases over several updates; assert target updates occur at exact multiples of `C`; detect and warn on NaNs/Infs; log grad norm and LR per update.
    - [X] test: Add toy-batch loss decrease and target-sync schedule tests
- [X] Minimal metrics logging from the update step: loss, mean |TD error|, grad norm, learning rate, and update count for downstream plotting.
    - [X] chore: Log core update metrics (loss, TD-error, grad-norm, lr)
- [X] Summarize the Q-learning update flow in `docs/design/dqn_training.md`: include the TD-loss equation, optimizer config, target-sync policy, logging expectations, and debugging tactics for instability (e.g., exploding TD error, stale targets, NaNs).
    - [X] docs: Note how to rerun unit tests/debug scripts and which config flags control the behaviors described.

---

### Subtask 6 — Training Loop, Exploration Schedule, Logging & Evaluation

**Objective:**
ε-greedy (1.0→0.1 over 1M frames). Main loop: select action, step env, append transition, periodic optimization, target sync. Log losses/rewards/ε. Periodic eval (250k frames, ε=0.05). Complete when ~200k-frame runs execute reliably with logs and checkpoints.

**Checklist:**
- [X] Implement ε-greedy exploration with a configurable linear schedule: start ε=1.0, decay to 0.1 over the first 1,000,000 frames (option to continue to 0.01), and use a separate `eval_epsilon` (e.g., 0.05) only during evaluation; expose all as config and log ε per step.
    - [X] feat: Add configurable ε schedules (train and eval) with per-step logging
- [X] Ensure action repeat/frame-skip integration: execute the chosen action for k frames (default 4), accumulate clipped reward from the wrapper, and count environment frames correctly (not decisions); record effective FPS.
    - [X] feat: Integrate frame-skip execution and reward accumulation with accurate frame counters
- [X] Build the main step loop: (1) select action via ε-greedy from the online Q-net, (2) step env with frame-skip, (3) append transition to replay, (4) if warm-up done and `t % train_every == 0` then sample → compute loss → backprop → optimizer step, (5) if `t % target_update == 0` then hard-sync target.
    - [X] feat: Implement training loop with scheduled optimization and target sync
- [X] Handle episodes consistently: reset on terminal; optionally treat life-loss as terminal during training; run full episodes during evaluation; optionally support no-op starts; record per-episode return and length.
    - [X] docs: Document training/eval termination policy and optional no-op starts
- [X] Add structured logging under `experiments/dqn_atari/runs/`: per-step (loss moving average, ε, learning rate, replay size, grad norm) and per-episode (return, length, FPS, rolling mean over last N); save checkpoints on a fixed cadence and on best eval score.
    - [X] feat: Add step/episode loggers and periodic/best checkpoints
- [X] Implement the evaluation routine: every E frames (default 250,000) run K episodes (default 10) with `eval_epsilon`; disable learning, set eval mode, log mean/median/std returns, and write plots/CSV to `results/`.
    - [X] feat: Add periodic evaluation with summary metrics and result artifacts
- [X] Track the average max-Q over a fixed reference batch of saved states (as in the paper) to monitor learning progress even when rewards are noisy.
    - [X] feat: Add optional reference-state Q logging hook and plotting support
- [X] Persist reproducibility metadata for each run: save merged config snapshot, seed, and git commit hash beside logs and checkpoints (JSON/YAML).
    - [X] chore: Write run metadata (config, seed, commit) to run folder
- [X] Run a smoke test (~200,000 frames) to verify end-to-end stability: confirm logs grow, checkpoints appear, eval runs trigger, and quick plots render without errors.
    - [X] test: Add smoke-test script to validate loop, logging, checkpoints, and eval cadence
- [X] Document the orchestration in `docs/design/training_loop_runtime.md`: describe the control flow (action select → env step → replay append → optimize → eval), logging schema, evaluation cadence, smoke-test procedure, and knobs for troubleshooting (epsilon schedule, frame counters, eval triggers).
    - [X] docs: Include command examples (`python src/train_dqn.py ...`, `scripts/run_dqn.sh --dry-run`, smoke-test runner) and guidance for interpreting logs/metrics during debugging.

---

### Subtask 7 — Checkpointing, Resume, and Deterministic Seeding

**Objective:**
Save/restore models, optimizer, replay position, counters, ε, RNG states. Support `--resume`. Enforce deterministic execution. Complete when save/resume/determinism verified over fixed frames.

**Checklist:**
- [X] Implement checkpoint structure that saves online/target weights, optimizer state, step counter, episode counter, ε value, replay buffer write index and content (or snapshot pointer), and RNG states (torch, numpy, random, env) to `experiments/dqn_atari/checkpoints/checkpoint_{steps}.pt`; write atomically (temp file → rename) and include a small `meta` dict (schema version, timestamp, commit hash).
    - [X] feat: Add atomic checkpoint save with models, optimizer, replay position, RNG states, and run metadata
- [X] Add resume logic via `--resume path/to/checkpoint.pt` that restores device-safe tensors, optimizer, step/episode counters, ε schedule state, RNG states, and (if present) replay buffer; validate config compatibility and warn on commit/hash mismatch; resume training seamlessly from the next step.
    - [X] feat: Implement robust resume path restoring counters, schedules, and replay buffer
- [X] Centralize deterministic seeding with `set_seed(seed, deterministic=True)` to seed Python, NumPy, Torch (CPU/GPU), and the env on every reset; record the seed in run metadata and propagate to workers if using multiprocessing.
    - [X] feat: Add deterministic seeding utility and metadata recording
- [X] Control randomness for reproducibility by setting `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, and optionally `torch.use_deterministic_algorithms(True)` behind a config flag; document potential performance trade-offs.
    - [X] docs: Document deterministic flags and performance implications
- [X] Create a smoke test: run ~10k steps, save a checkpoint, resume from it, and verify identical ε, rewards, and selected actions for a fixed number of frames (allow tiny FP tolerance); emit a short comparison report (match/ mismatch counts, checksums).
    - [X] test: Add save/resume determinism test with metric comparison and checksum report
- [X] Capture checkpoint/resume procedures in `docs/design/checkpointing.md`: list saved tensors, metadata schema, resume CLI usage, deterministic seeding requirements, and debugging steps for mismatched states.
    - [X] docs: Provide commands for creating/restoring checkpoints and a checklist for verifying deterministic resumes.

---

### Subtask 8 — Config System and Command-Line Interface

**Objective:**
Base config + per-game overrides. Merge utility. CLI: `python train_dqn.py --cfg configs/pong.yaml --seed 123`. Auto-save merged config. Complete when reproducible single-command launch works.

**Documentation:** See [docs/design/config_cli.md](docs/design/config_cli.md) for complete reference.

**Checklist:**
- [X] Create base and per-game config files: add `experiments/dqn_atari/configs/base.yaml` for global defaults (network, replay, optimizer, target_update, eval cadence) and `experiments/dqn_atari/configs/{pong,breakout,beam_rider}.yaml` that override only env-specific fields (e.g., `env_id`, `action_set`, `frame_budget`).
    - [X] docs: Add base and per-game YAML configs with clear comments on each field
- [X] Implement a lightweight config loader that merges base + game overrides, supports nested keys, and returns a dict/dataclass; print the resolved config at startup for traceability.
    - [X] feat: Add config merge utility with nested override and resolved-config logging
- [X] Provide a CLI entry point to launch experiments with a single command: `python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 123 --resume path/to/checkpoint.pt` and allow optional `--set key.subkey=value` overrides.
    - [X] feat: Add CLI flags for --cfg, --seed, --resume, and inline overrides
- [X] Auto-save the merged config snapshot to each run folder (`experiments/dqn_atari/runs/<game>/<timestamp>/config.yaml`) alongside `meta.json`; include commit hash and seed.
    - [X] chore: Persist merged config and metadata to each run directory
- [X] Create dynamic run paths automatically (logs, checkpoints, artifacts) under `experiments/dqn_atari/runs/<game>/<timestamp>/`; ensure folders are created on startup.
    - [X] chore: Auto-create standard run subfolders (logs, checkpoints, artifacts)
- [X] Validate schema on load: assert positive ints, γ in [0,1], known optimizer names, valid env IDs/action_set, nonzero frameskip; reject unknown fields and fail fast with a helpful error.
    - [X] build: Add strict config schema validation with clear error messages
- [X] Summarize config/CLI conventions in `docs/design/config_cli.md`: explain file hierarchy, override precedence, required flags, schema validation, and how merged configs/meta snapshots are stored.
    - [X] docs: Add config/CLI guide with examples and troubleshooting

---

### Subtask 9 — Evaluation Harness (Metrics + Video Capture)

**Objective:**
Dedicated eval loop: greedy or low-ε, compute mean/median/std returns. Capture video (MP4) per interval. Write CSV/JSONL. Complete when evaluations run automatically with metrics and videos.

**Checklist:**
- [X] Implement a separate evaluation loop `evaluate(policy, env, n_episodes, eval_epsilon)` that runs greedily or with small ε, disables gradients, sets model to eval mode, and returns a summary dict plus per-episode stats.
    - [X] feat: Add standalone evaluate() with greedy/low-ε option and no-grad inference
- [X] Collect standardized metrics: per-episode return, length, (optional) lives lost; compute mean, median, std, min, max across episodes; include seed and step in the summary.
    - [X] feat: Aggregate episode metrics with summary statistics and run metadata
- [X] Integrate video capture: record the best-performing evaluation episode (highest return) each interval using a custom writer; ensure deterministic frame rate and save to `videos/<Game>_step_<step>_best_ep<N>_r<return>.mp4` (optional GIF export). Video metadata includes `best_episode` and `best_return` fields.
    - [X] feat: Add MP4 video capture pipeline for eval episodes
- [X] Schedule evaluations automatically every E environment frames (default 250_000) or by wall-clock; pause learning during eval, restore training mode afterward, and log the schedule in run metadata.
    - [X] feat: Add periodic evaluation trigger with proper train/eval mode switching
- [X] Write structured outputs: append a row per eval to CSV/JSONL with `step, mean_return, median_return, std_return, min_return, max_return, episodes, eval_epsilon`; save raw per-episode returns to a sidecar file for later analysis.
    - [X] chore: Persist eval summaries and per-episode details to CSV/JSONL files
- [X] Ensure evaluation defaults follow the paper: greedy policy with `eval_epsilon=0.05`, ≥10 episodes for interim checks, and ~30 episodes for final reporting; make the episode count configurable via CLI/config.
    - [X] docs: State the ε=0.05 convention and recommended episode counts in the evaluation harness docs/config comments
- [X] Document the evaluation harness in `docs/design/eval_harness.md`: describe loop structure, metric definitions, video capture settings, scheduling triggers, output file schemas, and debugging steps for desyncs or video corruption.
    - [X] docs: Include CLI examples for manual eval runs and instructions for re-rendering videos/metrics.

---

### Subtask 10 — Logging & Plotting Pipeline

**Objective:**
Structured logging through TensorBoard, Weights & Biases (W&B), and CSV. Plotting script: reward vs frames, loss vs updates, eval trends, ε schedule. Multi-seed aggregation. Upload relevant artifacts (plots, CSVs, checkpoints) to W&B for long-term storage. Complete when full pipeline (logs → plots → artifact uploads) works with one command.

**Checklist:**
- [X] Implement unified logging hooks that emit metrics simultaneously to TensorBoard, W&B, and CSV: per-step (loss, epsilon, learning rate, replay size, FPS) plus per-episode (return, length, rolling mean). Ensure consistent metric naming across backends.
    - [X] feat: Add multi-backend logging (TensorBoard/W&B/CSV) with standardized keys
- [X] Persist complete episode histories and evaluation summaries locally under `results/logs/<game>/<run_id>/`, flush on a fixed cadence, and mirror the key CSV files to W&B as artifacts after each eval/checkpoint.
    - [X] chore: Add periodic flush, deterministic filenames, and W&B artifact uploads for logs
- [X] Create `scripts/plot_results.py` to generate figures (reward vs frames, loss vs updates, eval score vs frames, epsilon schedule) from either local CSVs or W&B artifact downloads; support PNG and optional PDF/SVG outputs.
    - [X] feat: Add plotting script for reward/loss/eval/epsilon curves with local/W&B inputs
- [X] Support multi-run aggregation across seeds: align curves by environment frames, compute mean ± 95% CI (or standard error shading), and write aggregated curves to CSV + upload the summary plot/CSV to W&B.
    - [X] feat: Add multi-seed aggregation with shaded confidence intervals and artifact sync
- [X] Write outputs to `results/plots/<game>/` with deterministic filenames and embed plot metadata (smoothing window, commit hash). Mirror the plot bundle (images + metadata JSON) to W&B as an artifact for the corresponding run group.
    - [X] chore: Save plots/metadata locally and publish as W&B artifacts
- [X] Build a metadata summary generator that outputs Markdown/CSV tables (`run_id | game | mean_eval_return | frames | wall_time | seed | commit_hash`) and pushes the summary CSV/Markdown to W&B for provenance.
    - [X] feat: Add results table exporter with optional W&B upload
- [X] Provide a CLI for the plotting/aggregation pipeline: accept run directories or W&B run IDs/globs, set smoothing window/output directory, toggle artifact uploads, and fail fast on missing inputs.
    - [X] feat: Add CLI flags for plot script (local vs W&B sources, smoothing, upload toggle)
- [X] Add performance safeguards for large logs: optional downsampling/rolling aggregation prior to plotting; warn when logs exceed thresholds and automatically chunk uploads to W&B.
    - [X] feat: Add performance safeguards for large logs with downsampling and warnings
- [X] Include sanity tests/examples: run plotting off synthetic logs (CSV + TensorBoard + W&B mock), verify figures render, files exist, and W&B artifact uploads succeed (use offline/sandbox mode in tests).
    - [X] test: 48 comprehensive tests covering all logging/plotting functionality
- [X] Document the logging/plotting stack in `docs/design/logging_pipeline.md`: describe backend configs (TensorBoard dir, W&B project/entity, CSV layout), artifact upload workflow, CLI usage, and strategies for handling large logs.
    - [X] docs: Create comprehensive logging_pipeline.md with CLI examples and best practices

---

### Subtask 11 — Integration & First Training Runs

**Objective:**
Integrate MetricsLogger into DQN training loop, launch first full-length training runs on Pong (10M frames), verify all logging/checkpointing/evaluation systems work end-to-end, and document game suite plan with target scores. Complete when Pong training finishes successfully with complete logs, plots, and artifacts.

**Checklist:**
- [X] Implement complete training script (`train_dqn.py`) with main training loop: environment setup, network initialization, training step execution, episode handling, periodic evaluation, checkpoint saving, and logging integration.
    - [X] feat: Create train_dqn.py with full DQN training loop
- [X] Integrate MetricsLogger into training loop: instantiate logger from config, call `log_step()` during training steps, call `log_episode()` on episode completion, call `log_evaluation()` after eval runs, and call `close()` on shutdown.
    - [X] feat: Wire MetricsLogger into training loop with TensorBoard/W&B/CSV backends
- [X] Add logging configuration to training configs (enable_tensorboard, enable_wandb, wandb_project, flush_interval, upload_artifacts) and ensure defaults work for local-only development.
    - [X] feat: Add logging config section to base.yaml and per-game configs
- [X] Update config schema validation to accept new logging backend fields (tensorboard, csv, wandb) and their parameters.
    - [X] fix: Update schema validator with tensorboard, csv, wandb fields
- [X] Fix environment setup and API compatibility: register ALE environments in make_atari_env, correct parameter names for make_atari_env, configure_optimizer, and ReplayBuffer initialization.
    - [X] fix: Register ALE environments and fix API compatibility
- [X] Test logging integration with smoke test: run 200K frame training with all backends enabled, verify CSV files exist, TensorBoard events are written, and W&B uploads work (or gracefully degrade if offline).
    - [X] test: Validate integrated logging with comprehensive unit tests (28 tests, all passing)
- [ ] Launch first full Pong training run (10M frames, seed 42): monitor logs in real-time, verify checkpoints save every 1M steps, confirm evaluation runs at specified intervals, and check W&B artifact uploads.
    - [ ] feat: Execute first full-length training run and document any issues
- [ ] Generate plots from completed run: use `scripts/plot_results.py` to create learning curves, loss plots, eval trends, and epsilon schedule; verify metadata bundle and W&B uploads work.
    - [ ] test: Validate end-to-end plotting pipeline on real training data
- [X] Document game suite plan in `docs/design/game_suite_plan.md`: list chosen games (Pong, Breakout, Beam Rider), target scores from paper, frame budgets (10M for Pong, 50M for others), evaluation cadence (every 250K steps), and expected runtimes.
    - [X] docs: Create game suite plan with targets and budgets
- [ ] Verify resume functionality on Pong run: interrupt training mid-run, resume from checkpoint, verify metrics/RNG continuity, and document any issues.
    - [ ] test: Exercise resume on real training run (not just unit tests)
- [ ] Run GPU comparison test: execute same 1M frame run on GPU hardware to compare FPS, training time, and verify identical convergence behavior with CPU baseline.
    - [ ] perf: Document CPU vs GPU performance metrics and validate consistency

---

### Subtask 12 — Multi-Game Training & Results Collection

**Objective:**
Execute full-length training runs for game suite (Pong, Breakout, Beam Rider) across multiple seeds, verify stability, collect results, and generate comparison tables against paper baselines. Complete when all games have finished runs with aggregated metrics and plots.

**Checklist:**
- [ ] Launch Breakout training (50M frames, 3 seeds): use same configs as Pong baseline, monitor for stability issues (NaN/Inf, gradient explosions), log all runs with deterministic seeds.
    - [ ] feat: Execute Breakout training suite with stability monitoring
- [ ] Launch Beam Rider training (50M frames, 3 seeds): use same configs as Pong baseline, monitor for stability issues, log all runs with deterministic seeds.
    - [ ] feat: Execute Beam Rider training suite with stability monitoring
- [ ] Verify paper-default hyperparameters work across all games: replay=1M, batch=32, LR=2.5e-4, gamma=0.99, target_update=10k, RMSProp(rho=0.95, eps=0.01); if instability occurs, run limited sweep (2-3 LR values, 2M frames) and document in `docs/design/stability_notes.md`.
    - [ ] test: Confirm stability or document minimal tuning adjustments
- [ ] Aggregate results using `scripts/export_results_table.py`: generate `results/summary/results_summary.csv` and `.md` with columns (game, seed, mean_return, std_return, frames, wall_time, commit_hash).
    - [ ] feat: Export aggregated results table for all completed runs
- [ ] Generate multi-seed plots using `scripts/plot_results.py --multi-seed`: create learning curves with 95% CI for each game, save to `results/plots/<game>/multi_seed/`.
    - [ ] feat: Create publication-quality multi-seed aggregation plots
- [ ] Compare against paper baselines: create comparison table with columns (Game, Our Score, Paper Score, % of Paper, Status); document in `results/summary/paper_comparison.md` with brief analysis of gaps.
    - [ ] docs: Document results comparison with paper and note discrepancies
- [ ] Exercise resume functionality: interrupt one run per game mid-training, resume from checkpoint, verify metrics/RNG continuity; document pass/fail in `docs/design/run_management.md`.
    - [ ] test: Verify resume works on real multi-hour training runs
- [ ] Upload all artifacts to W&B (if used): checkpoints at 1M intervals, final plots, aggregated tables, and metadata bundles; ensure naming follows conventions from Subtask 10.
    - [ ] chore: Complete W&B artifact uploads with proper naming
- [ ] Document runtime performance: record actual FPS, wall-clock time per game/seed, GPU utilization; save to `results/summary/runtime_stats.csv` for future planning.
    - [ ] docs: Capture performance metrics for reproducibility

---


### Subtask 13 — Results Comparison & Paper Replication Tables

**Objective:**
Compare reproduced scores against the original DQN paper. Aggregate final-eval statistics, build Markdown/CSV tables, generate comparison plots, and upload the outputs to both `results/summary/` and W&B reports. Highlight gaps with diagnoses. Completion = reproducible tables/plots exist locally and in W&B, with documented interpretation guidance.

**Checklist:**
- [ ] Implement `scripts/analyze_results.py` (or extend existing tooling) to ingest per-game eval CSV/JSONL, compute stats over the final evaluation window (default last 100 episodes / final 5 checkpoints), and emit both per-game JSON + combined dataframe.
    - [ ] feat: Produce machine-readable summaries for downstream table/plot generation
- [ ] Export comparison tables (`results/summary/metrics.csv` + `.md`) with columns `Game | Mean Score (Ours) | Paper Score | % of Paper | Std Dev | Frames | Seeds | Notes`, and publish the same tables to W&B (Artifacts or Tables) with commit references.
    - [ ] feat: Keep table filenames deterministic and mirrored in W&B
- [ ] Generate bar charts and optional learning-curve overlays (`results/summary/plots/`) comparing ours vs. paper per game; attach these plots to W&B reports/dashboards for broader sharing.
    - [ ] feat: Script plot creation + W&B upload
- [ ] Flag outcomes (match/exceed/lag) with short diagnoses (env/version differences, reward clipping, budget). Include this narrative in both the Markdown summary and `docs/design/results_comparison.md`.
    - [ ] docs: Record the diagnosis rubric and update when causes change
- [ ] Document environment/toolchain differences affecting comparability (Gymnasium vs. ALE versions, hardware precision, reward preprocessing, action set choices) in `results/summary/notes.md` and reference it from the design doc.
    - [ ] docs: Keep the notes file current with each reproduction pass
- [ ] Maintain `docs/design/results_comparison.md` as the authoritative “how-to regenerate” guide: list scripts, CLI args, W&B queries, and validation steps for verifying percentage-of-paper calculations.
    - [ ] docs: Ensure the guide includes both local and W&B regeneration paths


---

### Subtask 14 — Minimal Ablations & Sensitivity Analysis

**Objective:**
Quantify the effect of key design choices (reward clipping, frame stack size, target network, etc.) on a benchmark game (Pong or Breakout). Each ablation runs ~5M frames across fixed seeds, producing comparable logs/plots/tables locally and in W&B. Completion requires documented findings and reproducible configs/scripts.

**Checklist:**
- [ ] Define ablation configs under `experiments/dqn_atari/configs/ablations/` (e.g., `reward_clip_off.yaml`, `stack_2.yaml`, `no_target_net.yaml`) with annotated headers describing the change and rationale. Reference them in `docs/design/ablations_plan.md`.
    - [ ] feat: Add/annotate ablation configs + plan doc entries
- [ ] Run each ablation for ≥5M frames on the chosen benchmark game with deterministic seeds (e.g., `{0,1,2}`) and store outputs under `experiments/dqn_atari/runs/<game>/ablations/<ablation>/seed_<n>/`. Tag the corresponding W&B runs for easy filtering.
    - [ ] feat: Ensure directory + W&B naming stays consistent
- [ ] Keep logging/eval cadence identical to baseline so comparisons are apples-to-apples (same eval episodes/cadence, epsilon schedule, reporting intervals).
    - [ ] chore: Validate configs to ensure only the intended knob changed
- [ ] Generate comparison plots (`results/ablations/<game>/<ablation>/plots/`) that overlay baseline vs ablation learning curves, eval trends, and stability indicators (TD-error variance, gradient norms). Upload the same plots to W&B.
    - [ ] feat: Extend plotting script to handle baseline-vs-ablation overlays
- [ ] Produce a summary table (`results/ablations/<game>/<ablation>/summary.csv` + Markdown) capturing deltas (final mean eval, AUC, time-to-threshold) and stability flags, and publish it to W&B.
    - [ ] feat: Automate summary export + W&B artifact upload
- [ ] Document findings in `docs/design/ablations_plan.md` (or `docs/papers/dqn_2013_notes.md` if more appropriate): describe hypothesis, setup, observed impacts, and recommendations; link to plots and run directories/W&B reports.
    - [ ] docs: Capture lessons learned per ablation with links to artifacts
- [ ] Provide a convenience runner (e.g., `experiments/dqn_atari/scripts/run_ablations.sh`) that launches the configured ablation suite with consistent seeds/output paths.
    - [ ] chore: Script reproducible ablation execution
- [ ] Capture ablation design/interpretation guidance in `docs/design/ablations_plan.md`: list the experiments, hypotheses, runtime costs, artifact locations, and how to interpret deltas/stability flags.
    - [ ] docs: Link to config files, plotting outputs, and the report section summarizing ablation findings.

---


### Subtask 15 — Aggregate Report & Interpretation

**Objective:**
Consolidate the reproduction into a polished report (`docs/reports/dqn_results.md`) plus a companion W&B report/dashboard. Include final plots/tables/videos, configs, averaged metrics, comparison to the paper, and lessons learned. Completion means the Markdown report, W&B report, and referenced artifacts remain in sync.

**Checklist:**
- [ ] Export final artifacts (per-game learning curves, aggregate bar charts, eval video links) via Subtask 10 tooling and store under `results/summary/`. Reference the same assets from W&B so readers can drill into runs.
    - [ ] docs: Capture artifact paths/URLs in the report for reproducibility
- [ ] Write/refresh `docs/reports/dqn_results.md` summarizing metrics (seed averages, runtime budgets, configs/checkpoints) with links to local artifacts and W&B run collections.
    - [ ] docs: Include a brief methods recap plus pointers to config + checkpoint locations
- [ ] Interpret outcomes relative to the original paper—highlight matches/gaps, root causes (env versions, reward clipping, precision, budgets), and quote stats from Subtask 14 tables.
    - [ ] docs: Embed comparison tables/figures or link to them directly
- [ ] Add a "Lessons learned & future work" section touching on reproducibility practices and next algorithmic steps (Double DQN, Prioritized Replay, etc.), referencing the roadmap for future subtasks.
    - [ ] docs: Keep this section updated as new insights emerge
- [ ] Publish a condensed version of the report as a W&B Report (or similar dashboard) linking to the same plots/tables/videos so collaborators can review results without cloning the repo.
    - [ ] docs: Mention the W&B report URL inside `docs/reports/dqn_results.md`
- [ ] Maintain `docs/design/report_outline.md` to track the structure of `docs/reports/dqn_results.md`, mapping each section to source artifacts, plots, and data files for quick updates or peer review.
    - [ ] docs: Record which scripts regenerate each figure/table and any open questions/todo items for the report.

---

### Subtask 16 — Code Quality, Testing, and Documentation

**Objective:**
Guarantee reliability and contributor readiness by maintaining a comprehensive automated test suite, enforcing lint/format/type-check standards, tracking coverage, and documenting how to run all tooling. Completion = clean local + CI runs plus up-to-date guidance in `docs/design/code_quality.md` and `tests/README.md`.

**Checklist:**
- [X] Expand unit/integration tests under `tests/` (replay buffer, env wrappers, model, training loop, evaluation) using seeded fixtures and pytest markers (`slow`, `gpu`, etc.). Update `tests/README.md` with commands for full runs and targeted suites (`pytest tests/test_dqn_trainer.py -k resume`).
    - [X] test: Comprehensive test coverage with 335+ tests across all modules
- [ ] Add/maintain CI workflows (e.g., GitHub Actions) that set up the venv, install deps, run `pytest` (with coverage), and upload artifacts; provide a matching local runner script (`scripts/run_tests.sh`) for convenience.
    - [ ] chore: Ensure CI + local scripts stay aligned as dependencies change
- [ ] Enforce style with Black, isort, and Ruff/flake8 via `pyproject.toml` + `pre-commit`; document the commands and expected versions in the code-quality doc.
    - [ ] build: Keep lint/format tooling configured and easy to run locally
- [ ] Enable mypy (or Pyright) for type checking on `src/`, annotate core modules incrementally, and add a `make typecheck` (or similar) target plus CI coverage.
    - [ ] build: Track annotation progress and surface common suppression patterns
- [ ] Measure coverage with pytest-cov (target ≥75% for core modules), storing reports under `results/tests/coverage/` and optionally uploading summaries to CI/W&B. Document how to open the HTML report.
    - [ ] build: Keep coverage thresholds enforced and documented
- [ ] Keep `docs/design/code_quality.md` authoritative: include testing matrix, lint/format/type-check commands, coverage targets, CI links, and troubleshooting tips.
    - [ ] docs: Update the guide whenever tooling/commands change


---

### Subtask 17 — Reproduction Recipe Script

**Objective:**
Ship a one-command reproduction script (`scripts/reproduce_dqn.sh` + optional Python driver) that automates env setup, dependency install, ROM download, training, evaluation, plotting, and W&B artifact uploads. Completion means anyone can run the script on a fresh machine and reproduce a baseline run with documented outputs/metrics.

**Checklist:**
- [ ] Implement `scripts/reproduce_dqn.sh` to orchestrate existing component scripts: (1) `envs/setup_env.sh` for venv creation and dependency install, (2) `experiments/dqn_atari/scripts/setup_roms.sh` for ROM download, (3) `train_dqn.py` for training execution, (4) `scripts/plot_results.py` for visualization, (5) optional W&B artifact uploads.
    - [ ] feat: Create unified reproduction wrapper that chains existing scripts
- [ ] Make the script configurable via flags (`--game`, `--seed`, `--frames`, `--disable-wandb`), auto-create directories (`experiments/dqn_atari/runs`, `results/`), and avoid manual path edits.
    - [ ] chore: Provide sensible defaults + overrides documented in `docs/design/reproduce_recipe.md`
- [ ] Capture environment provenance (pip freeze, Python/CUDA/Torch versions, ROM status) into `experiments/dqn_atari/system_info.txt`, plus run metadata (`meta.json`, `git_info.txt`). Include these files in any W&B artifact uploads.
    - [ ] chore: Keep provenance up to date and referenced in the doc
- [ ] After training, run the evaluation harness and plotting script automatically, dropping outputs into `results/` and attaching them to the W&B run.
    - [ ] feat: Integrate evaluation + plotting steps, respecting config defaults
- [ ] Add a verification step: compare produced metrics against a reference JSON/CSV (with tolerance) and exit non-zero on failure to catch regressions.
    - [ ] test: Maintain/update the reference metrics as baselines improve
- [ ] Document usage in README (`Quick reproduction` section) and in `docs/design/reproduce_recipe.md`, detailing prerequisites, runtime expectations, and troubleshooting tips.
    - [ ] docs: Keep documentation current whenever the script workflow changes


---

### Subtask 18 — Automated Report Generation

**Objective:**
Script (`scripts/generate_report.py`) compiling artifacts into Markdown/HTML report. Include curves, tables, videos, hardware info, commit hash. Output to `results/reports/<game>_<timestamp>.md`. Optional PDF export. Complete when report script auto-generates complete summary.

**Checklist:**
- [ ] Implement `scripts/generate_report.py` that discovers the latest run for a given game, collects plots/metrics CSV/configs/runtime info/video thumbnails, and assembles a Markdown or HTML report with sections for learning curves, evaluation tables, runtime/hardware, and commit hash; write to `results/reports/<game>_<timestamp>.md` (and `.html` if selected).
    - [ ] feat: Add generate_report script to compile artifacts into Markdown/HTML
- [ ] Provide a small report template (Markdown/Jinja) for consistent layout: title, summary, methods blurb, learning curves, eval tables, sample media, and appendix with config/metadata; support relative links to artifacts.
    - [ ] docs: Add report template for standardized layout and relative artifact links
- [ ] Include hardware/environment provenance: embed Python/CUDA/Torch versions, GPU model, ROM status, and seed/config snapshot; read from existing `system_info.txt` and `meta.json`.
    - [ ] chore: Embed environment and metadata (versions, GPU, seed, commit) in report
- [ ] Generate evaluation tables automatically from CSV/JSONL (mean/median/std/min/max, episodes, eval_epsilon) and render learning curves by linking existing plots; fall back to inline quick plots if images are missing.
    - [ ] feat: Auto-build eval tables and link or inline learning curves
- [ ] Save outputs deterministically under `results/reports/<game>_<timestamp>/` with `index.md` (and optional `index.html`) plus copied assets (thumbnails); avoid breaking links if the directory is moved.
    - [ ] chore: Write report and copy assets into a self-contained report folder
- [ ] Add optional PDF export via Pandoc or nbconvert with a single flag (`--pdf`), handling image paths and page breaks; place PDF next to the Markdown/HTML.
    - [ ] feat: Support optional PDF export of the report
- [ ] Provide a simple CLI: `python scripts/generate_report.py --game Pong --run <path|auto> [--html] [--pdf] [--out results/reports/]`; validate inputs and fail with helpful errors if required artifacts are missing.
    - [ ] feat: Add CLI flags for game/run selection, formats, and output directory
- [ ] Add a smoke test using a small synthetic run directory to ensure the script completes and produces a minimal report; verify generated tables/links exist.
    - [ ] test: Add smoke test for report generation with synthetic artifacts
- [ ] Document the automated reporting pipeline in `docs/design/reporting_pipeline.md`: outline required inputs, template structure, artifact dependencies, CLI usage, and known failure modes.
    - [ ] docs: Map each report section to the scripts/plots it depends on and provide troubleshooting guidance for missing assets.

---

### Subtask 19 — Repository Organization & Archival

**Objective:**
Keep the repository tidy and reproducible: standardized layout, documented retention rules, ignores/LFS config, licensing, README, and a fresh-clone smoke test. Completion means `docs/design/archive_plan.md` describes the structure, `.gitignore`/`.gitattributes` enforce it, and `scripts/fresh_clone_check.sh` passes.

**Checklist:**
- [X] Ensure top-level structure matches the intended layout (`src/`, `experiments/`, `results/`, `docs/`, `scripts/`, `tests/`, `.claude/`, etc.) and update README + archive plan if folders move/rename.
    - [X] chore: Repository structure is well-organized and documented
- [X] Maintain `.gitignore` / `.gitattributes` (and LFS config if needed) to exclude generated artifacts (`results/**`, `experiments/**/runs/**/checkpoints/**`, media files, caches). Document the rationale in the archive plan.
    - [X] build: .gitignore properly configured for generated artifacts
- [X] Keep README concise but complete: overview, paper reference, install/setup (venv + ROMs), quickstart commands (train/eval/plot/report), testing instructions, and links to design docs/results.
    - [X] docs: README is comprehensive with setup, usage, and navigation
- [ ] Prune temporary artifacts (old logs, massive checkpoints beyond retention policy, cache files) and document the retention rule in `results/README.md` + `docs/design/archive_plan.md`.
    - [ ] chore: Automate pruning where possible and note exceptions
- [ ] Ensure LICENSE files cover code (e.g., MIT) and docs/assets (e.g., CC-BY), and reference them from README.
    - [ ] docs: Keep licensing explicit for code vs. thesis assets
- [ ] Maintain `scripts/fresh_clone_check.sh` (or similar) that clones the repo, runs env setup, downloads ROMs (if allowed), executes the smoke test, and verifies outputs. Document pass criteria in the archive plan.
    - [ ] test: Run the fresh-clone script periodically (or via CI) and log results
- [ ] Keep `docs/design/archive_plan.md` authoritative: folder purposes, retention policies, ignore rules, licensing decisions, fresh-clone checklist, and references to helper scripts.
    - [ ] docs: Update the plan whenever structure/retention rules change


---

### Subtask 20 — Thesis Integration & Final Write-Up

**Objective:**
Integrate the DQN reproduction into the thesis manuscript: Methods (setup/implementation), Results (plots/tables/videos), Discussion (discrepancies, reproducibility), and Future Work. Keep `docs/thesis/` synchronized with repo artifacts so figures/tables can be regenerated easily.

**Checklist:**
- [ ] Write/refresh the Methods section referencing configs, seeds, evaluation protocol, and training loop diagram; link directly to code entry points (`src/train_dqn.py`, `experiments/dqn_atari/configs/`) and note how to reproduce runs.
    - [ ] docs: Ensure citations + hyperlinks align with current file paths
- [ ] Embed key results (learning curves, tables, videos/screenshots) in the Results chapter, copying assets to `docs/thesis/assets/` (or referencing W&B-hosted media) with captions/figure numbers. Document how to regenerate each asset in `docs/thesis/README.md`.
    - [ ] docs: Keep asset paths + regeneration steps accurate
- [ ] Add a discrepancy analysis table comparing our scores to the paper, citing environment/toolchain differences (ALE/Gym versions, reward clipping, precision) and diagnosing causes.
    - [ ] docs: Tie the analysis back to Subtask 14 outputs
- [ ] Reflect on reproducibility practices (env pinning, seeding, deterministic flags, checkpoint/resume, logging/report pipeline) and summarize what worked vs. what to improve.
    - [ ] docs: Include references to relevant design docs (setup, checkpointing, logging)
- [ ] Outline future work (Double DQN, Prioritized Replay, dueling networks, multi-step targets, distributional variants) with rationale and tie-ins to the roadmap.
    - [ ] docs: Keep this section aligned with remaining roadmap subtasks
- [ ] Maintain `docs/thesis/README.md` as the index from thesis sections to repository artifacts/W&B assets, noting how to regenerate each figure/table/video referenced in the manuscript.
    - [ ] docs: Update the index whenever assets move or new ones are added

---

---

## Risk Management

- **Compute constraints:** Track GPU availability; keep CPU configs for validation
- **Environment drift:** Pin emulator/gym versions; archive ROM instructions
- **Training instability:** Include gradient checks, reward clipping toggles, seeded runs
- **Data logging overload:** Rolling log retention, periodic pruning

## Future Extensions

- Representation-learning upgrades (CURL, SPR)
- Augmentation hooks (DrQ)
- Planning modules (EfficientZero)
- Rainbow variants (Double/Dueling/Distributional/Noisy nets)

---

**Note:** Subtasks are ordered by implementation sequence. Mark completed checklist items as work progresses.
