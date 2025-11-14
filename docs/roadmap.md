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
- Current focus: **M1 – Environment + tooling smoke test** (Subtask 1)

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
| **M1** | Environment + tooling smoke test passes | Pending |
| **M2** | CartPole DQN converges (>195 reward) | Pending |
| **M3** | Pong score ≥ paper benchmark | Pending |
| **M4** | Three-game suite reproduced with reports | Pending |

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
- [X] Integrate video capture: record the first evaluation episode each interval using Gym RecordVideo or a custom writer; ensure deterministic frame rate and save to `results/videos/<game>/<step>.mp4` (optional GIF export).
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
- [ ] Create `scripts/plot_results.py` to generate figures (reward vs frames, loss vs updates, eval score vs frames, epsilon schedule) from either local CSVs or W&B artifact downloads; support PNG and optional PDF/SVG outputs.
    - [ ] feat: Add plotting script for reward/loss/eval/epsilon curves with local/W&B inputs
- [ ] Support multi-run aggregation across seeds: align curves by environment frames, compute mean ± 95% CI (or standard error shading), and write aggregated curves to CSV + upload the summary plot/CSV to W&B.
    - [ ] feat: Add multi-seed aggregation with shaded confidence intervals and artifact sync
- [ ] Write outputs to `results/plots/<game>/` with deterministic filenames and embed plot metadata (smoothing window, commit hash). Mirror the plot bundle (images + metadata JSON) to W&B as an artifact for the corresponding run group.
    - [ ] chore: Save plots/metadata locally and publish as W&B artifacts
- [ ] Build a metadata summary generator that outputs Markdown/CSV tables (`run_id | game | mean_eval_return | frames | wall_time | seed | commit_hash`) and pushes the summary CSV/Markdown to W&B for provenance.
    - [ ] feat: Add results table exporter with optional W&B upload
- [ ] Provide a CLI for the plotting/aggregation pipeline: accept run directories or W&B run IDs/globs, set smoothing window/output directory, toggle artifact uploads, and fail fast on missing inputs.
    - [ ] feat: Add CLI flags for plot script (local vs W&B sources, smoothing, upload toggle)
- [ ] Add performance safeguards for large logs: optional downsampling/rolling aggregation prior to plotting; warn when logs exceed thresholds and automatically chunk uploads to W&B.
    - [ ] perf: Add scalable log downsampling and chunked artifact uploads
- [ ] Include sanity tests/examples: run plotting off synthetic logs (CSV + TensorBoard + W&B mock), verify figures render, files exist, and W&B artifact uploads succeed (use offline/sandbox mode in tests).
    - [ ] test: Add plotting/logging pipeline smoke tests covering TensorBoard/W&B/CSV paths
- [ ] Document the logging/plotting stack in `docs/design/logging_pipeline.md`: describe backend configs (TensorBoard dir, W&B project/entity, CSV layout), artifact upload workflow, CLI usage, and strategies for handling large logs.
    - [ ] docs: Reference sample commands (`python scripts/plot_results.py ... --upload-wandb`) and expected outputs for single-run vs multi-seed cases.

---

### Subtask 11 — Game Suite Plan & Training Budgets

**Objective:**
Finalize game list (Pong, Breakout, Beam Rider + optional others), frame budgets, eval cadence, target scores. Estimate runtimes. Document in `experiments/dqn_atari/README.md`. Complete when plan is approved and documented.

**Checklist:**
- [ ] Select a subset of DQN paper games for reproduction (e.g., Pong, Breakout, Beam Rider; with optional additions: Seaquest, Space Invaders, Enduro, Q*bert) and list the chosen titles clearly at the top of `experiments/dqn_atari/README.md`.
    - [ ] docs: Record selected Atari games for reproduction in README
- [ ] Specify per-game training frame budgets and evaluation cadence (e.g., 10–20M frames, evaluations every 250k frames) in a single table `Game | Env ID | Frames | Eval cadence | Notes` within `experiments/dqn_atari/README.md`.
    - [ ] docs: Add per-game frame budgets and evaluation cadence table
- [ ] Estimate runtime per game based on expected FPS and hardware (GPU/CPU) by adding a small calculator (`scripts/estimate_runtime.py`) and saving a CSV (`experiments/dqn_atari/planning/game_plan.csv`) with columns `game,fps,frames,estimated_hours,hardware`.
    - [ ] feat: Add runtime estimator script and planning CSV for game budgets
- [ ] Define objective acceptance criteria per game (e.g., target score or % of paper baseline, number of seeds, final eval window) and include them as `Target score / % baseline` columns in the README table.
    - [ ] docs: Document per-game acceptance thresholds and evaluation window
- [ ] Add game-specific overrides called out in the paper (e.g., Space Invaders requires `frameskip=3` to avoid disappearing bullets) and ensure configs/README highlight any such deviations.
    - [ ] feat: Provide per-game config knobs (like `frame_skip_override`) and document them in the plan
- [ ] Consolidate the plan in one place: ensure `experiments/dqn_atari/README.md` contains the selected games, frame budgets, eval cadence, runtime estimates link/CSV, and acceptance criteria; link to configs and runs directories.
    - [ ] chore: Finalize and cross-link game suite plan in README
- [ ] Archive the planning decisions in `docs/design/game_suite_plan.md`: capture chosen games, rationale, budgets, eval cadence, runtime assumptions, and acceptance thresholds so future contributors see how the suite was selected.
    - [ ] docs: Link to the README table, runtime CSV, and any scripts used to estimate budgets.

---

### Subtask 12 — Hyper-Parameter Tuning & Stability Verification

**Objective:**
Verify paper defaults (replay=1M, batch=32, LR=2.5e-4, γ=0.99, target=10k, RMSProp). Run ≤2M-frame stability tests. Limited sweep if unstable. Complete when stable baseline confirmed for each game.

**Checklist:**
- [ ] Initialize tuning with paper-default hyperparameters: replay capacity 1,000,000; batch size 32; learning rate 2.5e-4; γ=0.99; target update 10,000 steps; train frequency 4; optimizer RMSProp (ρ=0.95, ε=1e-2); capture these as a named preset in `experiments/dqn_atari/configs/tuning/base_paper.yaml`.
    - [ ] feat: Add paper-default tuning preset (replay=1M, batch=32, LR=2.5e-4, γ=0.99, target=10k, train_every=4, RMSProp)
- [ ] Run short stability smoke tests (≤ 2M frames per game) using the preset to check for NaNs/Infs, exploding gradients, or stuck returns; enable assertions/warnings and log anomaly counters.
    - [ ] test: Add ≤2M-frame stability smoke tests with NaN/Inf detection and gradient norm checks
- [ ] Define a minimal sweep space for instability cases: limited LR grid (e.g., {1e-4, 2.5e-4, 5e-4}), ε schedule variants (final ε ∈ {0.1, 0.01}), and reward clipping on/off; encode each trial as a small YAML under `experiments/dqn_atari/configs/tuning/`.
    - [ ] feat: Add compact tuning configs for LR, ε-schedule, and reward clipping toggles
- [ ] Constrain sweep size to ≤5 runs per game and record each run's config hash, seed, and metrics; name runs deterministically (`tuning/<game>/<paramset>_<seed>`), and write a summary CSV per game.
    - [ ] chore: Enforce ≤5 runs per game with deterministic run IDs and per-game tuning summary CSVs
- [ ] Track stability metrics and early learning signals: log moving-average return, loss variance, TD-error stats, gradient norms, and replay utilization; mark a run unstable if NaNs/Infs or divergence thresholds are exceeded.
    - [ ] feat: Log stability indicators (loss variance, TD-error, grad-norm, replay usage) with instability flags
- [ ] Select the final stable baseline per game: choose the best stable config by mean eval return at a fixed frame budget (e.g., 2M), promote it to `experiments/dqn_atari/configs/{game}.yaml`, and record rationale in a short note.
    - [ ] docs: Promote chosen stable baseline to per-game config and document selection rationale
- [ ] Verify long-run viability: launch a sanity extension run to confirm the selected config can progress toward 20M frames without crashes or divergence; update the tuning summary with pass/fail.
    - [ ] test: Add long-run viability check entry and update tuning summary with result
- [ ] Consolidate tuning decisions in `docs/design/tuning_strategy.md`: capture default hyperparameters, sweep grid rationale, stability criteria, and how to interpret tuning summary CSVs for future debugging.
    - [ ] docs: Include commands for launching tuning runs, pointers to config presets, and troubleshooting tips for instability signals.

---

### Subtask 13 — Full Training Runs & Result Collection

**Objective:**
Execute complete training for each game with 3 seeds. Save logs, checkpoints, artifacts under `experiments/dqn_atari/runs/<game>/<seed>/`. Export to `results/`. Complete when all runs finish and artifacts are available.

**Checklist:**
- [ ] Launch full training for each selected game using the verified baseline config with three independent seeds (e.g., 0, 1, 2); name runs deterministically as `experiments/dqn_atari/runs/<game>/<seed>/`.
    - [ ] feat: Start baseline training runs for each game across seeds 0,1,2 with deterministic run IDs
- [ ] Ensure each run directory contains logs, checkpoints, artifacts, and metadata (`config.yaml`, `meta.json`, env/system info); create subfolders for `checkpoints/`, `logs/`, `artifacts/`, and `eval/`.
    - [ ] chore: Standardize run folder structure and persist config/meta snapshots
- [ ] Collect core metrics continuously: training reward, evaluation score, loss, epsilon, replay size, FPS, and wall-clock; flush logs at fixed intervals and rotate if large.
    - [ ] feat: Enable continuous logging of reward/loss/ε/FPS/runtime with periodic flush
- [ ] Save final model checkpoints and retain an intermediate cadence (e.g., every 1M frames); additionally store the last 100k frames of the replay buffer or a representative sample for post-hoc analysis.
    - [ ] feat: Persist final and periodic checkpoints plus sampled replay frames for analysis
- [ ] Verify automatic resumption: interrupt one run intentionally, resume from the latest checkpoint, and confirm counters (steps/episodes/ε) and metrics continue correctly.
    - [ ] test: Validate resume-from-checkpoint behavior with consistency checks
- [ ] Aggregate per-run and cross-seed metrics on completion: write per-seed summaries (final mean eval return, frames, runtime) and a cross-seed summary CSV/JSON under `results/aggregates/<game>/`.
    - [ ] feat: Produce per-seed and cross-seed summary tables in results/aggregates
- [ ] Export plots and artifacts to `results/` for each game: learning curves, eval score trends, epsilon schedule, and a short README pointing to the best checkpoint per seed.
    - [ ] docs: Save final plots and add pointers to best checkpoints per seed
- [ ] Perform a completion audit: check that all seeds finished for all games, required files exist (final checkpoint, logs, eval summaries), and results are reproducible; record a brief audit report.
    - [ ] chore: Add completion audit report confirming artifacts and reproducibility status
- [ ] Document run execution + artifact expectations in `docs/design/run_management.md`: outline directory conventions, required artifacts per seed, audit checklist, and links to aggregation scripts.
    - [ ] docs: Provide a template for the completion audit report and instructions for verifying resumability/log completeness.

---

### Subtask 14 — Results Comparison & Paper Replication Tables

**Objective:**
Compile results vs. paper. Compute mean/median over final 100 episodes. Generate comparison tables and bar charts. Document differences. Complete when aggregated tables/figures in `results/summary/` provide clear comparison.

**Checklist:**
- [ ] Implement `scripts/analyze_results.py` to read per-game eval CSV/JSONL, compute mean/median over the final 100 evaluation episodes (or last available), and output a per-game summary dict plus a combined dataframe.
    - [ ] feat: Add analyze_results.py to aggregate final-100 eval stats per game
- [ ] Generate a Markdown and CSV comparison table with columns `Game | Mean Score (Ours) | Paper Score | % of Paper | Std Dev | Frames | Notes`; write to `results/summary/metrics.{md,csv}`.
    - [ ] feat: Export comparison tables (Markdown/CSV) with % of paper baseline
- [ ] Create visualizations: side-by-side bar plots of ours vs paper scores per game and optional learning-curve overlays; save under `results/summary/plots/` with deterministic filenames.
    - [ ] feat: Add bar charts and optional curve overlays for paper comparison
- [ ] Flag outcomes: annotate which games match/exceed the paper and which lag; include a short diagnosis field per game (e.g., version differences, reward clipping, training budget).
    - [ ] docs: Add match/lag flags and brief diagnoses to the summary outputs
- [ ] Record environment and implementation differences that could affect comparability (Gym/Gymnasium vs ALE version, hardware precision, reward processing, action set, frameskip) and include them in a `results/summary/notes.md`.
    - [ ] docs: Document environment/implementation differences impacting score comparability
- [ ] Maintain `docs/design/results_comparison.md` as a living reference: describe metrics definitions, table/plot formats, interpretation heuristics, and how to regenerate comparison artifacts.
    - [ ] docs: Note the scripts/commands used for aggregation and any manual steps for validating percentages vs. paper baselines.

---

### Subtask 15 — Minimal Ablations & Sensitivity Analysis

**Objective:**
Test design choices (reward clipping, frame stack size, target network) on one game for ~5M frames. Compare to baseline. Complete when at least one ablation demonstrates effect of a component.

**Checklist:**
- [ ] Define targeted ablations as separate configs (e.g., disable reward clipping, change frame stack 4→2, remove target network) under `experiments/dqn_atari/configs/ablations/`, with clear names and comments explaining the change and expected effect.
    - [ ] feat: Add ablation configs for reward clipping off, stack=2, and no-target-network
- [ ] Select a benchmark game (Pong or Breakout) and launch each ablation for ≥5M frames using fixed seeds (e.g., 0,1,2) and deterministic settings; store outputs under `experiments/dqn_atari/runs/<game>/ablations/<ablation>/<seed>/`.
    - [ ] feat: Run ablation experiments on benchmark game for ≥5M frames across seeds
- [ ] Log and export comparable metrics for baseline vs ablations (reward curves, eval returns, TD-error stats, loss variance); ensure identical eval cadence and ε settings to isolate effects.
    - [ ] chore: Ensure consistent logging/eval cadence for baseline and ablation runs
- [ ] Generate comparison plots: overlay learning curves (reward vs frames), show eval score trajectories, and include stability indicators (e.g., TD-error variance); save under `results/ablations/<game>/<ablation>/plots/`.
    - [ ] feat: Add ablation comparison plots with stability indicators
- [ ] Summarize quantitative impact: compute deltas vs baseline (final mean eval return, area-under-curve, time-to-threshold) and mark stability outcomes (stable/unstable/diverged) in a small summary table.
    - [ ] feat: Export ablation summary table with deltas and stability flags
- [ ] Write a short report in `docs/papers/dqn_2013_notes.md` describing each ablation, expected rationale, observed effects on stability/convergence, and key takeaways; link to plots and run directories.
    - [ ] docs: Add brief ablation report with rationale, results, and links to artifacts
- [ ] Optionally add a convenience script `scripts/run_ablations.sh` to reproduce the set (configs, seeds, output paths) with a single command.
    - [ ] chore: Add ablation runner script for reproducible execution
- [ ] Capture ablation design/interpretation guidance in `docs/design/ablations_plan.md`: list the experiments, hypotheses, runtime costs, artifact locations, and how to interpret deltas/stability flags.
    - [ ] docs: Link to config files, plotting outputs, and the report section summarizing ablation findings.

---

### Subtask 16 — Aggregate Report & Interpretation

**Objective:**
Consolidate reproduction into structured report at `docs/reports/dqn_results.md`. Include curves, tables, videos, config details, averaged results, comparison to paper, lessons learned. Complete when report is thesis-ready.

**Checklist:**
- [ ] Generate final artifacts: export per-game learning curves, aggregate score bar charts, and links/thumbnails to sample evaluation videos; save under `results/summary/` and reference them from the report.
    - [ ] docs: Export final plots and sample video links to results/summary and reference paths
- [ ] Write the aggregate report at `docs/reports/dqn_results.md` summarizing key metrics with seed averages, config details, and run metadata; include a brief methods overview and pointers to configs/checkpoints.
    - [ ] docs: Author dqn_results.md with metrics, seed averages, config notes, and artifact links
- [ ] Interpret outcomes relative to the original paper: discuss where results match or differ (e.g., convergence speed, final score), and attribute plausible causes (env versions, reward clipping, precision, budgets).
    - [ ] docs: Add comparison to paper with explanations for matches/gaps
- [ ] Add a concise "Lessons learned & future work" section outlining reproducibility takeaways and next steps (Double DQN, Prioritized Replay, dueling networks, multi-step targets).
    - [ ] docs: Include lessons learned and future work section in the report
- [ ] Maintain `docs/design/report_outline.md` to track the structure of `docs/reports/dqn_results.md`, mapping each section to source artifacts, plots, and data files for quick updates or peer review.
    - [ ] docs: Record which scripts regenerate each figure/table and any open questions/todo items for the report.

---

### Subtask 17 — Code Quality, Testing, and Documentation

**Objective:**
Ensure reliability and maintainability. Unit tests under `src/tests/`. Style compliance (Black, isort, flake8). Docstrings (Google/NumPy style). Update `src/README.md` with architecture and quickstart. Complete when tests pass, style complies, and docs enable new contributors.

**Checklist:**
- [ ] Add unit tests under `src/tests/` covering: replay buffer sampling shapes and terminal-transition handling; model forward pass dimensions and absence of NaNs/Infs; target update timing and gradient flow; include fixtures for dummy observations and a seeded RNG; enable pytest markers for slow vs. fast tests.
    - [ ] test: Add replay/model/target-update unit tests with seeded fixtures and fast/slow markers
- [ ] Integrate continuous testing by adding a simple `pytest` runner script and optional CI config (e.g., GitHub Actions) that installs requirements, runs `pytest -q --disable-warnings`, and uploads a coverage report; fail build on test errors.
    - [ ] chore: Add CI test workflow and local test runner script
- [ ] Enforce code style: configure Black, isort, and flake8 (or ruff) with a shared line length; add a top-level `pyproject.toml` and a `pre-commit` configuration to auto-format and lint on commit.
    - [ ] build: Add Black/isort/flake8 (or ruff) via pyproject and pre-commit hooks
- [ ] Add concise docstrings for public modules, classes, and functions using a consistent style (Google or NumPy); include shapes/dtypes for tensors and side effects; verify with a docstring linter if available.
    - [ ] docs: Write consistent docstrings with tensor shapes/dtypes and behavior notes
- [ ] Update `src/README.md` with an architecture overview (modules, data flow), key entry points (training, evaluation, plotting), quickstart commands, and how to run tests/linters; link to configs and results directories.
    - [ ] docs: Refresh src/README with architecture, entry points, quickstart, and testing instructions
- [ ] Add coverage measurement (pytest-cov) with a sensible threshold (e.g., 70–80% for core modules); produce HTML and XML reports; exclude generated artifacts and scripts from coverage if needed.
    - [ ] build: Enable pytest-cov with reports and set a minimum coverage threshold
- [ ] Optional static checks: enable mypy on `src/` with a minimal config (ignore missing imports for third-party libs) and add a type-check target to CI; annotate key modules (replay, model, training loop) incrementally.
    - [ ] build: Introduce mypy type checking and initial annotations for core modules
- [ ] Summarize code-quality practices in `docs/design/code_quality.md`: detail testing strategy, lint/format tooling, coverage targets, type-check scope, and onboarding instructions for running the suite locally/CI.
    - [ ] docs: Include command inventory (`pytest`, `black`, `isort`, `flake8`, `mypy`) and guidance for triaging failures.

---

### Subtask 18 — Reproduction Recipe Script

**Objective:**
End-to-end script (`scripts/reproduce_dqn.sh`) automating setup, install, ROM download, training, eval, plotting. Complete when script reproduces full run on fresh machine with consistent metrics.

**Checklist:**
- [ ] Add `scripts/reproduce_dqn.sh` (plus optional `.py`) to automate: create/activate virtualenv, install `requirements.txt`, download ROMs via `python -m AutoROM --accept-license` (if permitted), run `train_dqn.py` with the verified baseline config (e.g., Pong), then run evaluation and plotting.
    - [ ] feat: Add end-to-end reproduce_dqn script (env setup, ROMs, train, eval, plots)
- [ ] Make the script self-contained: auto-create standard directories (`experiments/dqn_atari/{runs,checkpoints,artifacts}`, `results/{plots,summary,videos}`), set deterministic seed, and avoid any manual path edits; accept optional `--game`, `--seed`, and `--frames` overrides.
    - [ ] chore: Ensure reproduce script creates folders and supports basic CLI overrides
- [ ] Capture environment provenance: within the script, save `pip freeze`, Python/CUDA/Torch versions, and ROM status to `experiments/dqn_atari/system_info.txt`; log commit hash and merged config to the run directory.
    - [ ] chore: Record environment and commit metadata during reproduction
- [ ] Evaluate and visualize automatically: after training, run the evaluation harness (K episodes, ε_eval) and generate plots via `scripts/plot_results.py`; save artifacts to `results/` with deterministic filenames.
    - [ ] feat: Wire evaluation and plotting steps into the reproduction pipeline
- [ ] Provide a verification check: compare produced metrics against a reference JSON/CSV (tolerance window) and print a pass/fail summary; exit non-zero on failure to guard regressions.
    - [ ] test: Add metric tolerance check against reference results with pass/fail output
- [ ] Document usage: add a short section in `README.md` showing one-command reproduction and expected outputs, plus runtime notes and hardware assumptions.
    - [ ] docs: Document reproduce script usage, outputs, and assumptions
- [ ] Maintain `docs/design/reproduce_recipe.md` detailing the script flow, required environment preconditions, reference metrics, and troubleshooting steps for failures.
    - [ ] docs: Map each script stage to the underlying commands and artifacts it should create.

---

### Subtask 19 — Automated Report Generation

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

### Subtask 20 — Repository Organization & Archival

**Objective:**
Clean temporary files, standardize structure (src/, experiments/, results/, docs/, scripts/). Configure .gitignore/.gitattributes. Add LICENSE. Update top-level README. Complete when repo passes fresh-clone test.

**Checklist:**
- [ ] Clean temporary data by removing intermediate logs, stale artifacts, large unused checkpoints, and cache files; keep only the latest N checkpoints per run and all evaluation summaries; document the retention rule in `results/README.md`.
    - [ ] chore: Prune temporary artifacts and old checkpoints with documented retention policy
- [ ] Organize top-level folders into a standard structure and move files accordingly: `src/` (source), `experiments/` (configs and runs), `results/` (plots, metrics, videos, reports), `docs/` (roadmap, notes), `scripts/` (automation).
    - [ ] chore: Standardize repository layout (src/, experiments/, results/, docs/, scripts/)
- [ ] Create .gitignore and .gitattributes to exclude large or generated files (e.g., `results/**`, `experiments/**/checkpoints/**`, `*.mp4`, `*.gif`, `*.pt`, `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`); optionally route large binaries to Git LFS if needed.
    - [ ] build: Add .gitignore/.gitattributes for logs, checkpoints, and media (optional LFS rules)
- [ ] Add a LICENSE file (e.g., MIT for code; CC-BY-4.0 for thesis-related docs if desired) at the repo root and reference it from the README.
    - [ ] docs: Add LICENSE file and reference from README
- [ ] Write a concise top-level README.md with sections: project overview, DQN paper reference, installation steps, quickstart commands (train/eval/plot/report), and citation information; include pointers to configs and results directories.
    - [ ] docs: Author top-level README with overview, install, quickstart, and citation
- [ ] Perform a fresh-clone test: in a clean environment, run setup, download ROMs (if permitted), execute a short smoke run, and confirm outputs appear in the standard folders; add a helper `scripts/fresh_clone_check.sh` that automates these checks.
    - [ ] test: Add fresh-clone smoke test script and document pass criteria
- [ ] Record the archival/organization policy in `docs/design/archive_plan.md`: define folder purposes, retention rules, ignore patterns, licensing decisions, and fresh-clone checklist so future cleanups stay consistent.
    - [ ] docs: Link to `.gitignore`, `results/README.md`, and `scripts/fresh_clone_check.sh`, noting how to run the audit.

---

### Subtask 21 — Thesis Integration & Final Write-Up

**Objective:**
Integrate reproduction into thesis. Methods section: goals, setup, implementation. Results: curves, tables, videos. Discussion: discrepancies, reproducibility practices. Future work: extensions. Complete when thesis narrative is coherent and supported by repo artifacts.

**Checklist:**
- [ ] Write the Methods section summarizing goals, datasets/environments, implementation details, and challenges; reference configs, seeds, and evaluation protocol; include a brief diagram of the training loop and links to code entry points.
    - [ ] docs: Add Methods section with setup, implementation details, and evaluation protocol references
- [ ] Insert key results into the Results chapter: per-game learning curves, aggregate score tables, and representative evaluation videos or screenshots; ensure captions, figure numbers, and cross-references; copy assets to `docs/thesis/assets/`.
    - [ ] docs: Embed plots/tables/videos in Results chapter with captions and asset paths
- [ ] Analyze discrepancies vs. the original DQN paper: quantify gaps (faster/slower convergence, final score deltas), list environment/toolchain differences (ALE/Gym versions, reward clipping, precision), and provide a short diagnosis table with hypothesized causes.
    - [ ] docs: Add discrepancy analysis with comparison table and hypotheses
- [ ] Reflect on reproducibility practices: document environment pinning, seeding strategy, deterministic flags, checkpoint/resume, logging/plots, and report-generation pipeline; summarize what worked and what to improve.
    - [ ] docs: Add reproducibility reflection covering pinning, seeding, determinism, and logging
- [ ] Add a Future Work section outlining extensions (Double DQN, Prioritized Replay, dueling networks, multi-step targets, distributional/Rainbow variants) and proposed next-step experiments with brief rationale.
    - [ ] docs: Add Future Work section with prioritized extensions and rationale
- [ ] Maintain `docs/thesis/README.md` as the index between thesis sections and repository artifacts (configs, plots, videos); note how to regenerate each figure/table referenced in the manuscript.
    - [ ] docs: Update the index whenever assets move or new sections are drafted to keep integration friction low.

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
