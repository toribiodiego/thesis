## Main Task 1: Implement and Reproduce the Deep Q-Network (DQN) from Scratch

**Objective**  
Re-implement and reproduce DeepMind’s DQN (*Playing Atari with Deep Reinforcement Learning*, arXiv:1312.5602). Train an agent from raw pixels to play Atari 2600 games using a single architecture and shared hyper-parameters.

**Acceptance criteria**  
- Agent learns on a subset of the **seven games used in the paper**: Pong, Breakout, Space Invaders, Seaquest, Beam Rider, Enduro, Q*bert.  
- For at least two games, performance approaches or beats the paper’s reported baselines (within a reasonable margin given hardware/time).  
- Code, configs, and results are reproducible; runs and plots are saved and documented.

---

### Subtask 1 — Choose paper games, pin versions, and scaffold runs
**Objective:**  
This subtask establishes the experimental foundation by selecting a small, representative set of Atari 2600 games from the original DQN paper—such as Pong, Breakout, and Beam Rider—and ensuring that all environments and dependencies are fully reproducible. It involves documenting and pinning every relevant version of Python, CUDA, PyTorch, Gymnasium/ALE, and the ROMs, along with environment identifiers, action set types, frame-skip policies, and deterministic or stochastic settings. A consistent evaluation protocol is defined, including the exploration parameter used during evaluation, the termination criterion (life-loss or full episode), reward clipping, evaluation frequency, and frame budgets on the order of 10–20 million per game. Each game receives a configuration file under `experiments/dqn_atari/configs/`, and a launch script is created to manage training runs in `experiments/dqn_atari/runs/`. The subtask is complete when a random-policy dry run successfully produces preprocessed frames, lists the available actions, and saves a minimal evaluation report.


**Checklist**  
-  [ ] Choose initial 2–3 games from the paper for reproduction (recommended: Pong, Breakout, Beam Rider) and create config stubs at `experiments/dqn_atari/configs/{pong,breakout,beam_rider}.yaml`, plus note the chosen games in `experiments/dqn_atari/README.md`.
    - [ ] docs: Add selected Atari games (Pong, Breakout, Beam Rider) and config stubs
- [ ] Record official game IDs (Gymnasium/ALE names like `ALE/Pong-v5`, `ALE/Breakout-v5`, `ALE/BeamRider-v5`) in `experiments/dqn_atari/README.md` as a table with an action_set column (use minimal) and ROM acquisition instructions (add `scripts/setup_roms.sh` calling `python -m AutoROM --accept-license`).
    - [ ] docs: Document ALE v5 env IDs, minimal action set, and AutoROM setup script
- [ ] Pin env versions by adding exact versions to `requirements.txt` (e.g., Python 3.10.13, PyTorch 2.4.1+cu121, Gymnasium 0.29.1, ale-py 0.8.1) and documenting ALE runtime settings in the README (e.g., `repeat_action_probability=0.0`, `frameskip=4`, `full_action_space=false`); include `scripts/capture_env.sh` to write `experiments/dqn_atari/system_info.txt`.
    - [ ] build: Pin Python/Torch/Gym/ALE versions; capture system info; document deterministic ALE settings
- [ ] Define evaluation protocol in `experiments/dqn_atari/configs/base.yaml`: ε-greedy evaluation with small ε (e.g., `eval.epsilon=0.05` or greedy toggle), termination policy (full episode for eval; training may treat life-loss as terminal), `eval.episodes=10` per checkpoint, reward clipping to {−1,0,+1}, and per-game frame budgets (e.g., 10–20M for main runs, smaller for smoke tests).
    - [ ] feat: Add base eval/train protocol (ε=0.05, full-episode eval, life-loss training option, reward clipping, frame budgets)
- [ ] Seed & reproducibility by introducing a unified `set_seed(seed, deterministic=False)` utility (`src/utils/repro.py`), wiring `--seed` in the entry point, and saving with each run the git commit hash, merged config, seed, and ALE settings as `meta.json` in the run directory.  
    - [ ] feat: Add unified seeding utility and run metadata snapshot
- [ ] Scaffold runs by creating per-game YAMLs under `experiments/dqn_atari/configs/` (common defaults + small per-game overrides) and adding `scripts/run_dqn.sh` that launches `src/train_dqn.py` with logging to `experiments/dqn_atari/runs/`; support a `--dry-run` path that executes a short random rollout, saves a few preprocessed frame stacks, lists available actions, and writes a minimal evaluation report.
    - [ ] feat: Add run launcher and random-policy dry run with frames, action list, and eval report

---

### Subtask 2 — Implement Atari env wrapper (preprocess, frame-skip, reward clipping)

**Objective:**
This subtask delivers an environment wrapper that transforms raw Atari frames into the standardized input expected by the DQN. Each frame of size 210 × 160 × 3 is converted to grayscale, resized or cropped to 84 × 84 pixels, and combined with the three preceding frames to form a stacked state of shape (4, 84, 84). The wrapper applies an action repeat of four frames with max-pooling over the last two to reduce flicker and clips rewards to the set {−1, 0, +1}. It exposes a clear interface through `reset()` and `step(action)` methods and documents how episode termination is handled during training and evaluation. Completion is indicated by verified observation shapes and the successful generation of example frame stacks and short rollout logs stored in `experiments/dqn_atari/artifacts/frames/`.

**Checklist**

* [ ] Preprocess frames from RGB (210×160) to grayscale, resize/crop to 84×84, stack the last 4 frames as channels-first `(4,84,84)` using `uint8` storage and convert to `float32` on sample; save a few sample stacks as PNGs per game.

  * [ ] feat: Add grayscale→84×84 preprocessing and 4-frame stack with sample PNG export

* [ ] Implement action repeat/frame-skip (`k=4`) and elementwise max-pool over the last two raw frames before preprocessing to reduce flicker.

  * [ ] feat: Add 4-step action repeat and last-2-frame max-pooling

* [ ] Apply reward clipping to the set {−1, 0, +1} with a config toggle to disable for ablations.

  * [ ] feat: Add configurable reward clipping to {-1,0,+1}

* [ ] Align episode termination with evaluation policy: allow training to treat life loss as terminal, use full-episode termination for evaluation, and support optional no-op starts and auto-fire reset where needed; document choices in the wrapper docstring.

  * [ ] docs: Document termination (life-loss vs full-episode), no-op starts, and auto-fire behavior

* [ ] Produce debug artifacts: write a short random rollout log recording obs shape, action repeat behavior, and clipped reward stats; save preprocessed stacks under `experiments/dqn_atari/artifacts/frames/<game>/` and the rollout log in the corresponding run directory.

  * [ ] feat: Emit rollout debug log and per-game preprocessed frame artifacts


---

### Subtask 3 — Implement the DQN model (vision trunk + Q-head)
**Objective:**  
This subtask implements the convolutional neural network that maps processed game states to Q-values for each possible action. The architecture follows the structure described in the DQN paper: an input tensor of shape (4 × 84 × 84) followed by two convolutional layers (16 filters of size 8 × 8 with stride 4, then 32 filters of size 4 × 4 with stride 2), a fully connected layer with 256 ReLU units, and an output layer with one linear unit per action. All tensors use the channels-first format with float32 precision, and weights are initialized using Kaiming or Xavier initialization. The network includes checkpointing and model summary utilities, and a small verification script confirms correct output dimensions and stable forward passes without numerical issues.


**Checklist**

* [ ] Implement DQN CNN with input `(4,84,84)`: Conv1 (16, 8×8, stride 4, ReLU) → Conv2 (32, 4×4, stride 2, ReLU) → flatten → FC(256, ReLU) → linear head of size `|A|`; channels-first tensors, return dict with `q_values` and optional `features` for debugging.

  * [ ] feat: Add DQN model (Conv8x8s4→Conv4x4s2→FC256→Q-head)

* [ ] Set weight initialization and dtypes: use Kaiming normal (fan_out) for conv/linear with ReLU, zeros for biases; keep parameters in float32; expose a `to(device)` utility; ensure forward accepts `float32` inputs scaled to `[0,1]`.

  * [ ] build: Configure Kaiming init and float32 dtype for all layers

* [ ] Add model summary and shape checks: implement a small `model_summary(module, input_shape)` printer and log parameter count; assert expected output shape `(B, |A|)` for a dummy batch; handle dynamic `|A|` from env.

  * [ ] chore: Add model summary utility and output-shape assertion

* [ ] Create forward-path unit tests: with random input `(B=2, 4, 84, 84)` verify no NaNs/Infs, correct shapes for action spaces (e.g., Pong=6, Breakout=4, BeamRider=9), and gradients flow with a dummy MSE loss/backward.

  * [ ] test: Add forward/grad tests across multiple action sizes

* [ ] Implement save/load helpers: `save_checkpoint(path, state_dict, meta)` and `load_checkpoint(path)` for model-only, plus convenience `from_env(action_space_n)` constructor; ensure strict key matching and device-safe loading.

  * [ ] feat: Add checkpoint save/load and environment-aware constructor



### Subtask 4 — Experience Replay Buffer (uniform)
**Objective:**  
This subtask creates a replay memory for storing up to roughly one million transitions of the form (state, action, reward, next_state, done). The buffer operates as a circular queue that overwrites old data as new transitions arrive. Frames are stored efficiently as uint8 arrays and converted to normalized float32 tensors when sampled. The implementation provides methods for appending new transitions, sampling random minibatches, and reporting the current buffer size, while ensuring that sampled batches never cross episode boundaries. A short warm-up period of about 50,000 random steps precedes training to populate the buffer. The subtask is complete when the buffer returns correctly shaped batches, integrates cleanly with GPU-based training, and achieves practical sampling speed.


**Checklist**

* [ ] Implement a circular replay buffer (capacity e.g., `1_000_000`) that stores tuples `(s_t, a_t, r_t, s_{t+1}, done_t)` with a ring write index and per-step episode boundary markers to prevent cross-episode samples.

  * [ ] feat: Add uniform replay buffer with circular storage and episode boundary tracking

* [ ] Provide a minimal API: `append(state, action, reward, next_state, done)`, `sample(batch_size) -> dict(tensors)`, and `__len__`; include input validation and graceful handling when the buffer has fewer than `batch_size` valid indices.

  * [ ] feat: Expose append/sample/len API with basic validation

* [ ] Optimize memory layout: store frames (and frame stacks) as `uint8` to save RAM, keep a contiguous frame array plus indices for stacking, convert to `float32` only on sample, and normalize to `[0,1]` (configurable).

  * [ ] build: Store observations as uint8 and defer float32 conversion/normalization to sampling

* [ ] Enforce warm-up: add a configurable pre-fill (default `50_000` random steps) before any optimization; expose `can_sample(min_size)` helper used by the training loop.

  * [ ] feat: Add warm-up threshold and can_sample helper

* [ ] Implement uniform sampling without replacement: draw valid indices that have `t, t-1, t-2, t-3` within the same episode and available `t+1` (for `s'`), rejecting indices near wrap/episode boundaries; return batches with shapes `s: (B,4,84,84)`, `a: (B,)`, `r: (B,)`, `s_next: (B,4,84,84)`, `done: (B,)`.

  * [ ] feat: Add boundary-safe uniform sampler with no replacement

* [ ] Device transfer and speed: on `sample`, assemble stacks, convert to `float32`, normalize (or leave in 0–255 if configured), move tensors to GPU if available, and optionally use pinned host memory for faster H2D copies.

  * [ ] perf: Add device move, optional pinned memory, and normalization toggle

* [ ] Add tests: fill buffer past `batch_size`, call `sample`, verify exact shapes and dtypes, ensure no cross-episode indices, check wrap-around correctness at buffer edges, and assert reproducibility with a fixed RNG seed.

  * [ ] test: Add shape/boundary/repro tests for sampling and ring wrap-around




---

### Subtask 5 — Q-Learning Loss, Target Network, and Optimizer
**Objective:**  
This subtask defines the DQN learning rule and training procedure. Two identical networks are maintained: an online network for learning and a target network for stabilizing the updates. For each minibatch, the target value *y = r + γ × (1 − done) × maxₐ′ Q_target(s′, a′)* is computed and compared with the corresponding Q-value predicted by the online network. Training minimizes either the mean-squared error or the Huber loss (δ = 1.0) using RMSProp (ρ = 0.95, ε = 1e-2) or Adam, with a learning rate of 2.5 × 10⁻⁴, discount factor γ = 0.99, batch size 32, and gradient clipping around 10. The target network is updated every 10,000 environment steps, and optimization occurs every four steps. Successful completion is indicated by a stable loss curve, correct synchronization of target updates, and smooth end-to-end training on sampled batches.


**Checklist**

* [ ] Create online and target Q-networks with identical architecture; initialize target as a hard copy of online, freeze target grads, and provide `hard_update_target()` utility.

  * [ ] feat: Initialize online/target Q-nets and hard-copy sync helper

* [ ] Compute TD targets per minibatch using `y = r + γ * (1 - done) * max_a' Q_target(s', a')` under `no_grad`, and gather `Q_online(s, a)` with `gather` for chosen actions; ensure correct broadcasting and shapes `(B,)` after squeeze.

  * [ ] feat: Implement TD target computation and online Q selection

* [ ] Add configurable loss: default MSE on `(Q_selected - y)` with `reduction='mean'`, optional Huber (δ=1.0) via config flag; return loss and aux stats (mean |TD error|).

  * [ ] feat: Add MSE/Huber loss with TD-error metrics

* [ ] Configure optimizer and hyperparameters: RMSProp (ρ=0.95, ε=1e-2) or Adam via config; LR `2.5e-4`, γ `0.99`, batch size `32`; apply global-norm gradient clipping (e.g., `10.0`) right before `optimizer.step()`.

  * [ ] build: Add optimizer setup and global-norm gradient clipping

* [ ] Implement periodic target updates: call `hard_update_target()` every `C` environment steps (default `10_000`); track env step counter and log each sync step.

  * [ ] feat: Add step-scheduled hard target sync with logging

* [ ] Schedule training frequency: perform one optimization step every `k=4` environment steps after replay warm-up; skip updates if `can_sample` is false; support configurable `train_every`.

  * [ ] feat: Add train-every-k update scheduler with warm-up gating

* [ ] Add stability checks: unit test on a synthetic batch to confirm loss decreases over several updates; assert target updates occur at exact multiples of `C`; detect and warn on NaNs/Infs; log grad norm and LR per update.

  * [ ] test: Add toy-batch loss decrease and target-sync schedule tests

* [ ] Minimal metrics logging from the update step: loss, mean |TD error|, grad norm, learning rate, and update count for downstream plotting.

  * [ ] chore: Log core update metrics (loss, TD-error, grad-norm, lr)



---

### Subtask 6 — Training Loop, Exploration Schedule, Logging & Evaluation
**Objective:**  
This subtask integrates all major components into a working training system that interacts with the environment, learns from stored experience, and records progress for later analysis. The agent follows an ε-greedy policy where ε decays linearly from 1.0 to 0.1 over the first one million frames, optionally continuing to 0.01 thereafter. At each step, an action is chosen using the online network, executed for four repeated frames, and the resulting transition is added to the replay buffer. After the warm-up period, minibatches are drawn periodically to perform learning updates, and the target network is synchronized every 10,000 steps. The system logs losses, rewards, ε values, and performance metrics to `experiments/dqn_atari/runs/`, with periodic evaluations every ~250,000 frames at a low exploration rate (ε ≈ 0.05). The subtask is complete when a short training run of roughly 200,000 frames executes reliably, produces consistent logs and checkpoints, and shows early evidence of learning stability.

**Checklist**

* [ ] Implement ε-greedy exploration with a configurable linear schedule: start ε=1.0, decay to 0.1 over the first 1,000,000 frames (option to continue to 0.01), and use a separate `eval_epsilon` (e.g., 0.05) only during evaluation; expose all as config and log ε per step.

  * [ ] feat: Add configurable ε schedules (train and eval) with per-step logging

* [ ] Ensure action repeat/frame-skip integration: execute the chosen action for k frames (default 4), accumulate clipped reward from the wrapper, and count environment frames correctly (not decisions); record effective FPS.

  * [ ] feat: Integrate frame-skip execution and reward accumulation with accurate frame counters

* [ ] Build the main step loop: (1) select action via ε-greedy from the online Q-net, (2) step env with frame-skip, (3) append transition to replay, (4) if warm-up done and `t % train_every == 0` then sample → compute loss → backprop → optimizer step, (5) if `t % target_update == 0` then hard-sync target.

  * [ ] feat: Implement training loop with scheduled optimization and target sync

* [ ] Handle episodes consistently: reset on terminal; optionally treat life-loss as terminal during training; run full episodes during evaluation; optionally support no-op starts; record per-episode return and length.

  * [ ] docs: Document training/eval termination policy and optional no-op starts

* [ ] Add structured logging under `experiments/dqn_atari/runs/`: per-step (loss moving average, ε, learning rate, replay size, grad norm) and per-episode (return, length, FPS, rolling mean over last N); save checkpoints on a fixed cadence and on best eval score.

  * [ ] feat: Add step/episode loggers and periodic/best checkpoints

* [ ] Implement the evaluation routine: every E frames (default 250,000) run K episodes (default 10) with `eval_epsilon`; disable learning, set eval mode, log mean/median/std returns, and write plots/CSV to `results/`.

  * [ ] feat: Add periodic evaluation with summary metrics and result artifacts

* [ ] Persist reproducibility metadata for each run: save merged config snapshot, seed, and git commit hash beside logs and checkpoints (JSON/YAML).

  * [ ] chore: Write run metadata (config, seed, commit) to run folder

* [ ] Run a smoke test (~200,000 frames) to verify end-to-end stability: confirm logs grow, checkpoints appear, eval runs trigger, and quick plots render without errors.

  * [ ] test: Add smoke-test script to validate loop, logging, checkpoints, and eval cadence



---

### Subtask 7 — Checkpointing, Resume, and Deterministic Seeding
**Objective:**  
This subtask introduces robust checkpointing and deterministic behavior to ensure that any training run can be paused, resumed, and exactly reproduced. The system must save and restore all essential components: the online and target network weights, optimizer state, replay buffer position, step counter, exploration rate, and random number generator states for PyTorch, NumPy, and the environment. A command-line flag such as `--resume path/to/checkpoint.pt` allows seamless continuation from a saved checkpoint. Random seed management is centralized so that all sources of randomness are synchronized and recorded in the run metadata. Deterministic execution is enforced where possible by setting PyTorch’s deterministic mode and disabling cuDNN benchmarking. The subtask is considered complete when a short experiment can be checkpointed, resumed, and shown to produce identical rewards, Q-values, and exploration parameters through a fixed number of frames.


**Checklist**

* [ ] Implement checkpoint structure that saves online/target weights, optimizer state, step counter, episode counter, ε value, replay buffer write index and content (or snapshot pointer), and RNG states (torch, numpy, random, env) to `experiments/dqn_atari/checkpoints/checkpoint_{steps}.pt`; write atomically (temp file → rename) and include a small `meta` dict (schema version, timestamp, commit hash).

  * [ ] feat: Add atomic checkpoint save with models, optimizer, replay position, RNG states, and run metadata

* [ ] Add resume logic via `--resume path/to/checkpoint.pt` that restores device-safe tensors, optimizer, step/episode counters, ε schedule state, RNG states, and (if present) replay buffer; validate config compatibility and warn on commit/hash mismatch; resume training seamlessly from the next step.

  * [ ] feat: Implement robust resume path restoring counters, schedules, and replay buffer

* [ ] Centralize deterministic seeding with `set_seed(seed, deterministic=True)` to seed Python, NumPy, Torch (CPU/GPU), and the env on every reset; record the seed in run metadata and propagate to workers if using multiprocessing.

  * [ ] feat: Add deterministic seeding utility and metadata recording

* [ ] Control randomness for reproducibility by setting `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`, and optionally `torch.use_deterministic_algorithms(True)` behind a config flag; document potential performance trade-offs.

  * [ ] docs: Document deterministic flags and performance implications

* [ ] Create a smoke test: run ~10k steps, save a checkpoint, resume from it, and verify identical ε, rewards, and selected actions for a fixed number of frames (allow tiny FP tolerance); emit a short comparison report (match/ mismatch counts, checksums).

  * [ ] test: Add save/resume determinism test with metric comparison and checksum report



---

### Subtask 8 — Config System and Command-Line Interface

**Objective:**  
This subtask implements a configuration system and standardized command-line interface so that all experiment parameters are clearly defined, version-controlled, and easy to reproduce. A base configuration file stores global defaults for model architecture, replay settings, optimizer parameters, and evaluation frequency, while game-specific YAML files provide only the necessary overrides. A lightweight configuration loader merges these files and validates all fields before training begins. Each training run automatically saves a copy of the merged configuration and metadata (including commit hash and seed) into its run directory. A unified entry point, such as `python train_dqn.py --cfg configs/pong.yaml --seed 123`, should launch any experiment with a single command. Completion is indicated by a clean, modular configuration setup that allows consistent reproduction of results and traceability of every experimental parameter.


**Checklist**

* [ ] Create base and per-game config files: add `experiments/dqn_atari/configs/base.yaml` for global defaults (network, replay, optimizer, target_update, eval cadence) and `experiments/dqn_atari/configs/{pong,breakout,beam_rider}.yaml` that override only env-specific fields (e.g., `env_id`, `action_set`, `frame_budget`).

  * [ ] docs: Add base and per-game YAML configs with clear comments on each field

* [ ] Implement a lightweight config loader that merges base + game overrides, supports nested keys, and returns a dict/dataclass; print the resolved config at startup for traceability.

  * [ ] feat: Add config merge utility with nested override and resolved-config logging

* [ ] Provide a CLI entry point to launch experiments with a single command:
  `python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 123 --resume path/to/checkpoint.pt` and allow optional `--set key.subkey=value` overrides.

  * [ ] feat: Add CLI flags for --cfg, --seed, --resume, and inline overrides

* [ ] Auto-save the merged config snapshot to each run folder (`experiments/dqn_atari/runs/<game>/<timestamp>/config.yaml`) alongside `meta.json`; include commit hash and seed.

  * [ ] chore: Persist merged config and metadata to each run directory

* [ ] Create dynamic run paths automatically (logs, checkpoints, artifacts) under `experiments/dqn_atari/runs/<game>/<timestamp>/`; ensure folders are created on startup.

  * [ ] chore: Auto-create standard run subfolders (logs, checkpoints, artifacts)

* [ ] Validate schema on load: assert positive ints, γ in [0,1], known optimizer names, valid env IDs/action_set, nonzero frameskip; reject unknown fields and fail fast with a helpful error.

  * [ ] build: Add strict config schema validation with clear error messages



---

### Subtask 9 — Evaluation Harness (Metrics + Video Capture)

**Objective:**  
This subtask establishes a consistent and automated evaluation process for assessing agent performance. A dedicated evaluation loop runs the trained policy for a specified number of episodes using either greedy or low-ε behavior (e.g., ε = 0.05). For each evaluation, the framework computes aggregate metrics such as mean, median, and standard deviation of episode returns, lengths, and any game-specific statistics. Video capture is integrated to record at least one full evaluation episode at each evaluation interval, stored under `results/videos/<game>/<step>.mp4` or as a GIF. Evaluations occur automatically at regular intervals (for example, every 250,000 environment steps), and their results are logged to structured CSV or JSONL files. The subtask is complete when evaluations run seamlessly during training and generate both numerical summaries and visual recordings suitable for later comparison and reporting.


**Checklist**

* [ ] Implement a separate evaluation loop `evaluate(policy, env, n_episodes, eval_epsilon)` that runs greedily or with small ε, disables gradients, sets model to eval mode, and returns a summary dict plus per-episode stats.

  * [ ] feat: Add standalone evaluate() with greedy/low-ε option and no-grad inference

* [ ] Collect standardized metrics: per-episode return, length, (optional) lives lost; compute mean, median, std, min, max across episodes; include seed and step in the summary.

  * [ ] feat: Aggregate episode metrics with summary statistics and run metadata

* [ ] Integrate video capture: record the first evaluation episode each interval using Gym RecordVideo or a custom writer; ensure deterministic frame rate and save to `results/videos/<game>/<step>.mp4` (optional GIF export).

  * [ ] feat: Add MP4 video capture pipeline for eval episodes

* [ ] Schedule evaluations automatically every E environment frames (default 250_000) or by wall-clock; pause learning during eval, restore training mode afterward, and log the schedule in run metadata.

  * [ ] feat: Add periodic evaluation trigger with proper train/eval mode switching

* [ ] Write structured outputs: append a row per eval to CSV/JSONL with `step, mean_return, median_return, std_return, min_return, max_return, episodes, eval_epsilon`; save raw per-episode returns to a sidecar file for later analysis.

  * [ ] chore: Persist eval summaries and per-episode details to CSV/JSONL files



---

### Subtask 10 — Logging & Plotting Pipeline

**Objective:**  
This subtask automates experiment logging and visualization to enable quantitative and qualitative monitoring of training progress. The training system must record key metrics—such as training loss, ε value, mean reward, replay buffer size, and frames per second—using a consistent logging backend like TensorBoard, Weights & Biases, or structured CSV files. A separate plotting script (`scripts/plot_results.py`) processes these logs to generate standardized figures, including learning curves of reward versus environment frames, loss versus updates, evaluation score trends, and ε schedules. Support for aggregating results across multiple seeds is included to compute averages and confidence intervals. The outputs are saved under `results/plots/<game>/` and summarized in a concise Markdown or CSV table listing run identifiers, performance metrics, total frames, and relevant metadata. Completion is achieved when the entire experiment lifecycle—from raw logs to publication-quality plots—can be reproduced with a single plotting command.

**Checklist**

* [ ] Implement structured logging backend (TensorBoard, Weights & Biases, or CSV) that records per-step metrics (loss, epsilon, learning rate, replay size, FPS) and per-episode metrics (return, length, rolling mean over last N); standardize metric names and units.

  * [ ] feat: Add unified training/eval logging with standardized metric keys

* [ ] Persist complete episode histories and evaluation summaries alongside raw step logs; ensure log flushing at fixed intervals and stable file naming under `results/logs/<game>/<run_id>/`.

  * [ ] chore: Add periodic flush and consistent log directory structure

* [ ] Create `scripts/plot_results.py` to generate figures: reward vs frames (100-episode moving average), loss vs updates, eval score vs frames, and epsilon schedule vs frames; support saving PNG and optional PDF/SVG.

  * [ ] feat: Add plotting script for reward/loss/eval/epsilon curves

* [ ] Support multi-run aggregation across seeds: align curves by environment frames, compute mean and 95% confidence intervals (or standard error shading), and render aggregated plots.

  * [ ] feat: Add multi-seed aggregation with shaded confidence intervals

* [ ] Write outputs to `results/plots/<game>/` with deterministic filenames (include run set or seed list) and embed plot metadata (smoothing window, commit hash) in a sidecar JSON.

  * [ ] chore: Save plots with deterministic names and sidecar metadata

* [ ] Build a metadata summary generator that outputs Markdown and CSV tables listing `run_id | game | mean_eval_return | frames | wall_time | seed | commit_hash` and links to logs/plots.

  * [ ] feat: Add results table exporter (Markdown and CSV)

* [ ] Provide a CLI for the plotting pipeline: accept one or more run directories or a glob, set smoothing window, output directory, and options to include/exclude seeds; fail fast with helpful errors.

  * [ ] feat: Add CLI flags for plot script (inputs, smoothing, outputs, filters)

* [ ] Add performance safeguards for large logs: optional downsampling or rolling aggregation before plotting to keep runtime/memory reasonable on long runs.

  * [ ] perf: Add log downsampling/rolling aggregation for scalable plotting

* [ ] Include sanity tests and examples: run plotting on a small synthetic log to verify figures render and files are created; validate CSV headers and TensorBoard/W&B export parsing paths.

  * [ ] test: Add plotting smoke tests and log parser checks


---

### Subtask 11 — Game Suite Plan & Training Budgets

**Objective:**  
This subtask defines the experimental scope and resource plan for reproducing the DQN results. A subset of Atari 2600 games from the original paper is selected to balance diversity and feasibility—typically Pong, Breakout, and Beam Rider for initial replication, with the option to extend to Seaquest, Space Invaders, Enduro, or Q*bert later. For each game, the total number of environment frames, evaluation intervals, and target performance thresholds are established and documented. Expected training runtimes are estimated based on hardware throughput to guide scheduling and compute allocation. A clear table listing every chosen game, its frame budget, and target score is added to `experiments/dqn_atari/README.md`. The subtask is complete when these parameters are finalized and approved as the benchmark plan for all subsequent experiments.


**Checklist**
**Checklist**

* [ ] Select a subset of DQN paper games for reproduction (e.g., Pong, Breakout, Beam Rider; with optional additions: Seaquest, Space Invaders, Enduro, Q*bert) and list the chosen titles clearly at the top of `experiments/dqn_atari/README.md`.

  * [ ] docs: Record selected Atari games for reproduction in README

* [ ] Specify per-game training frame budgets and evaluation cadence (e.g., 10–20M frames, evaluations every 250k frames) in a single table `Game | Env ID | Frames | Eval cadence | Notes` within `experiments/dqn_atari/README.md`.

  * [ ] docs: Add per-game frame budgets and evaluation cadence table

* [ ] Estimate runtime per game based on expected FPS and hardware (GPU/CPU) by adding a small calculator (`scripts/estimate_runtime.py`) and saving a CSV (`experiments/dqn_atari/planning/game_plan.csv`) with columns `game,fps,frames,estimated_hours,hardware`.

  * [ ] feat: Add runtime estimator script and planning CSV for game budgets

* [ ] Define objective acceptance criteria per game (e.g., target score or % of paper baseline, number of seeds, final eval window) and include them as `Target score / % baseline` columns in the README table.

  * [ ] docs: Document per-game acceptance thresholds and evaluation window

* [ ] Consolidate the plan in one place: ensure `experiments/dqn_atari/README.md` contains the selected games, frame budgets, eval cadence, runtime estimates link/CSV, and acceptance criteria; link to configs and runs directories.

  * [ ] chore: Finalize and cross-link game suite plan in README


---

### Subtask 12 — Hyper-Parameter Tuning & Stability Verification

**Objective:**  
This subtask ensures that the baseline DQN configuration trains stably and performs within expectations before large-scale experiments begin. Initial tests use the hyperparameters from the original paper—replay capacity = 1 M, batch = 32, learning rate = 2.5 × 10⁻⁴, γ = 0.99, target update = 10 k, and training frequency = 4—together with RMSProp (ρ = 0.95, ε = 1e-2). Short training runs of up to two million frames are conducted to verify numerical stability, absence of NaNs, and early learning trends. If instability or divergence is observed, a limited sweep of key parameters such as learning rate, ε-schedule, and reward clipping is performed. All experimental configurations and results are logged under `experiments/dqn_atari/configs/tuning/`. This subtask is complete once stable baseline settings are confirmed for each selected game and can run to 20 million frames without crash or divergence.


**Checklist**

* [ ] Initialize tuning with paper-default hyperparameters: replay capacity 1,000,000; batch size 32; learning rate 2.5e-4; γ=0.99; target update 10,000 steps; train frequency 4; optimizer RMSProp (ρ=0.95, ε=1e-2); capture these as a named preset in `experiments/dqn_atari/configs/tuning/base_paper.yaml`.

  * [ ] feat: Add paper-default tuning preset (replay=1M, batch=32, LR=2.5e-4, γ=0.99, target=10k, train_every=4, RMSProp)

* [ ] Run short stability smoke tests (≤ 2M frames per game) using the preset to check for NaNs/Infs, exploding gradients, or stuck returns; enable assertions/warnings and log anomaly counters.

  * [ ] test: Add ≤2M-frame stability smoke tests with NaN/Inf detection and gradient norm checks

* [ ] Define a minimal sweep space for instability cases: limited LR grid (e.g., {1e-4, 2.5e-4, 5e-4}), ε schedule variants (final ε ∈ {0.1, 0.01}), and reward clipping on/off; encode each trial as a small YAML under `experiments/dqn_atari/configs/tuning/`.

  * [ ] feat: Add compact tuning configs for LR, ε-schedule, and reward clipping toggles

* [ ] Constrain sweep size to ≤5 runs per game and record each run’s config hash, seed, and metrics; name runs deterministically (`tuning/<game>/<paramset>_<seed>`), and write a summary CSV per game.

  * [ ] chore: Enforce ≤5 runs per game with deterministic run IDs and per-game tuning summary CSVs

* [ ] Track stability metrics and early learning signals: log moving-average return, loss variance, TD-error stats, gradient norms, and replay utilization; mark a run unstable if NaNs/Infs or divergence thresholds are exceeded.

  * [ ] feat: Log stability indicators (loss variance, TD-error, grad-norm, replay usage) with instability flags

* [ ] Select the final stable baseline per game: choose the best stable config by mean eval return at a fixed frame budget (e.g., 2M), promote it to `experiments/dqn_atari/configs/{game}.yaml`, and record rationale in a short note.

  * [ ] docs: Promote chosen stable baseline to per-game config and document selection rationale

* [ ] Verify long-run viability: launch a sanity extension run to confirm the selected config can progress toward 20M frames without crashes or divergence; update the tuning summary with pass/fail.

  * [ ] test: Add long-run viability check entry and update tuning summary with result



---

### Subtask 13 — Full Training Runs & Result Collection

**Objective:**  
This subtask executes the complete DQN training protocol for each selected game using the verified baseline configuration. Multiple independent runs—typically three seeds per game—are launched, and all artifacts are saved in a structured format under `experiments/dqn_atari/runs/<game>/<seed>/`. Each run records full training logs, evaluation metrics, loss values, exploration rates, frames per second, and wall-clock time. The final model checkpoints and a portion of the replay buffer are stored for post-hoc analysis. Automatic resumption and metric aggregation are tested to ensure recoverability and consistency across runs. Completion is achieved when all scheduled training runs finish successfully and their logs, plots, and checkpoints are available for analysis under `results/`.


**Checklist**

* [ ] Launch full training for each selected game using the verified baseline config with three independent seeds (e.g., 0, 1, 2); name runs deterministically as `experiments/dqn_atari/runs/<game>/<seed>/`.

  * [ ] feat: Start baseline training runs for each game across seeds 0,1,2 with deterministic run IDs

* [ ] Ensure each run directory contains logs, checkpoints, artifacts, and metadata (`config.yaml`, `meta.json`, env/system info); create subfolders for `checkpoints/`, `logs/`, `artifacts/`, and `eval/`.

  * [ ] chore: Standardize run folder structure and persist config/meta snapshots

* [ ] Collect core metrics continuously: training reward, evaluation score, loss, epsilon, replay size, FPS, and wall-clock; flush logs at fixed intervals and rotate if large.

  * [ ] feat: Enable continuous logging of reward/loss/ε/FPS/runtime with periodic flush

* [ ] Save final model checkpoints and retain an intermediate cadence (e.g., every 1M frames); additionally store the last 100k frames of the replay buffer or a representative sample for post-hoc analysis.

  * [ ] feat: Persist final and periodic checkpoints plus sampled replay frames for analysis

* [ ] Verify automatic resumption: interrupt one run intentionally, resume from the latest checkpoint, and confirm counters (steps/episodes/ε) and metrics continue correctly.

  * [ ] test: Validate resume-from-checkpoint behavior with consistency checks

* [ ] Aggregate per-run and cross-seed metrics on completion: write per-seed summaries (final mean eval return, frames, runtime) and a cross-seed summary CSV/JSON under `results/aggregates/<game>/`.

  * [ ] feat: Produce per-seed and cross-seed summary tables in results/aggregates

* [ ] Export plots and artifacts to `results/` for each game: learning curves, eval score trends, epsilon schedule, and a short README pointing to the best checkpoint per seed.

  * [ ] docs: Save final plots and add pointers to best checkpoints per seed

* [ ] Perform a completion audit: check that all seeds finished for all games, required files exist (final checkpoint, logs, eval summaries), and results are reproducible; record a brief audit report.

  * [ ] chore: Add completion audit report confirming artifacts and reproducibility status


---

### Subtask 14 — Results Comparison & Paper Replication Tables

**Objective:**  
This subtask compiles and analyzes the results of all completed training runs to evaluate reproduction accuracy relative to the original DQN paper. A dedicated analysis script processes the evaluation logs to compute mean and median scores over the final evaluation window (e.g., the last 100 episodes). The data are summarized in a comparison table showing each game’s average performance, corresponding score reported in the paper, standard deviation, and relative percentage of the baseline. Visual comparisons such as side-by-side bar charts or learning-curve overlays are generated to highlight agreement and discrepancies. Differences in environment versions, reward scaling, or hardware precision are documented. The subtask is complete when the aggregated tables and figures in `results/summary/` provide a clear quantitative comparison between reproduced and published performance.


**Checklist**

* [ ] Implement `scripts/analyze_results.py` to read per-game eval CSV/JSONL, compute mean/median over the final 100 evaluation episodes (or last available), and output a per-game summary dict plus a combined dataframe.

  * [ ] feat: Add analyze_results.py to aggregate final-100 eval stats per game

* [ ] Generate a Markdown and CSV comparison table with columns `Game | Mean Score (Ours) | Paper Score | % of Paper | Std Dev | Frames | Notes`; write to `results/summary/metrics.{md,csv}`.

  * [ ] feat: Export comparison tables (Markdown/CSV) with % of paper baseline

* [ ] Create visualizations: side-by-side bar plots of ours vs paper scores per game and optional learning-curve overlays; save under `results/summary/plots/` with deterministic filenames.

  * [ ] feat: Add bar charts and optional curve overlays for paper comparison

* [ ] Flag outcomes: annotate which games match/exceed the paper and which lag; include a short diagnosis field per game (e.g., version differences, reward clipping, training budget).

  * [ ] docs: Add match/lag flags and brief diagnoses to the summary outputs

* [ ] Record environment and implementation differences that could affect comparability (Gym/Gymnasium vs ALE version, hardware precision, reward processing, action set, frameskip) and include them in a `results/summary/notes.md`.

  * [ ] docs: Document environment/implementation differences impacting score comparability



---

### Subtask 15 — Minimal Ablations & Sensitivity Analysis

**Objective:**  
This subtask validates core design choices of the DQN by running small-scale ablation studies. Selected variations—such as disabling reward clipping, changing the frame stack size from 4 to 2, or removing the target network—are tested on at least one benchmark game like Pong or Breakout for a reduced number of frames (around 5 million). Learning curves from these runs are compared with the baseline to measure the impact of each change on stability and convergence. The results and brief interpretations are summarized in `docs/papers/dqn_2013_notes.md`. Completion is confirmed when at least one ablation provides clear evidence of the necessity or effect of a key component in the DQN design.

**Checklist**

* [ ] Define targeted ablations as separate configs (e.g., disable reward clipping, change frame stack 4→2, remove target network) under `experiments/dqn_atari/configs/ablations/`, with clear names and comments explaining the change and expected effect.

  * [ ] feat: Add ablation configs for reward clipping off, stack=2, and no-target-network

* [ ] Select a benchmark game (Pong or Breakout) and launch each ablation for ≥5M frames using fixed seeds (e.g., 0,1,2) and deterministic settings; store outputs under `experiments/dqn_atari/runs/<game>/ablations/<ablation>/<seed>/`.

  * [ ] feat: Run ablation experiments on benchmark game for ≥5M frames across seeds

* [ ] Log and export comparable metrics for baseline vs ablations (reward curves, eval returns, TD-error stats, loss variance); ensure identical eval cadence and ε settings to isolate effects.

  * [ ] chore: Ensure consistent logging/eval cadence for baseline and ablation runs

* [ ] Generate comparison plots: overlay learning curves (reward vs frames), show eval score trajectories, and include stability indicators (e.g., TD-error variance); save under `results/ablations/<game>/<ablation>/plots/`.

  * [ ] feat: Add ablation comparison plots with stability indicators

* [ ] Summarize quantitative impact: compute deltas vs baseline (final mean eval return, area-under-curve, time-to-threshold) and mark stability outcomes (stable/unstable/diverged) in a small summary table.

  * [ ] feat: Export ablation summary table with deltas and stability flags

* [ ] Write a short report in `docs/papers/dqn_2013_notes.md` describing each ablation, expected rationale, observed effects on stability/convergence, and key takeaways; link to plots and run directories.

  * [ ] docs: Add brief ablation report with rationale, results, and links to artifacts

* [ ] Optionally add a convenience script `scripts/run_ablations.sh` to reproduce the set (configs, seeds, output paths) with a single command.

  * [ ] chore: Add ablation runner script for reproducible execution



---

### Subtask 16 — Aggregate Report & Interpretation

**Objective:**  
This subtask consolidates the entire reproduction effort into a structured report summarizing methodology, results, and key observations. Final learning curves, per-game performance tables, and example videos are collected and presented together with configuration details and averaged results across seeds. The report, saved to `docs/reports/dqn_results.md`, provides a concise but comprehensive account of how the reproduced implementation compares to the original DQN, explaining performance differences and noting reproducibility considerations. It concludes with a brief discussion of lessons learned and suggestions for potential extensions such as Double DQN or prioritized replay. The subtask is complete when the report and all referenced plots are finalized, readable, and ready for inclusion in the thesis.

**Checklist**

* [ ] Generate final artifacts: export per-game learning curves, aggregate score bar charts, and links/thumbnails to sample evaluation videos; save under `results/summary/` and reference them from the report.

  * [ ] docs: Export final plots and sample video links to results/summary and reference paths

* [ ] Write the aggregate report at `docs/reports/dqn_results.md` summarizing key metrics with seed averages, config details, and run metadata; include a brief methods overview and pointers to configs/checkpoints.

  * [ ] docs: Author dqn_results.md with metrics, seed averages, config notes, and artifact links

* [ ] Interpret outcomes relative to the original paper: discuss where results match or differ (e.g., convergence speed, final score), and attribute plausible causes (env versions, reward clipping, precision, budgets).

  * [ ] docs: Add comparison to paper with explanations for matches/gaps

* [ ] Add a concise “Lessons learned & future work” section outlining reproducibility takeaways and next steps (Double DQN, Prioritized Replay, dueling networks, multi-step targets).

  * [ ] docs: Include lessons learned and future work section in the report


---

### Subtask 17 — Code Quality, Testing, and Documentation

**Objective:**  
This subtask focuses on ensuring that the DQN implementation is reliable, maintainable, and easy to understand. The codebase is organized and cleaned, with automated tests verifying the correctness of critical components such as the replay buffer, model forward pass, and target network updates. Unit tests are implemented under `src/tests/` and can be executed with `pytest` or a similar framework. The project adheres to consistent formatting and style conventions using tools like Black, isort, and flake8. All major classes and functions include clear docstrings following a standard convention (Google or NumPy style). The documentation in `src/README.md` is updated to describe the architecture, module structure, and quickstart instructions for running experiments. Completion is indicated by a fully passing test suite, style compliance, and documentation that enables a new contributor to set up and train a DQN agent without external guidance.

**Checklist**

* [ ] Add unit tests under `src/tests/` covering: replay buffer sampling shapes and terminal-transition handling; model forward pass dimensions and absence of NaNs/Infs; target update timing and gradient flow; include fixtures for dummy observations and a seeded RNG; enable pytest markers for slow vs. fast tests.

  * [ ] test: Add replay/model/target-update unit tests with seeded fixtures and fast/slow markers

* [ ] Integrate continuous testing by adding a simple `pytest` runner script and optional CI config (e.g., GitHub Actions) that installs requirements, runs `pytest -q --disable-warnings`, and uploads a coverage report; fail build on test errors.

  * [ ] chore: Add CI test workflow and local test runner script

* [ ] Enforce code style: configure Black, isort, and flake8 (or ruff) with a shared line length; add a top-level `pyproject.toml` and a `pre-commit` configuration to auto-format and lint on commit.

  * [ ] build: Add Black/isort/flake8 (or ruff) via pyproject and pre-commit hooks

* [ ] Add concise docstrings for public modules, classes, and functions using a consistent style (Google or NumPy); include shapes/dtypes for tensors and side effects; verify with a docstring linter if available.

  * [ ] docs: Write consistent docstrings with tensor shapes/dtypes and behavior notes

* [ ] Update `src/README.md` with an architecture overview (modules, data flow), key entry points (training, evaluation, plotting), quickstart commands, and how to run tests/linters; link to configs and results directories.

  * [ ] docs: Refresh src/README with architecture, entry points, quickstart, and testing instructions

* [ ] Add coverage measurement (pytest-cov) with a sensible threshold (e.g., 70–80% for core modules); produce HTML and XML reports; exclude generated artifacts and scripts from coverage if needed.

  * [ ] build: Enable pytest-cov with reports and set a minimum coverage threshold

* [ ] Optional static checks: enable mypy on `src/` with a minimal config (ignore missing imports for third-party libs) and add a type-check target to CI; annotate key modules (replay, model, training loop) incrementally.

  * [ ] build: Introduce mypy type checking and initial annotations for core modules



---

### Subtask 18 — Reproduction Recipe Script

**Objective:**  
This subtask produces a single end-to-end script that allows a complete DQN training run to be reproduced from a clean environment. The script, implemented as `scripts/reproduce_dqn.sh` or a Python equivalent, automates environment setup, dependency installation, ROM downloads (where permitted), training, evaluation, and result visualization. It uses the verified baseline configuration and performs one full training cycle for a chosen game such as Pong, saving checkpoints, plots, and evaluation metrics in the standard directories. The process requires no manual folder creation or parameter adjustment. Successful completion is demonstrated when the script can reproduce a full baseline run on a fresh machine, yielding consistent metrics and artifacts identical to those of a verified reference experiment.

**Checklist**

* [ ] Add `scripts/reproduce_dqn.sh` (plus optional `.py`) to automate: create/activate virtualenv, install `requirements.txt`, download ROMs via `python -m AutoROM --accept-license` (if permitted), run `train_dqn.py` with the verified baseline config (e.g., Pong), then run evaluation and plotting.

  * [ ] feat: Add end-to-end reproduce_dqn script (env setup, ROMs, train, eval, plots)

* [ ] Make the script self-contained: auto-create standard directories (`experiments/dqn_atari/{runs,checkpoints,artifacts}`, `results/{plots,summary,videos}`), set deterministic seed, and avoid any manual path edits; accept optional `--game`, `--seed`, and `--frames` overrides.

  * [ ] chore: Ensure reproduce script creates folders and supports basic CLI overrides

* [ ] Capture environment provenance: within the script, save `pip freeze`, Python/CUDA/Torch versions, and ROM status to `experiments/dqn_atari/system_info.txt`; log commit hash and merged config to the run directory.

  * [ ] chore: Record environment and commit metadata during reproduction

* [ ] Evaluate and visualize automatically: after training, run the evaluation harness (K episodes, ε_eval) and generate plots via `scripts/plot_results.py`; save artifacts to `results/` with deterministic filenames.

  * [ ] feat: Wire evaluation and plotting steps into the reproduction pipeline

* [ ] Provide a verification check: compare produced metrics against a reference JSON/CSV (tolerance window) and print a pass/fail summary; exit non-zero on failure to guard regressions.

  * [ ] test: Add metric tolerance check against reference results with pass/fail output

* [ ] Document usage: add a short section in `README.md` showing one-command reproduction and expected outputs, plus runtime notes and hardware assumptions.

  * [ ] docs: Document reproduce script usage, outputs, and assumptions



---

### Subtask 19 — Automated Report Generation

**Objective:**  
This subtask introduces an automated reporting pipeline that compiles the results of an experiment into a concise, human-readable document. A script (`scripts/generate_report.py`) gathers relevant artifacts—including configuration files, plots, metrics, evaluation summaries, and hardware information—and compiles them into a Markdown or HTML report. Each report includes learning curves, evaluation tables, representative screenshots or video thumbnails, runtime information, and the Git commit hash used for the run. The reports are stored under `results/reports/<game>_<timestamp>.md` and can optionally be exported to PDF for inclusion in the thesis. The subtask is complete when running the report script after training automatically produces a complete, well-formatted summary suitable for publication or appendix inclusion.

**Checklist**
**Checklist**

* [ ] Implement `scripts/generate_report.py` that discovers the latest run for a given game, collects plots/metrics CSV/configs/runtime info/video thumbnails, and assembles a Markdown or HTML report with sections for learning curves, evaluation tables, runtime/hardware, and commit hash; write to `results/reports/<game>_<timestamp>.md` (and `.html` if selected).

  * [ ] feat: Add generate_report script to compile artifacts into Markdown/HTML

* [ ] Provide a small report template (Markdown/Jinja) for consistent layout: title, summary, methods blurb, learning curves, eval tables, sample media, and appendix with config/metadata; support relative links to artifacts.

  * [ ] docs: Add report template for standardized layout and relative artifact links

* [ ] Include hardware/environment provenance: embed Python/CUDA/Torch versions, GPU model, ROM status, and seed/config snapshot; read from existing `system_info.txt` and `meta.json`.

  * [ ] chore: Embed environment and metadata (versions, GPU, seed, commit) in report

* [ ] Generate evaluation tables automatically from CSV/JSONL (mean/median/std/min/max, episodes, eval_epsilon) and render learning curves by linking existing plots; fall back to inline quick plots if images are missing.

  * [ ] feat: Auto-build eval tables and link or inline learning curves

* [ ] Save outputs deterministically under `results/reports/<game>_<timestamp>/` with `index.md` (and optional `index.html`) plus copied assets (thumbnails); avoid breaking links if the directory is moved.

  * [ ] chore: Write report and copy assets into a self-contained report folder

* [ ] Add optional PDF export via Pandoc or nbconvert with a single flag (`--pdf`), handling image paths and page breaks; place PDF next to the Markdown/HTML.

  * [ ] feat: Support optional PDF export of the report

* [ ] Provide a simple CLI: `python scripts/generate_report.py --game Pong --run <path|auto> [--html] [--pdf] [--out results/reports/]`; validate inputs and fail with helpful errors if required artifacts are missing.

  * [ ] feat: Add CLI flags for game/run selection, formats, and output directory

* [ ] Add a smoke test using a small synthetic run directory to ensure the script completes and produces a minimal report; verify generated tables/links exist.

  * [ ] test: Add smoke test for report generation with synthetic artifacts



---

### Subtask 20 — Repository Organization & Archival

**Objective:**  
This subtask ensures that the project repository is well-structured, lightweight, and ready for long-term use or public release. All temporary files, redundant checkpoints, and caches are removed. The repository structure is standardized into clearly defined directories—`src/` for source code, `experiments/` for configurations and run data, `results/` for metrics and plots, `docs/` for reports and notes, and `scripts/` for automation utilities. Appropriate `.gitignore` and `.gitattributes` files are configured to exclude large or unnecessary files. A license file (e.g., MIT or CC-BY-4.0) is added, and the top-level `README.md` provides a concise overview of the project, references the DQN paper, explains installation and usage, and includes citation information. The subtask is complete when the repository passes a “fresh-clone test,” meaning it can be cloned and used to reproduce results without modification or missing dependencies.

**Checklist**

* [ ] Clean temporary data by removing intermediate logs, stale artifacts, large unused checkpoints, and cache files; keep only the latest N checkpoints per run and all evaluation summaries; document the retention rule in `results/README.md`.

  * [ ] chore: Prune temporary artifacts and old checkpoints with documented retention policy

* [ ] Organize top-level folders into a standard structure and move files accordingly:
  `src/` (source), `experiments/` (configs and runs), `results/` (plots, metrics, videos, reports), `docs/` (roadmap, notes), `scripts/` (automation).

  * [ ] chore: Standardize repository layout (src/, experiments/, results/, docs/, scripts/)

* [ ] Create .gitignore and .gitattributes to exclude large or generated files (e.g., `results/**`, `experiments/**/checkpoints/**`, `*.mp4`, `*.gif`, `*.pt`, `.DS_Store`, `__pycache__/`, `.ipynb_checkpoints/`); optionally route large binaries to Git LFS if needed.

  * [ ] build: Add .gitignore/.gitattributes for logs, checkpoints, and media (optional LFS rules)

* [ ] Add a LICENSE file (e.g., MIT for code; CC-BY-4.0 for thesis-related docs if desired) at the repo root and reference it from the README.

  * [ ] docs: Add LICENSE file and reference from README

* [ ] Write a concise top-level README.md with sections: project overview, DQN paper reference, installation steps, quickstart commands (train/eval/plot/report), and citation information; include pointers to configs and results directories.

  * [ ] docs: Author top-level README with overview, install, quickstart, and citation

* [ ] Perform a fresh-clone test: in a clean environment, run setup, download ROMs (if permitted), execute a short smoke run, and confirm outputs appear in the standard folders; add a helper `scripts/fresh_clone_check.sh` that automates these checks.

  * [ ] test: Add fresh-clone smoke test script and document pass criteria


---

### Subtask 21 — Thesis Integration & Final Write-Up

**Objective:**  
This subtask completes the research cycle by integrating the DQN reproduction results into the thesis document. The methodology section summarizes the goals, approach, and implementation details of the reproduction effort, while the results section presents learning curves, performance tables, and evaluation figures derived directly from the experiment outputs. The discussion section interprets discrepancies between reproduced and published results, highlighting possible causes such as environment differences or hardware constraints. A final reflection on reproducibility practices and lessons learned is included, along with a brief outline of potential future extensions such as Double DQN, dueling architectures, or prioritized replay. The subtask is complete when the thesis text, figures, and tables form a coherent narrative that traces the project from implementation through evaluation and analysis, directly supported by the contents of the repository.

**Checklist**

* [ ] Write the Methods section summarizing goals, datasets/environments, implementation details, and challenges; reference configs, seeds, and evaluation protocol; include a brief diagram of the training loop and links to code entry points.

  * [ ] docs: Add Methods section with setup, implementation details, and evaluation protocol references

* [ ] Insert key results into the Results chapter: per-game learning curves, aggregate score tables, and representative evaluation videos or screenshots; ensure captions, figure numbers, and cross-references; copy assets to `docs/thesis/assets/`.

  * [ ] docs: Embed plots/tables/videos in Results chapter with captions and asset paths

* [ ] Analyze discrepancies vs. the original DQN paper: quantify gaps (faster/slower convergence, final score deltas), list environment/toolchain differences (ALE/Gym versions, reward clipping, precision), and provide a short diagnosis table with hypothesized causes.

  * [ ] docs: Add discrepancy analysis with comparison table and hypotheses

* [ ] Reflect on reproducibility practices: document environment pinning, seeding strategy, deterministic flags, checkpoint/resume, logging/plots, and report-generation pipeline; summarize what worked and what to improve.

  * [ ] docs: Add reproducibility reflection covering pinning, seeding, determinism, and logging

* [ ] Add a Future Work section outlining extensions (Double DQN, Prioritized Replay, dueling networks, multi-step targets, distributional/Rainbow variants) and proposed next-step experiments with brief rationale.

  * [ ] docs: Add Future Work section with prioritized extensions and rationale



---


