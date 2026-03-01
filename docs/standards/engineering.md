# Engineering Standards

Standards for writing reliable, reproducible, and efficient research
code. The goal is work that produces trustworthy results, runs
efficiently on limited compute, and remains understandable months later.

For development setup and commit conventions, see
[Contributing Guide](../../CONTRIBUTING.md). For documentation
conventions, see [Documentation Standards](./documentation.md).

<br><br>

## Core Principles

- **Reproducibility**: pin dependencies, fix seeds, log configurations.
  Anyone (including future you) must be able to reproduce a result.
- **Evidence**: support design and performance decisions with measurable
  data or a clear rationale. No "this feels faster."
- **Efficiency**: minimize unnecessary compute. GPU hours are expensive
  -- prefer bounded resource use and early stopping for dead runs.
- **Clarity**: keep changes focused, update docs when behavior changes,
  and add tests for new code paths.
- **Correctness first**: a fast but wrong experiment wastes more time
  than a slow but correct one.

<br><br>

## Evidence-Based Decisions

When making technical decisions, provide supporting evidence:

**For performance choices:**
- Include benchmark results comparing alternatives
- Measure before and after metrics (training time, memory, throughput)
- Document test conditions (game, number of steps, hardware, batch size)

**For architectural decisions:**
- Explain the trade-offs considered
- Reference similar patterns in the codebase or published work
- Document why alternatives were rejected

**Example:**
```text
perf: switch replay buffer storage to uint8

Reduces memory usage from 3.2GB to 800MB for 100K capacity buffer.
Allows larger replay buffers on Colab's 12GB RAM without OOM.
Trade-off: requires dtype conversion on sample, adds ~2ms per batch.

Tested with Breakout, 100K capacity, batch size 32.
```

<br><br>

## Reproducibility

Every experiment must be reproducible from its logged configuration:

**Seed management:**
- Fix random seeds for Python, NumPy, and PyTorch at the start of
  every training run
- Log the seed value to W&B or the experiment config
- Use separate seeds for environment, replay sampling, and network
  initialization when possible

**Dependency pinning:**
- `requirements.txt` pins exact versions (`torch==2.1.0`, not `torch>=2`)
- Document CUDA version and GPU type for training runs
- Use `setup/capture_env.sh` to snapshot the full environment

**Configuration logging:**
- Log the full YAML config to W&B at the start of every run
- Never rely on default values silently -- make all hyperparameters
  explicit in the config file
- Record which commit hash produced each result

<br><br>

## Repository Hygiene

Keep the repository clean and navigable:

**Repo root:** Only project-level files belong at the root
(`train_dqn.py`, `requirements.txt`, `README.md`, `.gitignore`).
Configs go in `experiments/`, scripts in `scripts/`, docs in `docs/`.

**Work-in-progress files:** Investigation notes, draft scripts, and
planning artifacts go in `tmp/` (gitignored). Only finalized
documentation gets committed to `docs/`. This keeps the committed tree
free of transient artifacts.

**Experiment artifacts:** Raw outputs (plots, logs, checkpoints) go in
`output/` (gitignored) or W&B. Only curated results and analysis
belong in `docs/reports/`.

**Tool-specific configuration:** Editor and IDE configs (`.vscode/`,
`.idea/`) stay in `.gitignore`, not committed.

<br><br>

## Focused Changes and Tests

Keep changes small and well-tested:

**Scope control:**
- One logical change per commit
- Avoid mixing refactoring with feature work
- Split large changes into reviewable increments

**Test coverage:**
- Add tests for new code paths
- Include both happy path and edge cases
- Test boundary conditions (empty replay buffer, episode boundaries,
  single-step episodes, observation shape mismatches)
- Update existing tests when behavior changes
- Run `pytest tests/ -x` before every commit

**Documentation:**
- Update relevant docs in `docs/` when behavior changes
- Add code comments for non-obvious logic (especially RL-specific
  tricks like target network updates, frame stacking, reward clipping)

<br><br>

## Compute Efficiency

GPU hours are the primary constraint. Write code that respects this:

**Training runs:**
- Use checkpointing so runs can resume after interruption (Colab
  session timeouts, preemptible instances)
- Log metrics frequently enough to detect divergence early
- Implement early stopping or kill criteria for clearly failed runs
- Profile before optimizing -- measure where time actually goes

**Memory management:**
- Store observations as `uint8` in replay buffer, convert to `float32`
  only at training time
- Monitor GPU memory usage, especially with larger CNN backbones
- Use `torch.no_grad()` for evaluation and target network computation

**Batching and I/O:**
- Batch W&B logging to avoid per-step overhead
- Pre-compute augmentations on GPU when possible (avoid CPU-GPU
  transfers per batch)
- Use `pin_memory=True` in DataLoaders for faster GPU transfers

**Example checklist for a new experiment config:**
- [ ] Seeds are set and logged
- [ ] Full config is logged to W&B
- [ ] Checkpointing is enabled at a reasonable interval
- [ ] Evaluation frequency is set (not every step)
- [ ] Expected runtime is estimated before launching
- [ ] Commit hash is recorded

<br><br>

## Experiment Hygiene

Treat experiments as first-class engineering work:

**Before running:**
- Verify the config produces expected behavior with a short dry run
  (100 steps, check shapes, losses, logging)
- Confirm the correct checkpoint will be saved and can be loaded
- Estimate wall-clock time and check it fits the compute budget

**During runs:**
- Monitor loss curves for obvious divergence
- Check GPU utilization to ensure the GPU is not starved by CPU
  preprocessing
- Save checkpoints at regular intervals, not just at the end

**After runs:**
- Record results in `output/` summary files and W&B
- Compare against baseline numbers before drawing conclusions
- Document any anomalies or unexpected behavior in the research log

<br><br>

[Back to Contributing Guide](../../CONTRIBUTING.md)
