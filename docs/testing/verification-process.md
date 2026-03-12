# Run Verification Process

This document describes the automated and manual checks used to
verify that each training configuration produces the expected
behavior. It exists so that validation decisions are traceable and
reproducible.


## Background

The project trains DQN variants on the Atari-100K benchmark with
optional feature flags: data augmentation, SPR (Self-Predictive
Representations), and Rainbow extensions. Each combination is
defined by a YAML config file. Because features are opt-in via
config flags, a silent wiring bug can cause a run to complete
successfully while the requested feature never activates. This
happened with SPR: 12 runs completed without errors but produced
no SPR-specific data.

To prevent this class of bug going forward, the project uses
three layers of automated verification plus a manual inspection
protocol.


## Config types

| Type         | Config example                          | Features active                          |
|--------------|-----------------------------------------|------------------------------------------|
| base         | `atari100k_boxing.yaml`                 | Vanilla DQN                              |
| aug          | `atari100k_boxing_aug.yaml`             | DQN + random shift augmentation          |
| spr          | `atari100k_boxing_spr.yaml`             | DQN + SPR auxiliary loss                 |
| both         | `atari100k_boxing_both.yaml`            | DQN + augmentation + SPR                 |
| rainbow      | `atari100k_boxing_rainbow.yaml`         | Rainbow DQN (distributional, NoisyNets, PER, dueling) |
| rainbow_spr  | `atari100k_boxing_rainbow_spr.yaml`     | Rainbow + SPR                            |


## Layer 1: Unit and integration tests (pytest)

Run with `pytest tests/ -x` before every commit.

**What they cover:**

- `tests/test_spr_integration.py::TestSPREndToEnd` -- Loads a SPR
  config through `initialize_components()`, runs training steps,
  and asserts that `spr_loss` and `cosine_similarity` are populated
  on every update step. Also checks that SPR parameters are added
  to the optimizer and that dropout is applied.

- `tests/test_rainbow_train_integration.py` -- Verifies Rainbow
  model creation, PER buffer usage, distributional loss, and
  NoisyNets epsilon=0 through the full training path.

- `tests/test_rainbow_backward_compat.py` -- Checks that NoisyNets
  forces epsilon to zero and that `keep_last_n` checkpoint retention
  works correctly.

**What they do NOT cover:**

- Whether features produce correct results at scale (400K frames).
- Numerical stability over long runs (gradient explosions, NaN).
- Whether augmentation actually improves representations (only that
  it activates).


## Layer 2: Smoke test (scripts/smoke_test_configs.py)

Runs each config type for a small number of frames (~2K-5K) on CPU,
then validates the output CSV.

```bash
python scripts/smoke_test_configs.py --types base aug spr both rainbow rainbow_spr --frames 2000
```

**Checks performed:**

- Expected CSV columns are populated (at least one non-empty value).
- Columns for inactive features are empty.
- NoisyNets configs have epsilon=0 throughout.

**What it catches:**

- Config flags silently ignored (feature defined in YAML but never
  read by `train_dqn.py`).
- Wrong model type created for a config.
- Missing CSV logging for a feature.

**What it does NOT catch:**

- Correctness at scale. A feature can activate (columns populated)
  but produce degenerate values over a full run.


## Layer 3: Post-run validation (scripts/validate_run.py)

Validates a completed 400K-frame run directory against its saved
`config.yaml`.

```bash
python scripts/validate_run.py experiments/dqn_atari/runs/<run-name>/
python scripts/validate_run.py experiments/dqn_atari/runs/<run-name>/ --json
```

**Checks performed:**

1. Run directory and config.yaml exist and parse correctly.
2. `progress.json` shows `status=complete` and `percent>=99`.
3. Core CSV columns populated: step, epsilon, replay_size, loss,
   td_error, grad_norm.
4. SPR columns populated when `spr.enabled=true`; empty otherwise.
5. Rainbow columns populated when `rainbow.enabled=true`; empty
   otherwise.
6. SPR loss values are not all zero and contain no NaN.
7. NoisyNets epsilon=0 on every row when `rainbow.noisy_nets=true`.
8. Correct number of periodic checkpoints present.
9. `best_model.pt` exists when `save_best=true`.


## Layer 4: Manual CSV inspection

After downloading a completed run, open `csv/training_steps.csv`
and verify the following (in addition to the automated checks).

**All configs:**

- loss is not stuck at a constant value and not exploding.
- td_error is finite and not constant.
- grad_norm is finite (no inf or NaN).
- learning_rate matches the config value.

**SPR configs (spr, both, rainbow_spr):**

- spr_loss is populated on every training row, not just some.
- spr_loss values are negative (cosine similarity loss).
- spr_loss magnitude is in a reasonable range (typically -0.5
  to -1.0).
- spr_loss shows a trend over time (not flat or random).
- cosine_similarity is in [-1, 1] and trends upward.
- ema_update_count increments monotonically.
- ema_update_count final value is approximately equal to total
  updates minus warmup steps.

**Rainbow configs (rainbow, rainbow_spr):**

- epsilon is 0.0 on every row (NoisyNets).
- distributional_loss is positive and finite.
- mean_is_weight is positive (importance sampling weights).
- mean_priority is positive.
- priority_entropy is positive (not degenerate).
- beta starts at beta_start and anneals toward beta_end.

**Augmentation configs (aug, both):**

- stdout contains "Augmentation: random_shift" or similar.
- Training is measurably slower than the base config (augmentation
  adds compute).


## Validation protocol for new config types

Before launching a batch of runs for a config type that has never
had a confirmed valid full run:

1. **Pre-flight**: Run the smoke test locally. If it fails, debug
   and fix before proceeding.
2. **Single-game validation**: Run Boxing for the full 400K frames
   on a GPU (Colab).
3. **Automated validation**: Download the run and run
   `validate_run.py`. Every check must pass.
4. **Manual CSV inspection**: Follow the checklists above.
5. **Checkpoint verification**: Confirm expected checkpoint count,
   load one checkpoint, run re-evaluation, compare to training
   eval scores.
6. **Eval sanity**: Final eval mean return is in a plausible range
   for the game and not degenerate.
7. **Record results**: Log the validation outcome in
   `docs/testing/validation-log.md` with run name, config type,
   date, final eval score, and pass/fail status.
8. **Batch launch**: Only after all checks pass, launch the
   remaining games.


## Validation status

Results of validation runs are recorded in
[validation-log.md](validation-log.md).
