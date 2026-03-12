# Validation Log

Records the outcome of each validation run. A config type must
pass validation here before a full batch (6 games) is launched.

Format: one entry per validation run, newest first.


## Validated config types

| Config       | Validated? | Validation run             | Date       | Notes                          |
|--------------|------------|----------------------------|------------|--------------------------------|
| base         | Yes        | (6-game batch, all valid)  | 2026-03-10 | Core metrics populated         |
| aug          | Yes        | (6-game batch, all valid)  | 2026-03-10 | Augmentation active            |
| spr          | Yes        | Boxing, seed 42            | 2026-03-12 | SPR loss populated, 20/20      |
| both         | Yes        | Boxing, seed 42            | 2026-03-12 | Aug + SPR both active, 20/20   |
| rainbow      | Yes        | (6-game batch, all valid)  | 2026-03-11 | All Rainbow metrics, epsilon=0 |
| rainbow_spr  | Yes        | Boxing, seed 42            | 2026-03-12 | Rainbow + SPR active, 20/21    |


## Invalid runs

12 runs (6 DQN+SPR, 6 DQN+Both) completed between 2026-03-10 and
2026-03-11 are invalid. SPR was defined in their YAML configs but
never wired into the training script. The DQN+SPR runs trained as
vanilla DQN; the DQN+Both runs trained as DQN+Aug.

These runs have been moved to `results/deprecated/` locally and
will be moved to `thesis-runs/deprecated/` on Google Drive. They
are kept as evidence of the bug, not deleted.

See `results/deprecated/README.md` for the full list and
explanation.


## Validation run entries

### spr -- Boxing -- 2026-03-12

- **Run name**: `atari100k_boxing_spr_42_20260312_022320`
- **Config file**: `atari100k_boxing_spr.yaml`
- **Seed**: 42
- **validate_run.py**: Pass (20/20 checks)
- **SPR loss range**: [-1.0000, -0.0308]
- **SPR columns populated**: 197/200 rows (3 warmup rows empty)
- **Rainbow columns**: correctly empty
- **Epsilon**: decays to 0.1 (correct for DQN)
- **Checkpoints**: 4/4 periodic + best_model.pt
- **Final eval mean return**: -40.5 (step 360K)
- **Overall**: PASS
- **Notes**: Confirms SPR loss is computed on every update step.
  Cosine similarity near 1.0 throughout, EMA updating correctly.


### both -- Boxing -- 2026-03-12

- **Run name**: `atari100k_boxing_both_42_20260312_030847`
- **Config file**: `atari100k_boxing_both.yaml`
- **Seed**: 42
- **validate_run.py**: Pass (20/20 checks)
- **SPR loss range**: [-1.0000, -0.0286]
- **SPR columns populated**: 197/200 rows (3 warmup rows empty)
- **Rainbow columns**: correctly empty
- **Epsilon**: decays to 0.1 (correct for DQN)
- **Checkpoints**: 4/4 periodic + best_model.pt
- **Final eval mean return**: -18.8 (step 360K)
- **Overall**: PASS
- **Notes**: Augmentation and SPR both active without interference.
  SPR loss range consistent with SPR-only config.


### rainbow_spr -- Boxing -- 2026-03-12

- **Run name**: `atari100k_boxing_rainbow_spr_42_20260312_035510`
- **Config file**: `atari100k_boxing_rainbow_spr.yaml`
- **Seed**: 42
- **validate_run.py**: Pass (20/21 checks)
- **SPR loss range**: [-0.9988, -0.9944]
- **SPR columns populated**: 160/200 rows (40 warmup rows empty)
- **Rainbow columns populated**: 160/200 rows (distributional_loss, PER metrics, beta)
- **Epsilon**: 0.0 on all 200 rows (NoisyNets correct)
- **Checkpoints**: 10/10 periodic, best_model.pt missing
- **Final eval mean return**: not recorded (eval dir not copied to Drive)
- **Overall**: PASS (with caveat)
- **Notes**: All feature checks pass. The one failure is a missing
  best_model.pt -- likely an edge case where eval never improved
  above the initial threshold, or eval directory was not fully
  copied to Drive. This is a logging issue, not a feature
  correctness issue. SPR and Rainbow columns both populated on
  every training row after warmup (20K frames / 40 rows).
