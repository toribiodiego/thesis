# Validation Log

Records the outcome of each validation run. A config type must
pass validation here before a full batch (6 games) is launched.

Format: one entry per validation run, newest first.


## Validated config types

| Config       | Validated? | Validation run             | Date       | Notes                          |
|--------------|------------|----------------------------|------------|--------------------------------|
| base         | Yes        | (6-game batch, all valid)  | 2026-03-10 | Core metrics populated         |
| aug          | Yes        | (6-game batch, all valid)  | 2026-03-10 | Augmentation active            |
| spr          | Pending    | --                         | --         | Requires validation run        |
| both         | Pending    | --                         | --         | Requires validation run        |
| rainbow      | Yes        | (6-game batch, all valid)  | 2026-03-11 | All Rainbow metrics, epsilon=0 |
| rainbow_spr  | Pending    | --                         | --         | Never run, requires validation |


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

<!-- Add entries below as validation runs complete. Template:

### <config_type> -- <game> -- <date>

- **Run name**: `<run-name>`
- **Config file**: `<config-file>`
- **Seed**: <seed>
- **Smoke test**: Pass/Fail
- **validate_run.py**: Pass/Fail (<N>/<N> checks)
- **Manual CSV inspection**: Pass/Fail
- **Checkpoint verification**: Pass/Fail
- **Final eval mean return**: <score>
- **Overall**: PASS / FAIL
- **Notes**: <any observations>

-->
