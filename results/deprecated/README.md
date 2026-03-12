# Deprecated Runs

These 12 runs are invalid due to a bug where SPR (Self-Predictive
Representations) was defined in the YAML config but never wired
into the training script (`train_dqn.py`). The runs completed
without errors and produced reasonable-looking scores, but the
SPR auxiliary loss was never computed.

- The 6 DQN+SPR runs trained as vanilla DQN (no SPR loss).
- The 6 DQN+Both runs trained as DQN+Aug (augmentation active,
  but no SPR loss).

The bug was fixed in commit 337905d, which initializes SPR
components in `train_dqn.py` when `spr.enabled=true`. Automated
smoke tests and post-run validators now catch this class of bug.

These runs are kept as evidence. Do not delete them.


## DQN+SPR (ran as vanilla DQN)

- atari100k_boxing_spr_42_20260310_232848
- atari100k_crazy_climber_spr_42_20260310_230914
- atari100k_frostbite_spr_42_20260311_000206
- atari100k_kangaroo_spr_42_20260310_235821
- atari100k_road_runner_spr_42_20260310_231014
- atari100k_up_n_down_spr_42_20260311_001035


## DQN+Both (ran as DQN+Aug)

- atari100k_boxing_both_42_20260311_022139
- atari100k_crazy_climber_both_42_20260311_020133
- atari100k_frostbite_both_42_20260311_025205
- atari100k_kangaroo_both_42_20260311_024627
- atari100k_road_runner_both_42_20260311_020220
- atari100k_up_n_down_both_42_20260311_030203


## Date deprecated

2026-03-11
