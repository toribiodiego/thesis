# Configs

`base.yaml` captures the default hyperparameters and runtime settings for the DQN reproduction. Create per-game overrides by copying the base file into `pong.yaml`, `breakout.yaml`, etc., and only changing the fields that differ (e.g., `env.id`, frame budgets, or evaluation windows).

Sample usage (once `train_dqn.py` exists):
```bash
python src/train_dqn.py --config experiments/dqn_atari/configs/base.yaml
```
