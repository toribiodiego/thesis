# Scripts

- `run_dqn.sh` – Convenience wrapper around `python src/train_dqn.py --config ...`. Pass additional CLI overrides after the config path.
  ```bash
  bash experiments/dqn_atari/scripts/run_dqn.sh experiments/dqn_atari/configs/pong.yaml --total_frames 2000000
  ```
- (planned) `eval_dqn.sh` – Will load checkpoints and run the evaluation harness for a fixed number of episodes.
- (planned) `dry_run.sh` – Will run a short random-policy rollout to verify preprocessing and logging.
