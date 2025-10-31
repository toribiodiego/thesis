## Thesis RL Experiments

This repository gathers experiments for a thesis on sample- and data-efficient reinforcement learning.  
The immediate goal is to **recreate the original Deep Q-Network (DQN) Atari results** (“Playing Atari with Deep Reinforcement Learning”) to establish a shared experimentation stack. The DQN reproduction will set the foundation for future work on MuZero-style planning, self-predictive representations, CURL, DrQ, and robustness studies.

- Roadmap: `docs/dqn_roadmap.md`
- Experiment space: `experiments/dqn_atari/`
- Shared modules scaffold: `src/`
- Environment specs (pending): `envs/`

As the project evolves, each experiment will reuse the shared tooling while contributing reports and insights back to the common documentation in `docs/`.
