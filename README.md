# Thesis RL Experiments

Masters thesis on sample- and data-efficient reinforcement learning. First milestone: reproduce DQN (Mnih et al., 2013) with reusable tooling for future algorithms (MuZero, EfficientZero, CURL, DrQ, SPR).

## Quick Start

```bash
# Setup environment
source envs/setup_env.sh

# See roadmap for detailed implementation plan
cat docs/roadmap.md
```

## Structure

```
├── docs/roadmap.md              # Project plan with 21 subtasks
├── envs/                        # Dependencies and setup scripts
├── src/                         # Reusable RL modules
├── experiments/dqn_atari/       # DQN configs and scripts
└── notes/                       # Paper summaries and planning
```

## Workflow

1. Check `docs/roadmap.md` for current subtask
2. Implement following checklist items
3. Use commit prefixes from `docs/git_commit_guide.md`
4. Mark completed items in roadmap
