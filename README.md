## Thesis RL Experiments

Home for our masters thesis on sample- and data-efficient reinforcement learning. The first milestone is reproducing “Playing Atari with Deep Reinforcement Learning” (DQN) while building tooling that future projects (MuZero, EfficientZero, CURL, DrQ, SPR) can reuse.

### Repo Map
| Path | Purpose |
|------|---------|
| `docs/roadmap.md` | Living project plan, milestones, and detailed subtasks. |
| `docs/progress.md` | Completed work summary and context for resuming work. |
| `docs/design/` | Architecture sketches, configuration strategy, workflow diagrams. |
| `notes/` | Paper summaries, revisit log, and planning checklists (public-friendly). |
| `envs/` | Reproducible dependency specs (`requirements.txt`) and `setup_env.sh`. |
| `src/` | Shared source modules (agents, env wrappers, replay buffers, configs, utils). |
| `experiments/dqn_atari/` | DQN-specific configs, run scripts, and experiment logs. |
| `reports/` (planned) | Generated plots, tables, and write-ups for the thesis. |

### Working Rhythm
1. **Plan** – Update `docs/roadmap.md` for scope changes; capture decisions in `docs/design/`.
2. **Read & Reflect** – Log each paper in `notes/papers/` using the template; track open questions in `notes/revisit.md`.
3. **Implement & Run** – Use `envs/setup_env.sh` to create a venv, then iterate inside `src/` and `experiments/dqn_atari/`.
4. **Document Results** – Archive metrics/plots under `reports/` and link back from the roadmap/notes.

### Thesis TODO
- [ ] Verify the new environment setup script (`envs/setup_env.sh`) on both CPU- and CUDA-capable machines.
- [ ] Finish the CartPole → Atari smoke-test pipeline (replay buffer, Nature CNN, logging hooks).
- [ ] Acquire Atari ROMs via AutoROM and script a deterministic evaluation harness.
- [ ] Stand up report-generation scripts so DQN runs emit thesis-ready figures/tables.

Once the DQN baseline is stable, subsequent algorithms should slot into the same structure with minimal friction.

### Contributing
- Follow the commit-message prefixes in `docs/git_commit_guide.md` (`feat:`, `fix:`, `docs:`, etc.) so history stays consistent.
- Update `docs/roadmap.md` and `docs/progress.md` when you complete or reprioritize checklist items.
- Capture experiment-specific notes in `experiments/dqn_atari/notes.md` so future runs have context.
