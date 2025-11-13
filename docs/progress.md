# DQN Reproduction Progress & Context

This document tracks completed work and provides context for continuing the DQN reproduction effort. It serves as a quick-start brief for Claude or any contributor joining the project.

_Reminder:_ Use the commit prefixes in `docs/git_commit_guide.md` when pushing changes referenced here so the progress history remains consistent.

---

## Current Status

**Active Milestone:** M1 – Environment + tooling smoke test
**Active Subtask:** Subtask 1 – Choose paper games, pin versions, and scaffold runs
**Overall Progress:** Early scaffolding phase; no training code yet

---

## Completed Work

### Repository Setup
- [x] Created repository structure with standard directories:
  - `src/` – Source code (agents, wrappers, replay, utils)
  - `experiments/dqn_atari/` – DQN experiment configs and scripts
  - `envs/` – Environment setup and dependencies
  - `docs/` – Roadmap, design docs, paper notes
  - `notes/` – Planning checklists and paper reviews
- [x] Initialized Git repository with initial commits
- [x] Created README files for major directories explaining their purpose

### Environment & Dependencies
- [x] Created `envs/requirements.txt` with core dependencies:
  - PyTorch, Gymnasium, ale-py (Atari Learning Environment)
  - NumPy, matplotlib, tensorboard
  - Development tools (pytest, black, flake8)
- [x] Created `envs/setup_env.sh` script for virtualenv creation and activation
- [x] Documented environment setup in `envs/README.md`

### Documentation & Planning
- [x] Read and annotated DQN paper (2013): `docs/papers/dqn_2013_notes.md`
- [x] Created comprehensive roadmap with 21 subtasks: `docs/roadmap.md`
  - [x] Restored detailed checklists from reference commit (bdd2c67) with actionable bullet points
- [x] Documented work package 1 checklist: `notes/wp1_planning_checklist.md`
- [x] Established paper review template: `notes/templates/paper_review.md`
- [x] Created design directory structure for architecture docs

### Experiment Scaffolding
- [x] Created `experiments/dqn_atari/` directory structure:
  - `configs/` – Configuration files directory (empty, ready for YAML files)
  - `scripts/` – Launch scripts directory (empty, ready for run scripts)
  - `notes.md` – Experiment log file
  - `README.md` – Experiment documentation

---

## Next Actions (Subtask 1)

The immediate next work is completing **Subtask 1** to establish the experimental foundation:

1. **Choose games and create config stubs:**
   - Select Pong, Breakout, Beam Rider as initial games
   - Create `experiments/dqn_atari/configs/pong.yaml`
   - Create `experiments/dqn_atari/configs/breakout.yaml`
   - Create `experiments/dqn_atari/configs/beam_rider.yaml`
   - Document game selection in `experiments/dqn_atari/README.md`

2. **Document environment IDs and ROM setup:**
   - Add table with game IDs (`ALE/Pong-v5`, etc.), action sets (minimal)
   - Create `scripts/setup_roms.sh` with AutoROM commands
   - Update `experiments/dqn_atari/README.md` with ROM acquisition instructions

3. **Pin exact dependency versions:**
   - Update `envs/requirements.txt` with exact versions:
     - Python 3.10.13 (document in README)
     - PyTorch 2.4.1+cu121 (or CPU equivalent)
     - Gymnasium 0.29.1
     - ale-py 0.8.1
   - Document ALE runtime settings (frameskip=4, repeat_action_probability=0.0)
   - Create `scripts/capture_env.sh` to write system info

4. **Define evaluation protocol:**
   - Create `experiments/dqn_atari/configs/base.yaml` with:
     - Evaluation epsilon (0.05)
     - Termination policies (life-loss for training, full-episode for eval)
     - Number of eval episodes (10)
     - Reward clipping {−1, 0, +1}
     - Frame budgets (10–20M for full runs, smaller for smoke tests)

5. **Add reproducibility utilities:**
   - Create `src/utils/` directory
   - Implement `src/utils/repro.py` with `set_seed()` function
   - Wire `--seed` flag in future training scripts
   - Implement metadata snapshot saving (commit hash, config, seed) to `meta.json`

6. **Scaffold run launcher:**
   - Create per-game config YAMLs with game-specific overrides
   - Implement `scripts/run_dqn.sh` that will launch `src/train_dqn.py`
   - Add `--dry-run` mode for random-policy rollouts
   - Ensure dry run saves: preprocessed frames, action lists, minimal eval report

---

## Repository Layout

```
thesis/
├── README.md                          # Top-level project overview
├── docs/
│   ├── roadmap.md                    # Main task plan with 21 subtasks
│   ├── progress.md                   # This file - completed work & context
│   ├── design/                       # Architecture and design docs (planned)
│   └── papers/
│       └── dqn_2013_notes.md        # DQN paper annotations
├── notes/
│   ├── wp1_planning_checklist.md    # Work Package 1 tracking
│   ├── revisit.md                   # Open questions log
│   └── templates/
│       └── paper_review.md          # Template for paper summaries
├── envs/
│   ├── README.md                    # Environment setup documentation
│   ├── requirements.txt             # Python dependencies (needs version pinning)
│   └── setup_env.sh                 # Virtualenv creation script
├── src/                             # Source code (to be implemented)
│   ├── README.md                    # Source code overview
│   ├── agents/                      # DQN agent implementations (pending)
│   ├── replay/                      # Experience replay buffers (pending)
│   ├── wrappers/                    # Atari env wrappers (pending)
│   ├── utils/                       # Utilities (logging, seeding, etc.) (pending)
│   └── tests/                       # Unit tests (pending)
├── experiments/
│   └── dqn_atari/
│       ├── README.md                # DQN experiment documentation
│       ├── configs/                 # YAML config files (empty - Subtask 1)
│       ├── scripts/                 # Launch scripts (empty - Subtask 1)
│       └── notes.md                 # Experiment log
└── results/                         # Generated plots/tables (planned, not created yet)
```

---

## Key Design Decisions

### Configuration Strategy
- **Approach:** Base YAML + per-game overrides
- **Location:** `experiments/dqn_atari/configs/`
- **Merge strategy:** To be implemented in Subtask 8

### Logging Strategy
- **Options considered:** TensorBoard, Weights & Biases, local CSV
- **Decision:** TBD (leaning toward TensorBoard + CSV for portability)
- **Implementation:** Subtask 10

### Evaluation Protocol
- **Training termination:** Life-loss as terminal (configurable)
- **Eval termination:** Full episode only
- **Eval epsilon:** 0.05 (low exploration)
- **Eval frequency:** Every 250k frames (configurable)
- **Eval episodes:** 10 per checkpoint

### Reproducibility
- **Seeding:** Unified utility in `src/utils/repro.py`
- **Determinism:** Optional deterministic mode with cuDNN settings
- **Metadata:** Every run saves commit hash, merged config, seed

---

## Paper Notes Summary

Key insights from DQN 2013 paper (`docs/papers/dqn_2013_notes.md`):

- **Preprocessing:** 210×160×3 RGB → 84×84 grayscale, 4-frame stack
- **Network:** Conv(16,8×8,s4) → Conv(32,4×4,s2) → FC(256) → linear(|A|)
- **Replay:** 1M capacity, uniform sampling, 50k warm-up
- **Hyperparameters:**
  - Learning rate: 2.5e-4
  - Discount γ: 0.99
  - Batch size: 32
  - Target network update: every 10k steps
  - Training frequency: every 4 steps
  - Optimizer: RMSProp (ρ=0.95, ε=0.01)
- **Exploration:** ε annealed linearly from 1.0 to 0.1 over 1M frames
- **Frame skip:** 4 (action repeated, rewards accumulated)
- **Reward clipping:** {−1, 0, +1}

---

## Git Workflow

**Main branch:** `main` (currently clean)

**Recent commits:**
- `bdd2c67` - docs: add roadmap.md with detailed DQN plan
- `ddd4ede` - docs: scaffold DQN roadmap structure
- `a88776a` - initial commit

**Commit conventions:**
- Prefix with type: `feat:`, `docs:`, `test:`, `build:`, `chore:`
- Include Claude Code attribution for AI-assisted commits
- Keep commits focused and atomic

---

## Hardware & Environment Notes

**Development environment:**
- Platform: macOS (Darwin 23.5.0)
- Python: TBD (target 3.10.13)
- CUDA: TBD (will support both CPU and GPU paths)

**Compute plan:**
- Initial development: CPU (CartPole, short Atari tests)
- Full training: GPU required (10–20M frames per game)
- Estimated time per game: TBD (depends on hardware)

---

## Open Questions & Decisions Needed

Tracked in `notes/revisit.md` and work package checklists:

1. **Configuration framework:** Hydra vs. pure YAML + dataclasses?
2. **Logging backend:** W&B (requires network) vs. TensorBoard vs. CSV?
3. **Testing strategy:** Pytest conventions, coverage threshold?
4. **Code style:** Black line length, import ordering?

---

## Quick Start for Claude

When resuming work on this project:

1. **Check current status:** Read this file and `docs/roadmap.md`
2. **Identify active subtask:** See "Next Actions" section above
3. **Review implementation details:** Check `docs/implementation_guide.md` for execution-ready code snippets and verification commands
4. **Review paper notes:** Check `docs/papers/dqn_2013_notes.md` for hyperparameters and algorithm details
5. **Follow checklist format:** Use nested checkboxes with type labels (feat/docs/test/build/chore)
6. **Update progress:** Mark completed items, move to next subtask when done
7. **Maintain formatting:** Keep roadmap/progress docs consistent with existing style

---

## References

- **Primary paper:** *Playing Atari with Deep Reinforcement Learning* (Mnih et al., 2013) - arXiv:1312.5602
- **Nature paper:** *Human-level control through deep reinforcement learning* (Mnih et al., 2015)
- **Roadmap:** `docs/roadmap.md` – Complete 21-subtask implementation plan (high-level objectives and checklists)
- **Implementation guide:** `docs/implementation_guide.md` – Execution-ready details with code snippets, file paths, and verification commands
- **Paper notes:** `docs/papers/dqn_2013_notes.md` – Detailed annotations and hyperparameters
