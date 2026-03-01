# Archive and Retention Plan

> **Status**: REFERENCE | Policies for managing artifacts, logs, and experiment outputs.
> **Purpose**: Define what to keep, archive, or delete to maintain a clean and reproducible repository.

## Purpose

This document establishes retention rules and gitignore policies for managing the growing volume of training artifacts, experiment outputs, and temporary files. It ensures reproducibility while preventing repository bloat.

## Inputs/Outputs

**Inputs** (what gets generated):
- Training runs (`experiments/**/runs/`)
- Checkpoints (`.pt` files, 500MB-1GB each)
- Plots and visualizations (PNG, PDF, SVG)
- CSV logs and metrics
- Videos of gameplay

**Outputs** (what to keep/archive/delete):
- Essential artifacts for reproducibility
- Thesis-ready figures and tables
- Clean repository for collaboration

<br><br>

## Retention Categories

### Keep Forever (Git-Tracked)

These files are essential for reproducibility and should be version-controlled:

**Source code**:
- `src/**/*.py` - All implementation code
- `tests/**/*.py` - Test suite
- `scripts/**/*.py` - Analysis and plotting scripts
- `experiments/**/configs/*.yaml` - Training configurations

**Documentation**:
- `docs/**/*.md` - All documentation
- `README.md` - Project overview
- `requirements.txt` - Python dependencies

**Essential configs**:
- `.gitignore` - File exclusion rules
- `.gitattributes` - Git LFS configuration (if used)
- `pyproject.toml` / `setup.py` - Package metadata

**Rationale**: Code and docs must be tracked for reproducibility and collaboration.

<br><br>

### Archive Locally (Gitignored, Keep Selectively)

These artifacts should be excluded from git but retained selectively:

**Final checkpoints**:
- `experiments/**/runs/*/checkpoints/best_model.pt` - Best checkpoint per run
- Keep: Final production runs only (delete exploratory/test runs)
- Size: ~500MB-1GB per checkpoint
- Location: Local disk, external drive, or cloud storage (W&B artifacts)

**Evaluation CSVs**:
- `experiments/**/runs/*/csv/evaluations.csv` - Periodic eval results
- Keep: All production runs (needed for thesis)
- Size: <1MB per run
- Location: Local disk (small enough to keep all)

**Episode/step CSVs**:
- `experiments/**/runs/*/csv/episodes.csv` - Per-episode metrics
- `experiments/**/runs/*/csv/training_steps.csv` - Per-step metrics
- Keep: Final production runs only
- Delete: Exploratory runs, test runs
- Size: 10-100MB per run
- Rationale: Can regenerate plots from evaluations.csv if needed

**Thesis-ready plots**:
- `output/plots/**/*.png` - Generated visualizations
- Keep: Final versions for thesis
- Delete: Intermediate/test plots
- Size: 1-5MB per plot
- Regenerate: Can recreate from CSVs using `scripts/plot_results.py`

**Rationale**: Selective retention balances reproducibility with disk space.

<br><br>

### Delete Immediately (Gitignored, Disposable)

These can be safely deleted and regenerated:

**Intermediate checkpoints**:
- `experiments/**/runs/*/checkpoints/checkpoint_*.pt` (not best_model.pt)
- Delete: After training completes
- Rationale: Only need final/best checkpoint for inference

**Test run artifacts**:
- `experiments/**/runs/*/` - Smoke tests, validation runs, debugging runs
- Delete: After confirming functionality
- Keep only: Production baseline runs

**TensorBoard logs**:
- `experiments/**/runs/*/tensorboard/` - Event files
- Delete: After training completes (metrics already in CSVs)
- Rationale: W&B provides online alternative, CSVs have same data

**Videos**:
- `experiments/**/runs/*/videos/*.mp4` - Gameplay recordings
- Keep: Best episodes for presentation/thesis
- Delete: All others (can regenerate from checkpoints if needed)
- Size: 10-50MB per video

**W&B local cache**:
- `wandb/` - Local W&B sync cache
- Delete: Safe to delete anytime (data is on W&B cloud)
- Rationale: Regenerated automatically on next W&B run

**Python caches**:
- `__pycache__/`, `*.pyc`, `.pytest_cache/`
- Delete: Always safe to delete
- Rationale: Regenerated automatically

**Rationale**: Reduce disk usage, keep repo clean.

<br><br>

## Retention Rules by Use Case

### Production Baseline Runs (Keep)

For thesis-critical experiments:

**Keep**:
- Best checkpoint
- All CSVs (episodes, training_steps, evaluations)
- Final plots
- Config + metadata (meta.json, system_info.txt)
- Best evaluation videos (1-3 per game)

**Delete**:
- Intermediate checkpoints
- TensorBoard logs
- All other videos

**Disk per run**: ~1-2 GB (checkpoint 500MB-1GB + CSVs 100MB + plots/config 10MB)

### Exploratory Runs (Delete)

For testing hyperparameters, debugging, validation:

**Keep**:
- Config file (for record-keeping)
- Final eval CSV (if promising results)

**Delete**:
- All checkpoints
- All other CSVs
- All plots
- All videos

**Action**: Delete entire run directory after extracting key findings to notes.

### Ablation Studies (Archive)

For systematic experiments:

**Keep**:
- Best checkpoint (if results are significant)
- Evaluation CSV (for comparison tables)
- Plots showing comparison to baseline

**Delete**:
- Intermediate checkpoints
- Episode/step CSVs (unless needed for detailed analysis)

**Disk per study**: ~500MB-1GB per variant

<br><br>

## Git Configuration

### .gitignore Strategy

**Current configuration** (verified in `.gitignore`):

```gitignore
# Experiment artifacts (never commit)
experiments/**/runs/
experiments/**/checkpoints/
experiments/**/artifacts/
output/

# Media files (regenerable or very large)
*.mp4
*.gif
*.png
*.jpg
*.jpeg

# Model checkpoints (very large)
*.pth
*.pt

# Logs
*.log
logs/

# W&B cache
wandb/

```

**Rationale**:
- **Never commit large binaries** (checkpoints, videos, plots)
- **Never commit generated outputs** (output/, runs/, wandb/)
- **Personal workspace files** belong in `.git/info/exclude` (local-only)

### Exceptions (Should Be Tracked)

If you need to commit specific artifacts:

**Option 1: Git LFS (not currently used)**
```bash
# For very important checkpoints (e.g., published baseline)
git lfs track "experiments/baselines/*.pt"
```

**Option 2: External storage**
- Upload to W&B artifacts (recommended for checkpoints)
- Use Google Drive / Dropbox for large plots
- Link URLs in documentation

**Option 3: Exception pattern**
```gitignore
# In .gitignore
output/*

# To track specific file
!output/final_thesis_plots/
```

**Recommendation**: Use W&B artifacts for checkpoints, track only code/docs in git.

<br><br>

## Cleanup Procedures

### Routine Cleanup (Weekly)

```bash
# Remove intermediate checkpoints
find experiments/ -name "checkpoint_*.pt" ! -name "best_model.pt" -delete

# Remove TensorBoard logs
find experiments/ -name "tensorboard" -type d -exec rm -rf {} +

# Remove W&B local cache
rm -rf wandb/

# Remove test run directories (adjust pattern as needed)
rm -rf experiments/dqn_atari/runs/*_test_*
rm -rf experiments/dqn_atari/runs/*_debug_*
```

**Est. space saved**: 50-90% of total experiment disk usage

### Deep Cleanup (After Thesis Submission)

```bash
# Archive production runs to external storage
tar -czf thesis_runs_archive.tar.gz \
    experiments/dqn_atari/runs/pong_42_* \
    experiments/dqn_atari/runs/pong_123_* \
    experiments/dqn_atari/runs/pong_456_*

# Verify archive
tar -tzf thesis_runs_archive.tar.gz | head

# Delete local copies (keep archive safe!)
rm -rf experiments/dqn_atari/runs/

# Keep only: code, docs, final plots, summary tables
```

**Est. space saved**: 95%+ (keep only ~1 GB instead of ~20+ GB)

### Emergency Cleanup (Disk Full)

```bash
# Immediate: Delete largest files
du -sh experiments/dqn_atari/runs/* | sort -rh | head -10

# Remove all checkpoints except best
find experiments/ -name "*.pt" ! -name "best_model.pt" -delete

# Remove all videos
find experiments/ -name "*.mp4" -delete

# Remove all plots (regenerable)
rm -rf output/plots/
```

**Est. space saved**: 70-80% immediately

<br><br>

## Backup Strategy

### Critical Data (Must Backup)

**Source code and docs**:
- Git remote (GitHub, GitLab, etc.) - automatic via git push
- Frequency: Every commit
- Retention: Forever

**Final production checkpoints**:
- W&B artifacts - upload via training script
- External drive - manual backup after run completes
- Frequency: After each production run
- Retention: Keep until thesis published, then archive

**Thesis-ready outputs**:
- Results CSVs, final plots
- Copy to thesis repo / writing folder
- Frequency: When generating thesis content
- Retention: Keep with thesis indefinitely

### Non-Critical Data (Optional Backup)

**Exploratory runs**:
- No backup needed (disposable)

**Intermediate artifacts**:
- No backup needed (regenerable from code + checkpoints)

<br><br>

## Disk Space Planning

### Per-Run Estimates

| Run Type | Checkpoints | CSVs | Videos | Plots | Total |
|----------|-------------|------|--------|-------|-------|
| 10M Pong | 500 MB | 50 MB | 200 MB | 10 MB | ~760 MB |
| 50M Breakout | 800 MB | 200 MB | 500 MB | 10 MB | ~1.5 GB |
| Ablation variant | 500 MB | 50 MB | 0 MB | 5 MB | ~555 MB |

### Repository Growth

**Minimal thesis** (Pong 3-seed only):
- 3 runs × 760 MB = ~2.3 GB

**Strong thesis** (Pong + Breakout + Beam Rider, 3 seeds each):
- 9 runs × 1.2 GB avg = ~11 GB

**With ablations** (3 games × 3 seeds + 5 ablations):
- Base: ~11 GB
- Ablations: 5 × 555 MB = ~2.8 GB
- **Total**: ~14 GB

**Recommendation**: Provision 20-30 GB disk space for comfortable development.

<br><br>

## Next Steps

**To apply retention rules**:
1. Review current disk usage: `du -sh experiments/ output/`
2. Identify production vs exploratory runs
3. Run cleanup scripts for intermediate artifacts
4. Archive production runs to external storage

**To maintain clean repo**:
- Run routine cleanup weekly during active development
- Delete test runs immediately after validation
- Upload final checkpoints to W&B before local deletion

**For thesis submission**:
- Create final archive of all production runs
- Copy thesis-ready plots/tables to thesis repo
- Document W&B run URLs in thesis appendix
- Keep archive safe (external drive + cloud backup)

<br><br>

## Related Documents

- [Reporting Pipeline](reporting-pipeline.md) - Result generation workflow
- [Logging Pipeline](logging-pipeline.md) - Artifact generation during training
- [Applied Research Quickstart](../guides/applied-research-quickstart.md) - Phase 4 (document findings)
- [Checkpointing](checkpointing.md) - Checkpoint format and resumption

<br><br>

**Last Updated**: 2025-12-23
