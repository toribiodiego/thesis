# Maintenance Checklists

Standard operating procedures for maintaining the DQN reproduction codebase.

## Release-Ready Checklist

Before publishing results or sharing code:

- [ ] **Environment Setup**
  - [ ] Rebuild virtual environment: `rm -rf .venv && python -m venv .venv`
  - [ ] Install dependencies: `pip install -r requirements.txt`
  - [ ] Verify PyTorch version matches expected: `python -c "import torch; print(torch.__version__)"`

- [ ] **Code Quality**
  - [ ] Run linters: `ruff check src/ tests/`
  - [ ] Run formatters: `black src/ tests/ && isort src/ tests/`
  - [ ] All tests pass: `pytest tests/ -x`
  - [ ] No uncommitted changes: `git status`

- [ ] **Smoke Test**
  - [ ] Short training run completes:
    ```bash
    python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
        --seed 42 --set training.total_frames=100000
    ```
  - [ ] Verify artifacts created: `python scripts/verify_artifacts.py <run_dir>`
  - [ ] Review summary: `python scripts/summarize_run.py <run_dir>`

- [ ] **W&B Artifacts** (if enabled)
  - [ ] W&B API key configured: `echo $WANDB_API_KEY`
  - [ ] Artifacts upload successfully
  - [ ] Verify runs appear in W&B dashboard
  - [ ] Check artifact file integrity

- [ ] **Documentation**
  - [ ] Update [TODO](../../TODO) with completed tasks
  - [ ] Review `CLAUDE.md` for accuracy
  - [ ] Ensure design docs reflect current implementation
  - [ ] Troubleshooting guide is current

- [ ] **Version Control**
  - [ ] All changes committed with descriptive messages
  - [ ] No sensitive data in commits (API keys, credentials)
  - [ ] Branch is up-to-date with main
  - [ ] CI pipeline passes

---

## Pre-Training Checklist

Before starting a full training run:

- [ ] **Configuration**
  - [ ] Verify config file exists and is valid YAML
  - [ ] Check environment ID is correct (e.g., `PongNoFrameskip-v4`)
  - [ ] Confirm seed is set for reproducibility
  - [ ] Review hyperparameters match paper defaults

- [ ] **System Resources**
  - [ ] Sufficient disk space for replay buffer (~8GB for 1M capacity)
  - [ ] GPU available (if CUDA enabled)
  - [ ] Memory available for training
  - [ ] Stable internet (if W&B enabled)

- [ ] **Logging Setup**
  - [ ] TensorBoard/W&B enabled as needed
  - [ ] CSV logging enabled for offline analysis
  - [ ] Checkpoint save interval configured
  - [ ] Video recording settings appropriate

- [ ] **Monitoring**
  - [ ] TensorBoard accessible: `tensorboard --logdir experiments/dqn_atari/runs/`
  - [ ] W&B dashboard accessible (if enabled)
  - [ ] Process monitoring in place (htop, nvidia-smi)

---

## Post-Training Checklist

After completing a training run:

- [ ] **Artifact Verification**
  - [ ] Run: `python scripts/verify_artifacts.py <run_dir>`
  - [ ] All required files present
  - [ ] No empty or corrupted files

- [ ] **Results Review**
  - [ ] Run: `python scripts/summarize_run.py <run_dir>`
  - [ ] Check final evaluation return
  - [ ] Compare to paper baselines
  - [ ] Review training curves for anomalies

- [ ] **Plots Generation**
  - [ ] Generate learning curves: `python scripts/plot_results.py <run_dir>`
  - [ ] Compare multiple runs (if applicable)
  - [ ] Export high-resolution figures

- [ ] **Best Model Validation**
  - [ ] Best checkpoint saved
  - [ ] Can load model without errors
  - [ ] Evaluation matches logged metrics

- [ ] **Data Backup**
  - [ ] Checkpoints archived (if needed)
  - [ ] Videos backed up
  - [ ] CSV logs preserved
  - [ ] W&B sync complete

- [ ] **Documentation**
  - [ ] Record run parameters and results
  - [ ] Note any issues or anomalies
  - [ ] Update experiment notes

---

## Weekly Maintenance

Regular tasks to keep the repository healthy:

- [ ] **Dependency Updates**
  - [ ] Check for security vulnerabilities: `pip audit`
  - [ ] Review outdated packages: `pip list --outdated`
  - [ ] Test with updated dependencies

- [ ] **Disk Cleanup**
  - [ ] Remove old experiment runs (keep important ones)
  - [ ] Clear TensorBoard caches
  - [ ] Clean up temporary files

- [ ] **Code Hygiene**
  - [ ] Remove dead code
  - [ ] Update deprecated APIs
  - [ ] Refactor complex functions

- [ ] **Test Coverage**
  - [ ] Review test coverage: `pytest --cov=src tests/`
  - [ ] Add tests for new features
  - [ ] Fix flaky tests

---

## Documentation Review

Quarterly documentation maintenance (or before major milestones):

- [ ] **Navigation and Structure**
  - [ ] Verify `docs/README.md` navigation is current
  - [ ] Check all new docs are linked from appropriate sections
  - [ ] Confirm folder structure matches documented layout
  - [ ] Review `README.md` docs links point to correct locations

- [ ] **Link Integrity**
  - [ ] Run link check: `grep -r "\[.*\](.*\.md)" docs/ | grep -v "^Binary" > /tmp/doc_links.txt`
  - [ ] Manually verify external links (arXiv, GitHub, W&B)
  - [ ] Check internal cross-references resolve correctly
  - [ ] Fix any broken links or outdated paths

- [ ] **Content Accuracy**
  - [ ] Reference docs match current implementation
  - [ ] Config examples match actual YAML files
  - [ ] Command examples work with current CLI
  - [ ] Hyperparameters match current defaults
  - [ ] File paths reflect current repository structure

- [ ] **Status and Metadata**
  - [ ] Update "Last Updated" dates for modified docs
  - [ ] Check status callouts (DRAFT, ACTIVE, REFERENCE, DEPRECATED)
  - [ ] Archive obsolete docs to `docs/archive/` if needed
  - [ ] Update `docs/changelog.md` with major doc changes

- [ ] **Conventions Compliance**
  - [ ] File names use kebab-case
  - [ ] H1 titles match filenames
  - [ ] Reports use `report-` prefix, plans use `plan-` prefix
  - [ ] New docs follow minimal outline template (Purpose, Inputs/Outputs, Next Steps)
  - [ ] Docs use proper heading hierarchy (H1 → H2 → H3, max 3 levels)

- [ ] **Cross-Reference Audit**
  - [ ] Check `docs/guides/architecture.md` links to component specs
  - [ ] Verify `docs/plans/plan-report-outline.md` script paths are valid
  - [ ] Confirm `docs/thesis/README.md` artifact paths exist
  - [ ] Review `docs/reference/reporting-pipeline.md` workflow matches current scripts

---

## Troubleshooting Common Issues

Quick reference for frequent problems:

### Training Crashes
1. Check GPU memory: `nvidia-smi`
2. Reduce batch size if OOM
3. Verify environment installs correctly
4. Check for NaN/Inf in loss

### Slow Performance
1. Profile with `py-spy`: `py-spy record -o profile.svg -- python train_dqn.py ...`
2. Check CPU utilization
3. Verify GPU is being used
4. Review frame skip settings

### W&B Issues
1. Verify API key: `wandb login`
2. Check internet connectivity
3. Review W&B logs in run directory
4. Try offline mode: `WANDB_MODE=offline`

### Reproducibility Issues
1. Set all seeds consistently
2. Use deterministic mode if needed
3. Check for non-deterministic operations
4. Verify exact package versions

---

## Emergency Procedures

When things go wrong:

### Interrupted Training
1. Locate last checkpoint: `ls -la <run_dir>/checkpoints/`
2. Resume training: `python train_dqn.py --resume <checkpoint_path>`
3. Verify state restoration
4. Monitor for issues

### Corrupted Artifacts
1. Use backup if available
2. Check partial data integrity
3. Consider rerunning from checkpoint
4. Document what was lost

### CI Pipeline Failures
1. Check logs in GitHub Actions
2. Run linters locally: `ruff check src/ tests/`
3. Run tests: `pytest tests/ -x`
4. Fix issues before pushing

---

## Scripts Quick Reference

```bash
# Summarize a run
python scripts/summarize_run.py <run_dir>

# Verify artifacts
python scripts/verify_artifacts.py <run_dir>

# Generate plots
python scripts/plot_results.py <run_dir>

# Run smoke test
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
    --seed 42 --set training.total_frames=100000

# Full test suite
pytest tests/ -x --tb=short

# Linting
ruff check src/ tests/

# Formatting
black src/ tests/ && isort src/ tests/

# Check doc links
grep -r "\[.*\](.*\.md)" docs/ | grep -v "^Binary" > /tmp/doc_links.txt
```
