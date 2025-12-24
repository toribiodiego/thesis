# Output Directory

This directory contains generated analysis outputs (plots, summaries, and statistical analyses).

**Note**: This directory is gitignored. All contents are generated from training runs and can be recreated using analysis scripts.

## Subdirectories

- `plots/` - Visualizations (learning curves, eval trends, Q-value plots)
- `summary/` - Aggregated metrics tables and multi-seed comparisons
- `analysis/` - Statistical analyses and comparison reports

## Generating Outputs

```bash
# Plot results from a training run
python scripts/plot_results.py --episodes experiments/dqn_atari/runs/pong_42_*/csv/episodes.csv --output output/plots/pong_42

# Export summary table
python scripts/export_results_table.py --runs-dir experiments/dqn_atari/runs --output output/summary/metrics.csv

# Run statistical analysis
python scripts/analyze_results.py --metrics output/summary/metrics.csv --output output/analysis/results.txt
```

## Retention Policy

- All files in `output/` are disposable and can be safely deleted
- Regenerate outputs from source data in `experiments/**/runs/` as needed
- For archival, use W&B artifacts or manually backup specific analyses

## See Also

- [Reporting Pipeline](../docs/reference/reporting-pipeline.md) - How to generate outputs
- [Archive Plan](../docs/reference/archive-plan.md) - Retention policies
- [Thesis Artifact Index](../docs/thesis/README.md) - Thesis-ready outputs
