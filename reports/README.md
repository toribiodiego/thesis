# Reports

Working documents for thesis results and analysis. These are
intermediate artifacts -- drafts, data summaries, and reference
records -- not the final thesis (which lives in `writing/`).

Compile any `.tex` file with `tectonic <file>.tex`.


## Living documents (updated as experiments progress)

- `eval.tex` -- Figure and results catalog. Maps every figure and
  table to its purpose, data source, thesis section, and current
  status. Includes regeneration commands. This is the master
  reference for what we will produce.

- `working-results.tex` -- Current results draft with figures and
  tables from completed runs. Updated after each batch of runs.

- `outline.tex` -- Thesis chapter outline and section structure.


## Reference documents (completed, stable)

- `baseline-validation.tex` -- Rainbow hyperparameter audit.
  Documents the 7 mismatches found vs SPR paper Table 3, the
  fixes applied, and before/after validation scores on all 6
  games. Config is now locked.

- `eval-pipeline.tex` -- Decoupled evaluation pipeline. Documents
  the architecture (train on A100, assess on T4), config changes,
  7 script fixes, and Colab validation results.


## Planned (stubs, not yet written)

- `ablation-analysis.tex` -- Ablation breakdowns and interaction
  effects from the 2x2 factorial. Feeds into Results 4.2-4.4.

- `interpretability.tex` -- Linear probe results, latent space
  visualization, and transition model analysis. Feeds into Results
  4.3 and Discussion 5.2-5.3. Requires Tasks 47-52.
