# Git Commit Message Guide

Use the prefixes below to keep history and changelogs consistent.

| Prefix | Meaning | Example | Notes |
|--------|---------|---------|-------|
| `feat:` | Adds a new feature or behavior | `feat: add Docling ingestion backend` | Use when introducing user-visible functionality or components. |
| `fix:` | Bug fix or error correction | `fix: handle missing Docling parser config` | Describe what was fixed; helps keep changelogs readable. |
| `chore:` | Non-functional changes (setup, config, maintenance) | `chore: bump Gymnasium pin in requirements` | No logic change; environment or repo hygiene only. |
| `refactor:` | Internal restructuring without behavior change | `refactor: simplify ingestion pipeline loop` | Improves readability, structure, or performance without altering outputs. |
| `docs:` | Documentation-only updates | `docs: update README with ingestion infrastructure` | Covers README files, design docs, or docstrings. |
| `test:` | Adds or modifies tests | `test: add ingestion pipeline unit tests` | Includes unit, integration, or regression tests. |
| `perf:` | Performance optimization | `perf: cache Docling parser output` | Reserve for measurable performance improvements. |
| `build:` | Build / packaging / CI changes | `build: update Dockerfile to include tesseract` | Applies to Dockerfiles, CI pipelines, packaging scripts, etc. |
| `wip:` | Work in progress (not ready for review) | `wip: prototype Python serialization handler` | Use for exploratory commits; squash or reword before merging. |
| `style:` | Formatting or stylistic changes only | `style: reformat docling_ingest.py for PEP8` | No logic changes—pure lint or formatting fixes. |

Tips:
- Keep subject lines in imperative mood and ≤72 characters when possible.
- Follow with an optional body describing the “why” when the change needs context.
- Reference issue IDs or roadmap items in the body, not the prefix.
