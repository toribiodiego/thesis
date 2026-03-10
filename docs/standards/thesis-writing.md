# Thesis Writing Standards

Conventions for all LaTeX content under `writing/` and `reports/`.
For project documentation standards, see [Documentation Standards](./documentation.md).

<br><br>

## Table Captions

Maximum two sentences. The caption must allow the reader to interpret
the table without reading the surrounding text.

- **Sentence 1:** What the table shows (the data, the grouping, the
  comparison).
- **Sentence 2:** How to read it (what varies across rows/columns,
  what the key distinction is).

```text
Good:  "Atari-100K benchmark games grouped by how frequently the
        agent receives non-zero reward. Dense games yield reward on
        most steps; sparse games have long stretches of zero reward."

Bad:   "Atari-100K games by reward density."
       (cannot interpret without reading the paragraph above)

Bad:   "The 26 Atari-100K benchmark games grouped by reward density,
        which determines how much learning signal a vanilla DQN agent
        receives within the 100K step budget. Our six selected games
        are drawn from all three categories. Categorization based on
        published scores in Schwarzer et al. (2021), Table 4."
       (too long -- move context to body text)
```

<br><br>

## Figure Captions

Same two-sentence rule as tables. If the figure has multiple panels,
a third sentence identifying the panels is acceptable.

<br><br>

## Caption Placement

Place captions **below** tables and figures, left-aligned within a
centered float. Use `\raggedright` inside the float so the caption
text is left-aligned while the table/figure itself stays centered.
Add vertical space (`\vspace{1.2em}`) between the table body and the
caption.

```latex
\begin{table}[h]
\centering
% ... table content ...
\vspace{1.2em}
{\raggedright\small \textbf{Table N.} Caption text here.\par}
\end{table}
```

<br><br>

## Technical Terms

Define RL and ML terminology inline on first use. The thesis advisor
is not an RL specialist -- write so that a machine learning
generalist can follow without consulting external references.

Keep definitions brief and parenthetical or set off with a dash:

```text
Good:  "The agent explores by choosing random actions with
        probability epsilon, which decays from 1.0 (fully random)
        to 0.1 over the first half of training."

Bad:   "Epsilon decays from 1.0 to 0.1 over 200K frames."
       (assumes reader knows what epsilon means in this context)
```

<br><br>

## Paragraph Density

Each paragraph should make one point. When a section covers multiple
concepts (e.g., benchmark description, method motivation, experiment
scope), split them into separate paragraphs with `\medskip` between
them rather than packing everything into a wall of text.

<br><br>

## Voice

Use first-person plural ("we") for the thesis narrative, matching
standard academic convention. This differs from `docs/` which uses
second-person ("you").
