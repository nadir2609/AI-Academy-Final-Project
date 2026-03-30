# Math4AI ACM-Style Report Package

This is a professional ACM-style LaTeX report package tailored for the Math4AI capstone project.

## Files
- `main.tex` — main report template with integrated guidance comments
- `macros.tex` — packages and project-specific macros
- `references.bib` — starter bibliography matching the recommended references from the capstone handout
- `figures/` — put your figure files here

## Recommended workflow
1. Upload the whole folder to Overleaf.
2. Replace author names, emails, and placeholder text.
3. Keep the full story in the main paper.
4. Use the appendix only for supporting extras.
5. Delete the advanced-track subsection you do not use.
6. Make sure every figure/table is discussed in the text.

## Suggested main-paper structure
- Introduction / Framing
- Background and Mathematical Setup
- Methods
- Implementation Sanity Checks
- Experiments
- Advanced Track
- Discussion
- Limitations

## Suggested appendix contents
- extra plots
- extended derivations
- supplementary tables
- seed-by-seed breakdowns
- additional implementation details

## Important note
This package uses the `acmart` class. Overleaf normally has it available by default. If compiling locally, install the ACM `acmart` package in your LaTeX distribution.
