# Release checklist

See **PUBLISHING_GUIDE.md** for the full step-by-step with exact commands.

Automated locally, verified 2026-07-18 against a clean Python 3.12 environment:
version synced (0.1.0); 16/16 tests pass; `ruff check .` clean; `python -m build`
produces wheel + sdist and the wheel-from-sdist build succeeds; both artifacts pass
`twine check`; a clean-env wheel install imports `choir` and runs `load_demo()`
(6000 rows). Two packaging fixes were applied today: `src/choir/datasets/demo_fars.csv`
is now git-tracked (root `.gitignore` `*.csv` PII guard was excluding it, so clones
and the sdist shipped without it), and `pyproject.toml` gained a `[tool.hatch.build]
artifacts` entry so both sdist and wheel carry the demo table.

Steps that require your accounts/credentials (I cannot do these for you):

1. **GitHub.** Target repo is `pozapas/choircert` (already created, empty). See
   PUBLISHING_GUIDE.md Step 1 for the clean standalone-export-and-push procedure.
   CI in `.github/workflows/ci.yml` runs tests on 3.10-3.12, lint, and a wheel build.
2. **PyPI.** Name `choircert` is free (import stays `choir`). `choir` is taken by an
   abandoned 2013 package. To publish: `uv build` then `uvx twine upload dist/*` with a
   PyPI API token. Do a TestPyPI dry run first.
3. **Zenodo.** Enable the GitHub-Zenodo integration on the repo, then cut a GitHub
   release; Zenodo mints the DOI automatically from `.zenodo.json`. Paste the DOI badge
   into README and the version DOI into CITATION.cff.
4. **Docs.** `mkdocs gh-deploy` publishes the site (mkdocs.yml + docs/) to GitHub Pages.

Nothing above changes the code; it is account setup. Ask me to prepare the exact
commands for any step.
