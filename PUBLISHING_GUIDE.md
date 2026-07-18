# Publishing guide: `choircert` (import name `choir`)

This is a step-by-step guide to publish the package to PyPI, push the source to
GitHub, and (optionally) mint a Zenodo DOI and deploy the docs. Everything that
does **not** need your personal accounts has already been done and verified; the
steps below are the ones that require your credentials.

Package facts:

- Distribution name on PyPI: **`choircert`** (the name `choir` is taken by an
  abandoned 2013 package). Import name stays **`choir`**.
- Version: **0.1.0**, license MIT, core dependency `numpy` only.
- Repo URL declared in `pyproject.toml` and `CITATION.cff`:
  `https://github.com/pozapas/choircert` (already updated to your account).

---

## 0. What is already done and verified (no action needed)

Run against a clean Python 3.12 environment on 2026-07-18:

- 16/16 tests pass (`pytest`).
- `ruff check .` is clean (fixed 3 lint issues in `benchmarks/vs_mapie_crepes.py`).
- `python -m build` produces both a wheel and an sdist, and building the wheel
  from the sdist succeeds.
- Both artifacts pass `twine check`.
- A clean-environment install of the wheel imports `choir`, runs `load_demo()`
  (6000 rows, 7 features), and reports `choir.__version__ == "0.1.0"`.
- The bundled demo table `src/choir/datasets/demo_fars.csv` is now **git-tracked**
  (a targeted exception was added to the root `.gitignore`, which otherwise
  excludes all `*.csv` as a PII guard). Before this, the file was invisible to git,
  so fresh clones and the sdist would have shipped without the demo data.

The built artifacts are in `choir/dist/`:

```
choircert-0.1.0-py3-none-any.whl
choircert-0.1.0.tar.gz
```

> If you edit any code after today, rebuild before uploading (Step 2).

---

## 1. GitHub  (target: `https://github.com/pozapas/choircert`, already created, empty)

Your `choir/` directory currently lives **inside** the private `WM` research
monorepo (which holds PII crash data). The goal is to publish **only** `choir/`
as its own public repo, with none of the parent repo's history or data. The clean
way to do that is to export the tracked `choir/` files into a fresh standalone
repo and push that. This avoids a nested git repo and guarantees no WM history or
PII can leak into the public repo.

### 1a. Commit the package changes into your WM repo first

So the export below picks them up. From the WM root (`.../WorldModels/WM`):

```bash
git add choir/ .gitignore
git commit -m "choircert 0.1.0: track demo dataset, fix packaging, lint clean, repo URLs"
```

`choir/src/choir/datasets/demo_fars.csv` is a public FARS-schema demo (columns
`age, speed_limit, restraint, light, vehicle_age, rural, body_class, y`, no PII)
and is safe to commit. The blanket `*.csv` guard on the real CRIS data is untouched.

### 1b. Export a clean standalone copy of `choir/`

`git archive` emits only git-tracked files, so envs, caches, and `dist/`
artifacts are excluded automatically. From the WM root:

```bash
mkdir -p ../choircert            # a fresh directory OUTSIDE the WM tree
git archive HEAD:choir | tar -x -C ../choircert
cd ../choircert
ls                               # sanity check: src/, tests/, pyproject.toml, README.md, .gitignore, demo csv
```

Confirm the demo data made it and no junk did:

```bash
test -f src/choir/datasets/demo_fars.csv && echo "demo data present"
ls -a | grep -E '\.venv|__pycache__|dist' || echo "clean (no envs/build artifacts)"
```

### 1c. Initialize the standalone repo and push

```bash
# still in ../choircert
git init
git add -A
git commit -m "choircert 0.1.0 initial public release"
git branch -M main
git remote add origin https://github.com/pozapas/choircert.git
git push -u origin main
```

If the push is rejected because the empty GitHub repo was created with a README
or license, run `git pull --rebase origin main` once, then push again.

> Alternative (not recommended): `git init` directly inside `choir/`. It works but
> creates a git repo nested inside the WM repo, which is confusing to maintain and
> risks mixing the two histories. The export approach above is cleaner.

CI (`.github/workflows/ci.yml`) will run tests on Python 3.10-3.12, lint, and a
wheel build on push.

---

## 2. Build the artifacts with `uv` (recommended)

Use `uv`, which is already on your PATH and picks its own interpreter, so there is
no fragile hardcoded Python path. Run from inside the repo (`.../choircert`):

```bash
uv build                 # writes dist/choircert-0.1.0.tar.gz and .whl
uvx twine check dist/*   # both should print PASSED
```

> Do NOT paste a `C:/Users/.../python.exe` path into MINGW/Git Bash; a trailing
> carriage return from the paste makes bash report "No such file or directory"
> even when the file exists. `uv` sidesteps this entirely.

Verified 2026-07-18: `uv build` produces both artifacts and `uvx twine check`
passes for both; the demo table is present in the wheel.

---

## 3. PyPI

You need a PyPI account and an API token. Create the token at
<https://pypi.org/manage/account/token/> (use an account-wide token for the first
upload; scope it to the `choircert` project afterward).

### 3a. Dry run on TestPyPI first (strongly recommended)

Create a separate token at <https://test.pypi.org/manage/account/token/>, then
publish the built artifacts with `uv publish`:

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-XXXXXXXX
```

Verify a clean install from TestPyPI (a throwaway environment via uv):

```bash
uv run --no-project --with "choircert" \
  --index "https://test.pypi.org/simple/" \
  --index-strategy unsafe-best-match \
  python -c "from choir.datasets import load_demo; print(len(load_demo()[0]), 'rows')"
```

The extra index strategy lets `numpy` resolve from real PyPI while `choircert`
comes from TestPyPI. It should print `6000 rows`.

### 3b. Real upload

```bash
uv publish --token pypi-XXXXXXXX
```

(`uv publish` defaults to the real PyPI.) After this, `pip install choircert`
works for everyone and the README install line is true.

> Prefer `twine` instead? `uvx twine upload --repository testpypi dist/*` then
> `uvx twine upload dist/*`, entering `__token__` as the username and the token
> (with its `pypi-` prefix) as the password.

> Version numbers on PyPI are immutable: you cannot re-upload `0.1.0` after it is
> published. If you find a problem, bump to `0.1.1` in `pyproject.toml`, rebuild,
> and upload again.

---

## 4. Zenodo DOI (optional, for citation)

1. Log in to <https://zenodo.org> with GitHub and enable the Zenodo-GitHub
   integration for the `pozapas/choircert` repo (Zenodo "GitHub" settings page,
   flip the repo toggle on).
2. On GitHub, cut a release (tag `v0.1.0`). Zenodo mints a DOI automatically from
   `.zenodo.json`.
3. Paste the DOI badge into `README.md` and the version DOI into `CITATION.cff`,
   then commit and push.

---

## 5. Docs site (optional)

```bash
# from choir/, with mkdocs installed (pip install mkdocs mkdocs-material)
mkdocs gh-deploy
```

This publishes `mkdocs.yml` + `docs/` to GitHub Pages.

---

## Order of operations (recommended)

1. Commit and push to GitHub (Step 1), confirm CI is green.
2. TestPyPI dry run and verify install (Step 3a).
3. Real PyPI upload (Step 3b).
4. Cut the GitHub release; let Zenodo mint the DOI (Step 4).
5. Add the DOI badge, commit, push, then `mkdocs gh-deploy` (Step 5).

Nothing above changes the package code. If you want, I can prepare the exact
`git`, `twine`, and `mkdocs` command blocks pre-filled once you confirm the final
GitHub owner/repo name.
