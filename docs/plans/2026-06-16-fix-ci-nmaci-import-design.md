# Fix CI nmaci Import Error

## Problem

The CI `setup` job fails with:

```
ModuleNotFoundError: No module named 'nmaci'
```

This happens because `ci/select_notebooks.py` does `from nmaci.select_notebooks import main`, but the current `setup-ci-tools` action only downloads nmaci scripts to the `ci/` directory via wget and installs from `ci/requirements.txt`. It never actually installs the `nmaci` Python package, so imports fail.

## Root Cause

The current `setup-ci-tools/action.yml` uses a tarball-download-into-ci/ pattern:

1. Downloads nmaci source archive from GitHub
2. Extracts `scripts/` to `ci/` and caches that directory
3. Installs dependencies from `ci/requirements.txt`

This installs CLI tools into `ci/` but does not install the `nmaci` Python package itself.

## Design

### 1. Simplify `.github/actions/setup-ci-tools/action.yml`

Adopt the simpler approach used by `course-content`:

**Removed steps:**
- `Get nmaci latest commit` — SHA detection via `git ls-remote`
- `Cache nmaci tools` — tarball download + `ci/` directory cache
- `Download nmaci tools` — wget + tar extraction
- `Install nmaci dependencies` — `ci/requirements.txt` pip install
- `Ignore ci directory` — `.gitignore` modification

**Retained steps:**
- `Detect nmaci branch` — parses commit message for `nmaci:branch-name` override
- `nmaci-branch` output — exposes branch name to workflow
- `Stub ipywidgets` — copies `stub_widgets.py` to IPython startup directory

**Added step:**
- `Install nmaci package` — `pip install "git+https://github.com/neuromatch/nmaci@$BRANCH"`

### 2. Update GitHub Actions versions

Update all deprecated Node.js 20 actions to their latest Node 24-compatible versions:

| Action | Current | New |
|--------|---------|-----|
| `actions/checkout` | v4 | v5 |
| `actions/setup-python` | v5 | v6 |
| `actions/cache` | v4 | v5 |
| `actions/upload-artifact` | v4 | v5 |
| `actions/download-artifact` | v4 | v5 |
| `tj-actions/changed-files` | v47 | latest |
| `ad-m/github-push-action` | v0.6.0 | latest |
| `tlylt/install-graphviz` | v1 | latest |

### Files Modified

- `.github/actions/setup-ci-tools/action.yml` — major rewrite
- `.github/workflows/notebook-pr.yaml` — action version bumps
- `.github/actions/setup-python-env/action.yml` — action version bump
- `.github/actions/setup-rendering-deps/action.yml` — action version bumps

### Files Removed

None. The `ci/` directory and its scripts remain as installed by the `nmaci` pip package.

### Files Unchanged

- `.github/actions/setup-ci-tools/stub_widgets.py` — still used by the stub step
