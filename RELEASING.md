# Releasing assumpcheck

This project publishes to **TestPyPI first**, then **PyPI**, using **GitHub Actions Trusted Publishing**.

## One-time setup

Create and verify accounts on:

- [PyPI](https://pypi.org/)
- [TestPyPI](https://test.pypi.org/)

Then configure trusted publishers for this repository.

Repository settings to use:

- Repository owner: `Josiah-DeValois`
- Repository name: `assumpcheck`
- Workflow file: `.github/workflows/publish.yml`

### TestPyPI trusted publisher

If `assumpcheck` does not yet exist on TestPyPI, create a **pending publisher** from your account publishing settings with:

- Project name: `assumpcheck`
- Environment: `testpypi`

If the project already exists, add a normal trusted publisher under that project instead.

### PyPI trusted publisher

If `assumpcheck` does not yet exist on PyPI, create a **pending publisher** from your account publishing settings with:

- Project name: `assumpcheck`
- Environment: `pypi`

If the project already exists, add a normal trusted publisher under that project instead.

## Local preflight

Run this before creating a release tag:

```bash
python -m pip install --upgrade build twine
python -m pytest -q
python -m build
python -m twine check dist/*
```

## Release flow

### 1. Dry run to TestPyPI

Create and push a release-candidate tag:

```bash
git tag v0.1.0rc1
git push origin v0.1.0rc1
```

The `publish.yml` workflow will:

- build the sdist and wheel
- run `twine check`
- publish to TestPyPI using Trusted Publishing

After it succeeds, verify installation in a clean environment:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple assumpcheck
python -c "from assumpcheck import check_anova, check_linear_regression, check_logistic_regression"
```

### 2. Production release to PyPI

Create and push the final tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The same workflow will publish to PyPI when the tag does **not** include `rc`.

After it succeeds, verify the public install:

```bash
pip install assumpcheck
python -c "from assumpcheck import check_anova, check_linear_regression, check_logistic_regression"
```

## After the first PyPI release

- Update the README install instructions to `pip install assumpcheck`
- Create GitHub release notes for the published version
- Bump the version in `pyproject.toml` before preparing the next release
