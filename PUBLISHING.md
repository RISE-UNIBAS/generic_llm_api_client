# Publishing Guide

This guide explains how to publish the package to PyPI using GitHub Actions automation.

## Overview

This repository uses GitHub Actions to:
1. **Run tests automatically** on every push and pull request
2. **Publish to PyPI automatically** when you create a GitHub release

## Setup (One-Time Configuration)

### 1. Configure PyPI Trusted Publishing

Modern PyPI uses "Trusted Publishers" instead of API tokens. This is more secure and doesn't require storing secrets.

#### Steps:

1. **Go to PyPI**: https://pypi.org/manage/account/publishing/

2. **Add a new publisher**:
   - **PyPI Project Name**: `generic-llm-api-client`
   - **Owner**: `RISE-UNIBAS`
   - **Repository name**: `generic_llm_api_client`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

3. **Click "Add"**

That's it! No API tokens needed. GitHub Actions will authenticate automatically using OIDC.

### 2. Optional: Set up Codecov (for coverage reports)

1. Go to https://codecov.io/ and sign in with GitHub
2. Enable the `RISE-UNIBAS/generic_llm_api_client` repository
3. Get your Codecov token
4. Add it as a GitHub secret:
   - Go to: `https://github.com/RISE-UNIBAS/generic_llm_api_client/settings/secrets/actions`
   - Click "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: Your Codecov token
   - Click "Add secret"

## Automated Testing Workflow

The test workflow (`.github/workflows/tests.yml`) runs automatically on:
- Every push to `main`, `master`, or `develop` branches
- Every pull request

### What it does:

- ✅ Tests on **3 operating systems**: Ubuntu, Windows, macOS
- ✅ Tests on **4 Python versions**: 3.9, 3.10, 3.11, 3.12
- ✅ Runs **unit tests only** (no API calls, no cost)
- ✅ Generates **coverage reports**
- ✅ Checks **code formatting** (black)
- ✅ Runs **linting** (ruff)

### Viewing test results:

Go to: https://github.com/RISE-UNIBAS/generic_llm_api_client/actions

Each commit will show a green checkmark ✅ or red X ❌.

## Publishing to PyPI

### Automated Publishing (Recommended)

Publishing happens automatically when you create a GitHub release:

#### Steps:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment version number
   ```

2. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git push
   ```

3. **Create a GitHub Release**:

   **Option A: Via GitHub UI**
   - Go to: https://github.com/RISE-UNIBAS/generic_llm_api_client/releases/new
   - Tag: `v0.1.1` (must match pyproject.toml version)
   - Release title: `v0.1.1` or `Version 0.1.1`
   - Description: List changes (see example below)
   - Click "Publish release"

   **Option B: Via GitHub CLI**
   ```bash
   gh release create v0.1.1 \
     --title "v0.1.1" \
     --notes "## Changes
   - Added support for feature X
   - Fixed bug Y
   - Improved Z"
   ```

4. **Wait for automation** (1-2 minutes):
   - GitHub Actions builds the package
   - Publishes to PyPI
   - Uploads signed artifacts to GitHub release

5. **Verify on PyPI**:
   - Check: https://pypi.org/project/generic-llm-api-client/

### Manual Publishing (If Needed)

If you need to publish manually for some reason:

```bash
# 1. Install build tools
pip install build twine

# 2. Clean old builds
rm -rf dist/ build/ *.egg-info

# 3. Build the package
python -m build

# 4. Upload to PyPI (requires PyPI API token)
twine upload dist/*
```

## Versioning Strategy

Follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.2.0): New features, backward compatible
- **Patch** (0.1.1): Bug fixes, backward compatible

### Examples:

```toml
# Bug fix release
version = "0.1.1"

# New feature release
version = "0.2.0"

# Breaking change (when ready for v1)
version = "1.0.0"
```

## Release Checklist

Before creating a release:

### 1. Run tests locally
```bash
# Run unit tests
pytest

# Run with coverage
pytest --cov=ai_client --cov-report=term-missing

# Optionally: Run integration tests
pytest -m integration
```

### 2. Update version number
Edit `pyproject.toml`:
```toml
version = "0.1.1"  # Increment appropriately
```

### 3. Update CHANGELOG (optional but recommended)
Create/update `CHANGELOG.md` with release notes:
```markdown
## [0.1.1] - 2025-10-28

### Added
- Support for OpenRouter and sciCORE providers

### Fixed
- Image MIME type detection for all providers
- Gemini client compatibility with google-genai 1.46.0

### Changed
- Improved error messages in integration tests
```

### 4. Commit changes
```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to 0.1.1"
git push
```

### 5. Create GitHub release
```bash
gh release create v0.1.1 \
  --title "v0.1.1" \
  --notes-file CHANGELOG.md
```

### 6. Monitor the workflow
Watch: https://github.com/RISE-UNIBAS/generic_llm_api_client/actions

You should see:
1. ✅ Build distribution
2. ✅ Publish to PyPI
3. ✅ Upload artifacts to release

### 7. Verify publication
Check: https://pypi.org/project/generic-llm-api-client/

Test installation:
```bash
pip install --upgrade generic-llm-api-client
```

## Troubleshooting

### "Trusted publisher configuration does not match"

**Problem**: PyPI rejects the publish because settings don't match.

**Solution**: Check that PyPI trusted publisher settings match exactly:
- Repository owner: `RISE-UNIBAS`
- Repository name: `generic_llm_api_client`
- Workflow: `publish.yml`
- Environment: `pypi`

### "Version already exists"

**Problem**: You're trying to publish a version that's already on PyPI.

**Solution**: You can't overwrite PyPI versions. Increment the version number:
```toml
version = "0.1.2"  # New version
```

### Tests fail in CI but pass locally

**Problem**: Tests pass on your machine but fail on GitHub Actions.

**Possible causes**:
1. Missing dependency in `pyproject.toml`
2. OS-specific issue (test on Linux if you develop on Windows)
3. Python version incompatibility

**Solution**: Check the GitHub Actions logs to see what failed.

### Manual publish if automation fails

If GitHub Actions fails and you need to publish immediately:

```bash
# Get a PyPI API token from https://pypi.org/manage/account/token/
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...your-token...

python -m build
twine upload dist/*
```

## Best Practices

### 1. Always create releases from main/master branch
```bash
# Make sure you're on main
git checkout main
git pull

# Then create release
gh release create v0.1.1
```

### 2. Use descriptive release notes
Bad:
```
v0.1.1
Bug fixes
```

Good:
```
v0.1.1 - Improved Provider Support

## Changes
- Added OpenRouter and sciCORE integration tests
- Fixed image MIME type detection for PNG files across all providers
- Updated Gemini client for google-genai 1.46.0 compatibility

## Bug Fixes
- Fixed Claude client rejecting PNG images (was expecting JPEG)
- Fixed finish_reason enum parsing in Gemini responses
```

### 3. Test before releasing
Always run tests before creating a release:
```bash
pytest
```

### 4. Keep CHANGELOG.md updated
Document all changes in CHANGELOG.md before releasing.

## Summary

**Automated workflow:**
1. Update version in `pyproject.toml`
2. Commit and push
3. Create GitHub release (tag must be `v{version}`)
4. Wait 1-2 minutes
5. Package is on PyPI! 🎉

**Key URLs:**
- Repository: https://github.com/RISE-UNIBAS/generic_llm_api_client
- Actions: https://github.com/RISE-UNIBAS/generic_llm_api_client/actions
- PyPI: https://pypi.org/project/generic-llm-api-client/
- Releases: https://github.com/RISE-UNIBAS/generic_llm_api_client/releases

For questions, open an issue: https://github.com/RISE-UNIBAS/generic_llm_api_client/issues
