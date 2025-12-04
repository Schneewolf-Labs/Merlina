# GitHub Actions Workflows Guide

## Overview

Two automated testing workflows have been added to ensure code quality and catch bugs early.

## Workflows

### 🧪 Full Test Suite (`tests.yml`)

**When it runs**: Push to main/master/develop, all pull requests

**What it does**:
```
1. Sets up Python 3.10 and 3.11 (parallel matrix)
2. Installs all dependencies
3. Runs comprehensive API tests (29/29 endpoints)
4. Generates coverage reports
5. Uploads coverage to Codecov
6. Runs code quality checks
7. Posts summary to PR
```

**Time**: ~3-5 minutes

**Example output**:
```
✅ test (3.10) - 60+ tests passed
✅ test (3.11) - 60+ tests passed
✅ lint - Code quality checks passed
📊 Coverage: 85% (uploaded to Codecov)
```

### ⚡ Quick Tests (`quick-test.yml`)

**When it runs**: Every push to any branch

**What it does**:
```
1. Sets up Python 3.10
2. Runs comprehensive API tests
3. Stops on first failure (fast feedback)
4. Comments on PR if tests fail
```

**Time**: ~2-3 minutes

**Example output**:
```
✅ 60+ tests passed in 2m 15s
```

## Status Badges

The following badges are now shown in README.md:

- **Tests**: Full test suite status
- **Quick Tests**: Fast test status
- **Codecov**: Test coverage percentage
- **Python**: Supported Python versions
- **API Coverage**: 100% endpoint coverage

## Viewing Results

### On Pull Requests

1. Open your pull request on GitHub
2. Scroll to the bottom to see check results
3. Click "Details" to view full logs
4. Check summary tab for coverage report

### On Main Branch

1. Go to Actions tab in GitHub
2. Click on latest workflow run
3. View test results and coverage

## Local Development

Run the same tests locally before pushing:

```bash
# Quick test (matches quick-test.yml)
pytest tests/test_api_comprehensive.py -v --tb=short -x

# Full test with coverage (matches tests.yml)
pytest --cov=src --cov=merlina --cov-report=html -v

# Code quality checks
ruff check .
black --check .
mypy src/
```

## Configuration

### Branch Protection

Recommended settings for main branch:

1. Go to Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable: "Require status checks to pass before merging"
4. Select checks:
   - `test (3.10)`
   - `test (3.11)`
   - `quick-test`

### Codecov Setup (Optional)

For private repositories:

1. Go to [codecov.io](https://codecov.io)
2. Enable Merlina repository
3. Copy upload token
4. Add as secret: `CODECOV_TOKEN`

For public repositories, no setup needed!

## Troubleshooting

### ❌ Tests Fail in CI but Pass Locally

**Cause**: Different environment or cached files

**Fix**:
```bash
# Match CI environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r tests/requirements-test.txt
pytest tests/test_api_comprehensive.py -v
```

### ⏱️ Workflow Times Out

**Cause**: Tests taking too long (>5 min timeout on quick-test)

**Fix**: Check for:
- Actual external API calls (should be mocked)
- Large file operations
- Network timeouts

### 📉 Coverage Drops

**Cause**: New code without tests

**Fix**: Add tests for new endpoints in `test_api_comprehensive.py`

## Best Practices

### ✅ DO

- Run tests locally before pushing
- Keep tests fast with proper mocking
- Add tests for new endpoints immediately
- Review coverage reports regularly
- Keep dependencies updated

### ❌ DON'T

- Skip failing tests
- Mock everything (test real logic)
- Ignore code quality warnings
- Push directly to main
- Disable workflows without reason

## Customization

### Add More Python Versions

Edit `.github/workflows/tests.yml`:

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

### Change Timeout

Edit `.github/workflows/quick-test.yml`:

```yaml
jobs:
  quick-test:
    timeout-minutes: 10  # Change from 5
```

### Add Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/test_api_comprehensive.py -x
        language: system
        pass_filenames: false
```

## Monitoring

### GitHub Actions Dashboard

- View all workflow runs
- Download artifacts (coverage reports)
- Re-run failed workflows
- Cancel running workflows

### Codecov Dashboard

- Track coverage trends over time
- View coverage by file
- See uncovered lines
- Compare branches

## Support

### Resources

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.io/)
- [Tests README](../tests/README.md)

### Getting Help

If workflows fail unexpectedly:

1. Check workflow logs on GitHub
2. Run tests locally to reproduce
3. Check recent changes to dependencies
4. Review test file changes
5. Ask in issues or discussions

---

**Created**: 2025-11-16
**Maintained by**: Merlina Development Team
**Status**: ✅ Active and running on all PRs
