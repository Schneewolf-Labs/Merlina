# GitHub Actions Workflows

This directory contains CI/CD workflows for automated testing and quality checks.

## Workflows

### 1. Tests (`tests.yml`)

**Triggers**: Push to main/master/develop, Pull Requests

**What it does**:
- Runs on Python 3.10 and 3.11 (matrix testing)
- Installs all dependencies
- Runs comprehensive API test suite
- Generates coverage reports
- Uploads coverage to Codecov
- Runs code quality checks (ruff, black, mypy)

**Jobs**:
- `test`: Runs all tests with coverage
- `lint`: Code quality checks
- `test-summary`: Posts summary to GitHub

**Duration**: ~3-5 minutes

### 2. Quick Tests (`quick-test.yml`)

**Triggers**: Push to any branch

**What it does**:
- Fast feedback on every commit
- Runs only comprehensive API tests
- Times out after 5 minutes
- Comments on PR if tests fail

**Duration**: ~2-3 minutes

## Status Badges

Add these badges to your README:

```markdown
[![Tests](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/tests.yml/badge.svg)](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/tests.yml)
[![Quick Tests](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/quick-test.yml/badge.svg)](https://github.com/Schneewolf-Labs/Merlina/actions/workflows/quick-test.yml)
[![codecov](https://codecov.io/gh/Schneewolf-Labs/Merlina/branch/main/graph/badge.svg)](https://codecov.io/gh/Schneewolf-Labs/Merlina)
```

## Configuration

### Required Secrets

None required for basic testing. Optional:

- `CODECOV_TOKEN`: For private repo coverage reports
- `HF_TOKEN`: For testing with gated models

### Permissions

Default permissions are sufficient. The workflows need:
- Read access to repository
- Write access to checks (automatic)

## Running Locally

To run the same tests locally:

```bash
# Run all tests
pytest tests/test_api_comprehensive.py -v

# Run with coverage
pytest --cov=src --cov=merlina --cov-report=html

# Quick test (matches quick-test.yml)
pytest tests/test_api_comprehensive.py -v --tb=short -x
```

## Customization

### Change Python Versions

Edit `tests.yml`:

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

### Add More Tests

Edit the test step:

```yaml
- name: Run all tests
  run: |
    pytest tests/ -v
```

### Disable Lint Checks

Remove or comment out the `lint` job in `tests.yml`.

## Troubleshooting

### Tests Fail Locally but Pass in CI

- Check Python version: `python --version`
- Clear pytest cache: `pytest --cache-clear`
- Install exact dependencies: `pip install -r requirements.txt -r tests/requirements-test.txt`

### Slow Tests

- Use `quick-test.yml` for fast feedback
- Run specific tests: `pytest tests/test_api_comprehensive.py::TestTrainingEndpoints`
- Enable parallel testing: `pytest -n auto`

### Coverage Not Uploading

- Check Codecov token (for private repos)
- Verify coverage.xml is generated
- Check Codecov action logs

## Best Practices

1. **Keep tests fast**: Use mocking to avoid slow operations
2. **Fail fast**: Use `-x` flag to stop on first failure
3. **Matrix testing**: Test multiple Python versions
4. **Branch protection**: Require tests to pass before merge
5. **Coverage goals**: Maintain >80% coverage

## Maintenance

These workflows are automatically maintained and updated with the test suite. No manual updates needed unless:

- Adding new Python versions
- Changing test structure
- Adding new dependencies
- Modifying coverage requirements

---

**Last Updated**: 2025-11-16
**Maintained by**: Merlina Development Team
