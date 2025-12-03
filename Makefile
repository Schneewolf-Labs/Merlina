.PHONY: help test test-cov test-parallel test-api test-unit clean-test install-test

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)Merlina Test Commands$(NC)"
	@echo ""
	@echo "$(GREEN)make install-test$(NC)     - Install test dependencies"
	@echo "$(GREEN)make test$(NC)             - Run all tests"
	@echo "$(GREEN)make test-cov$(NC)         - Run tests with coverage report"
	@echo "$(GREEN)make test-parallel$(NC)    - Run tests in parallel (faster)"
	@echo "$(GREEN)make test-api$(NC)         - Run only API tests"
	@echo "$(GREEN)make test-unit$(NC)        - Run only unit tests"
	@echo "$(GREEN)make test-verbose$(NC)     - Run tests with verbose output"
	@echo "$(GREEN)make clean-test$(NC)       - Clean test artifacts"
	@echo ""

install-test:
	@echo "$(BLUE)Installing test dependencies...$(NC)"
	pip install -r tests/requirements-test.txt

test:
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest --cov=src --cov=merlina --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

test-parallel:
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	pytest -n auto

test-api:
	@echo "$(BLUE)Running API tests...$(NC)"
	pytest -m api -v

test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest -m unit -v

test-verbose:
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	pytest -vv

test-comprehensive:
	@echo "$(BLUE)Running comprehensive API tests...$(NC)"
	pytest tests/test_api_comprehensive.py -v

clean-test:
	@echo "$(BLUE)Cleaning test artifacts...$(NC)"
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Test artifacts cleaned!$(NC)"

# Development commands
dev-setup: install-test
	@echo "$(BLUE)Setting up development environment...$(NC)"
	pip install -r requirements.txt
	pip install -r tests/requirements-test.txt
	@echo "$(GREEN)Development environment ready!$(NC)"

# Quick test command for CI
ci-test:
	@echo "$(BLUE)Running CI tests...$(NC)"
	pytest --cov=src --cov=merlina --cov-report=xml --cov-report=term -v
