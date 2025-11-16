# Makefile for FluxPipeline
# Common development and deployment commands

.PHONY: help install install-dev install-cuda test lint format clean docker-build docker-run gui setup pre-commit

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Python and pip commands
PYTHON := python3
PIP := $(PYTHON) -m pip

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BLUE)Usage:$(NC)\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

install: ## Install basic dependencies (CPU only)
	@echo "$(BLUE)Installing CPU dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete!$(NC)"

install-dev: ## Install with development tools
	@echo "$(BLUE)Installing with development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Development installation complete!$(NC)"

install-cuda: ## Install with CUDA support
	@echo "$(BLUE)Installing CUDA dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements_cuda.txt
	@echo "$(GREEN)✓ CUDA installation complete!$(NC)"

install-rocm: ## Install with ROCm support (AMD GPUs)
	@echo "$(BLUE)Installing ROCm dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements_rocm.txt
	@echo "$(GREEN)✓ ROCm installation complete!$(NC)"

install-intel: ## Install with Intel OneAPI support
	@echo "$(BLUE)Installing Intel dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements_intel.txt
	@echo "$(GREEN)✓ Intel installation complete!$(NC)"

setup: install-dev ## Complete development setup (install + pre-commit)
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if ! command -v pre-commit > /dev/null; then \
		echo "$(YELLOW)Installing pre-commit...$(NC)"; \
		$(PIP) install pre-commit; \
	fi
	pre-commit install
	cp -n .env.example .env || true
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo "$(YELLOW)Don't forget to edit .env with your settings$(NC)"

##@ Code Quality

lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "$(YELLOW)→ Ruff$(NC)"
	ruff check .
	@echo "$(YELLOW)→ Black$(NC)"
	black --check .
	@echo "$(YELLOW)→ MyPy$(NC)"
	mypy . --ignore-missing-imports --no-strict-optional || true
	@echo "$(GREEN)✓ Linting complete!$(NC)"

format: ## Auto-format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	black .
	ruff check --fix .
	@echo "$(GREEN)✓ Code formatted!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	mypy . --ignore-missing-imports --no-strict-optional
	@echo "$(GREEN)✓ Type checking complete!$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks complete!$(NC)"

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v -m unit
	@echo "$(GREEN)✓ Unit tests complete!$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v -m integration
	@echo "$(GREEN)✓ Integration tests complete!$(NC)"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-quick: ## Run tests without slow markers
	@echo "$(BLUE)Running quick tests...$(NC)"
	pytest tests/ -v -m "not slow and not gpu"
	@echo "$(GREEN)✓ Quick tests complete!$(NC)"

##@ Application

gui: ## Launch Gradio web interface
	@echo "$(BLUE)Starting Gradio interface...$(NC)"
	$(PYTHON) gui.py

run: ## Run CLI image generation
	@echo "$(BLUE)Running FluxPipeline CLI...$(NC)"
	$(PYTHON) main.py

interactive: ## Run interactive generation mode
	@echo "$(BLUE)Starting interactive mode...$(NC)"
	$(PYTHON) interactive_generation.py

##@ Docker

docker-build-cpu: ## Build CPU Docker image
	@echo "$(BLUE)Building CPU Docker image...$(NC)"
	docker build -f Dockerfile.cpu -t flux_pipeline:cpu .
	@echo "$(GREEN)✓ CPU image built!$(NC)"

docker-build-cuda: ## Build CUDA Docker image
	@echo "$(BLUE)Building CUDA Docker image...$(NC)"
	docker build -f Dockerfile.cuda -t flux_pipeline:cuda .
	@echo "$(GREEN)✓ CUDA image built!$(NC)"

docker-build-rocm: ## Build ROCm Docker image
	@echo "$(BLUE)Building ROCm Docker image...$(NC)"
	docker build -f Dockerfile.rocm -t flux_pipeline:rocm .
	@echo "$(GREEN)✓ ROCm image built!$(NC)"

docker-build-intel: ## Build Intel Docker image
	@echo "$(BLUE)Building Intel Docker image...$(NC)"
	docker build -f Dockerfile.intel -t flux_pipeline:intel .
	@echo "$(GREEN)✓ Intel image built!$(NC)"

docker-build-all: docker-build-cpu docker-build-cuda docker-build-rocm docker-build-intel ## Build all Docker images
	@echo "$(GREEN)✓ All Docker images built!$(NC)"

docker-run-cpu: ## Run CPU Docker container
	@echo "$(BLUE)Running CPU container...$(NC)"
	docker run -p 7860:7860 flux_pipeline:cpu

docker-run-cuda: ## Run CUDA Docker container
	@echo "$(BLUE)Running CUDA container...$(NC)"
	docker run --gpus all -p 7860:7860 flux_pipeline:cuda

docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting docker-compose services...$(NC)"
	docker-compose up

docker-compose-down: ## Stop docker-compose services
	@echo "$(BLUE)Stopping docker-compose services...$(NC)"
	docker-compose down


##@ Documentation

docs-install: ## Install documentation dependencies
	@echo "$(BLUE)Installing documentation dependencies...$(NC)"
	$(PIP) install -r docs/requirements-docs.txt
	@echo "$(GREEN)✓ Documentation dependencies installed!$(NC)"

docs-build: ## Build documentation with Sphinx
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && sphinx-build -b html . _build/html
	@echo "$(GREEN)✓ Documentation built! Open docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation build...$(NC)"
	rm -rf docs/_build docs/.doctrees
	@echo "$(GREEN)✓ Documentation cleaned!$(NC)"

docs-rebuild: docs-clean docs-build ## Clean and rebuild documentation
	@echo "$(GREEN)✓ Documentation rebuilt!$(NC)"

docs-check: ## Check documentation for errors
	@echo "$(BLUE)Checking documentation...$(NC)"
	cd docs && sphinx-build -b html -W --keep-going . _build/html
	@echo "$(GREEN)✓ Documentation check complete!$(NC)"

##@ Maintenance

clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ htmlcov/ .coverage 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

clean-models: ## Remove downloaded model files (USE WITH CAUTION)
	@echo "$(RED)WARNING: This will delete all model files!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/ checkpoints/ *.safetensors *.pt *.pth *.ckpt; \
		echo "$(GREEN)✓ Model files removed$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

clean-outputs: ## Remove generated images and outputs
	@echo "$(BLUE)Cleaning output files...$(NC)"
	rm -rf workspace/ output/ generated*/ history.json
	@echo "$(GREEN)✓ Outputs cleaned!$(NC)"

clean-all: clean clean-outputs ## Deep clean (cache + outputs, keeps models)
	@echo "$(GREEN)✓ Deep clean complete!$(NC)"

update-deps: ## Update dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Dependencies updated!$(NC)"

##@ Development

dev: install-dev setup ## Quick development setup
	@echo "$(GREEN)✓ Ready for development!$(NC)"

check: lint test ## Run linters and tests
	@echo "$(GREEN)✓ All checks passed!$(NC)"

ci: lint test-cov ## Run CI checks locally
	@echo "$(GREEN)✓ CI checks complete!$(NC)"

##@ Information

info: ## Show project information
	@echo "$(BLUE)FluxPipeline Project Information$(NC)"
	@echo "Version: 0.1.0"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "pip: $$($(PIP) --version)"
	@echo ""
	@echo "$(BLUE)GPU Information:$(NC)"
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>/dev/null || echo "PyTorch not installed"

env-info: ## Show environment information
	@echo "$(BLUE)Environment Variables:$(NC)"
	@env | grep -E "CUDA|TORCH|HF_|GRADIO" || echo "No relevant environment variables set"

##@ Quick Start

quickstart: ## Quick start guide
	@echo "$(BLUE)FluxPipeline Quick Start$(NC)"
	@echo ""
	@echo "1. $(GREEN)make setup$(NC)       - Set up development environment"
	@echo "2. Edit $(YELLOW).env$(NC) file      - Configure your settings"
	@echo "3. $(GREEN)make test$(NC)        - Run tests to verify installation"
	@echo "4. $(GREEN)make gui$(NC)         - Start the web interface"
	@echo ""
	@echo "Or for Docker:"
	@echo "1. $(GREEN)make docker-build-cuda$(NC) - Build Docker image"
	@echo "2. $(GREEN)make docker-run-cuda$(NC)   - Run container"
	@echo ""
	@echo "See $(GREEN)make help$(NC) for all available commands"
