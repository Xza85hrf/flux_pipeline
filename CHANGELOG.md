# Changelog

All notable changes to the FluxPipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project infrastructure improvements:
  - Added `.gitignore` to prevent committing build artifacts and sensitive files
  - Added `pyproject.toml` for modern Python package configuration
  - Added `.env.example` template for environment configuration
  - Added GitHub Actions CI/CD workflows (tests, linting, Docker builds, security scans)
  - Added this CHANGELOG.md to track project changes

### Changed
- Updated Python version requirement from 3.8+ to 3.10+ (recommended: 3.11-3.13)
- Upgraded core dependencies to latest compatible versions:
  - PyTorch: 2.5.0 → 2.5.1
  - Transformers: 4.46.1 → 4.57.1
  - Diffusers: 0.31.0 → 0.34.0
  - Gradio: 5.4.0 → 5.9.1
  - OpenCV-Python: 4.8.1.78 → 4.10.0.84
  - Scikit-image: 0.21.0 → 0.24.0
  - PyTest: 7.4.2 → 8.3.4
  - And many more (see pyproject.toml for complete list)
- Updated Dockerfile.cuda base image from CUDA 12.0.1 to 12.4.1
- Updated CUDA Python from 12.4.0 to 12.6.0
- Improved exception handling by replacing bare `except:` with specific exception types

### Fixed
- Fixed bare exception handling in gui.py (added json.JSONDecodeError, IOError)
- Fixed bare exception handling in core/memory_manager.py (added RuntimeError, AttributeError)
- Fixed bare exception handling in pipeline/flux_pipeline.py (added RuntimeError, AttributeError)
- Created missing `requirements_cpu.txt` file referenced by Dockerfile.cpu

### Security
- Updated opencv-python to address CVE-2023-4863 (libwebp vulnerability)
- All dependencies updated to latest versions with security patches
- Added GitHub Actions security scanning workflow

## [0.1.0] - 2024-11-16

### Added
- Initial project setup with FLUX.1-schnell image generation support
- Core components:
  - Memory Manager for GPU and system memory optimization
  - GPU Manager for multi-vendor GPU support (NVIDIA, AMD, Intel)
  - Prompt Manager for text processing
  - Seed Manager for reproducible generation
  - Flux Pipeline for image generation workflow
- User interfaces:
  - Gradio-based web GUI with batch processing
  - Command-line interface for programmatic access
- Docker support:
  - Dockerfile.cpu for CPU-only environments
  - Dockerfile.cuda for NVIDIA GPU support
  - Dockerfile.rocm for AMD GPU support
  - Dockerfile.intel for Intel GPU support
- Comprehensive test suite with 15 test files (3,463 lines)
- Documentation:
  - README.md with installation and usage instructions
  - conda_setup.md for Anaconda environment setup
  - project_summary.md with architecture overview
  - tests.md for testing guidelines

### Features
- Single image generation with various seed profiles
- GIF sequence generation
- Batch processing capabilities
- Multi-GPU vendor support (CUDA, ROCm, OneAPI)
- Memory-efficient attention mechanisms
- Real-time progress tracking
- Generation history management
- Example prompt library
- Configurable generation parameters

---

## Version History

### Version Numbering
This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for added functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

### Release Schedule
- Development releases: As needed
- Stable releases: When major features are complete and tested
- Security patches: As soon as possible after discovery

---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Reporting Issues

Found a bug or have a feature request? Please check our [issue tracker](https://github.com/Xza85hrf/flux_pipeline/issues).
