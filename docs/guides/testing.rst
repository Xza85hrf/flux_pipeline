Testing
=======

FluxPipeline includes a comprehensive test suite using pytest.

Running Tests
-------------

All Tests
~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ -v --cov=. --cov-report=html

   # Run with markers
   pytest tests/ -v -m unit
   pytest tests/ -v -m integration

Specific Test Files
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run specific test file
   pytest tests/unit/test_seed_manager.py -v

   # Run specific test
   pytest tests/unit/test_seed_manager.py::TestSeedManager::test_get_seed -v

Using Makefile
~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   make test

   # Run with coverage
   make test-cov

   # Run unit tests only
   make test-unit

   # Run integration tests only
   make test-integration

Test Structure
--------------

Directory Layout
~~~~~~~~~~~~~~~~

::

   tests/
   ├── __init__.py
   ├── conftest.py              # Shared fixtures
   ├── unit/                    # Unit tests
   │   ├── __init__.py
   │   ├── test_gpu_manager.py
   │   ├── test_memory_manager.py
   │   ├── test_seed_manager.py
   │   ├── test_prompt_manager.py
   │   ├── test_gui.py
   │   ├── test_interactive_generation.py
   │   └── test_main.py
   ├── integration/             # Integration tests
   │   ├── __init__.py
   │   ├── test_pipeline_integration.py
   │   └── test_gui_integration.py
   └── performance/             # Performance tests
       ├── __init__.py
       └── test_generation_performance.py

Test Markers
~~~~~~~~~~~~

- ``@pytest.mark.unit`` - Unit tests (fast, isolated)
- ``@pytest.mark.integration`` - Integration tests (slower, full stack)
- ``@pytest.mark.slow`` - Slow tests (may take minutes)
- ``@pytest.mark.gpu`` - Requires GPU
- ``@pytest.mark.cuda`` - Requires CUDA
- ``@pytest.mark.performance`` - Performance benchmarks

Writing Tests
-------------

Unit Test Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   """Unit test example."""
   import pytest
   from core import SeedManager, SeedProfile

   @pytest.mark.unit
   class TestSeedManager:
       """Test SeedManager functionality."""

       def test_get_seed_with_profile(self):
           """Test seed generation with profile."""
           manager = SeedManager()
           
           # Test conservative profile
           seed = manager.get_seed(SeedProfile.CONSERVATIVE)
           assert 42 <= seed <= 9999
           
           # Test balanced profile
           seed = manager.get_seed(SeedProfile.BALANCED)
           assert 10000 <= seed <= 999999
           
           # Test creative profile
           seed = manager.get_seed(SeedProfile.CREATIVE)
           assert 1000000 <= seed <= 2147483647

       def test_fixed_seed(self):
           """Test fixed seed returns exact value."""
           manager = SeedManager()
           
           fixed_seed = 42
           result = manager.get_seed(seed=fixed_seed)
           assert result == fixed_seed

Integration Test Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   """Integration test example."""
   import pytest
   from pipeline import FluxPipeline
   from utils import setup_workspace

   @pytest.mark.integration
   @pytest.mark.slow
   class TestPipelineIntegration:
       """Test full pipeline integration."""

       @pytest.fixture
       def pipeline(self, tmp_path):
           """Create test pipeline."""
           workspace = tmp_path / "workspace"
           workspace.mkdir()
           return FluxPipeline(workspace=workspace)

       def test_full_generation_workflow(self, pipeline):
           """Test complete generation workflow."""
           # Load model
           assert pipeline.load_model()
           
           # Generate image
           prompt = "A test image"
           image, seed = pipeline.generate_image(
               prompt=prompt,
               height=512,
               width=512,
               num_inference_steps=1
           )
           
           assert image is not None
           assert isinstance(seed, int)
           assert seed >= 0

Fixtures
--------

Common Fixtures (conftest.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   """Shared test fixtures."""
   import pytest
   from pathlib import Path

   @pytest.fixture
   def temp_workspace(tmp_path):
       """Create temporary workspace."""
       workspace = tmp_path / "workspace"
       workspace.mkdir()
       return workspace

   @pytest.fixture
   def mock_pipeline(mocker, temp_workspace):
       """Create mocked pipeline."""
       pipeline = mocker.Mock()
       pipeline.workspace = temp_workspace
       return pipeline

Using Fixtures
~~~~~~~~~~~~~~

.. code-block:: python

   def test_with_workspace(temp_workspace):
       """Test using workspace fixture."""
       output_file = temp_workspace / "test.txt"
       output_file.write_text("test")
       
       assert output_file.exists()
       assert output_file.read_text() == "test"

Mocking
-------

Mock External Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch

   @pytest.mark.unit
   def test_with_mocked_torch(mocker):
       """Test with mocked PyTorch."""
       # Mock torch.cuda.is_available
       mocker.patch('torch.cuda.is_available', return_value=True)
       
       # Your test code
       from core import MultiGPUManager
       manager = MultiGPUManager()
       # Test logic...

Mock Pipeline Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.mark.unit
   def test_generation_mocked(mocker):
       """Test generation with mocked pipeline."""
       from PIL import Image
       import numpy as np
       
       # Create fake image
       fake_image = Image.fromarray(
           np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
       )
       
       # Mock generate_image
       mocker.patch(
           'pipeline.FluxPipeline.generate_image',
           return_value=(fake_image, 42)
       )
       
       # Test code that uses pipeline
       # ...

Coverage
--------

Generate Coverage Report
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # HTML report
   pytest tests/ --cov=. --cov-report=html
   open htmlcov/index.html

   # Terminal report
   pytest tests/ --cov=. --cov-report=term

   # XML report (for CI)
   pytest tests/ --cov=. --cov-report=xml

Coverage Configuration
~~~~~~~~~~~~~~~~~~~~~~

See ``.coveragerc``:

.. code-block:: ini

   [run]
   omit = 
       tests/*
       */venv/*
       */__pycache__/*

   [report]
   exclude_lines =
       pragma: no cover
       def __repr__
       raise AssertionError
       raise NotImplementedError

Continuous Integration
----------------------

GitHub Actions
~~~~~~~~~~~~~~

Tests run automatically on push and PR. See ``.github/workflows/tests.yml``.

Local CI Simulation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run like CI
   make test-all

   # Check code quality
   make lint

Best Practices
--------------

1. **Write tests first** (TDD when possible)
2. **Test one thing** per test function
3. **Use descriptive names** for test functions
4. **Mock external dependencies** to keep tests fast
5. **Use fixtures** for common setup
6. **Add markers** to categorize tests
7. **Maintain >80% coverage** for core modules
8. **Run tests before commit** (pre-commit hook)

Debugging Tests
---------------

Run with Debug Output
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Show print statements
   pytest tests/ -v -s

   # Show locals on failure
   pytest tests/ -v -l

   # Drop to debugger on failure
   pytest tests/ -v --pdb

Using IDE Debugger
~~~~~~~~~~~~~~~~~~

Most IDEs (PyCharm, VSCode) support pytest debugging:

1. Set breakpoints in test code
2. Run test in debug mode
3. Step through execution

Next Steps
----------

- See :doc:`contributing` for contribution guidelines
- Check :doc:`architecture` for system design
- Browse ``tests/`` directory for examples
