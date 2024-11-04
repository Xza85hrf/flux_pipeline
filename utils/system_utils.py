"""System utility functions for the FluxPipeline project.

This module provides essential system utilities including:
- Stdout suppression for clean output
- NLTK setup and initialization
- Safe imports for optional dependencies
- Workspace directory management
- Warning suppression
- Unique filename generation

The utilities handle common system operations with proper error handling
and logging, making them suitable for both development and production use.

Example:
    Basic usage:
    ```python
    from utils.system_utils import setup_workspace, suppress_warnings
    from pathlib import Path
    
    # Setup workspace
    workspace = setup_workspace(Path("my_project"))
    
    # Suppress warnings
    suppress_warnings()
    
    # Use context manager for clean output
    with suppress_stdout():
        # Run noisy operation
        pass
    ```

Note:
    All functions include proper error handling and logging,
    making them safe to use in production environments.
"""

import os
import gc
import contextlib
import warnings
import nltk
from pathlib import Path
from typing import Optional, Tuple, Any
from config.logging_config import logger


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output.

    This context manager temporarily redirects stdout to devnull,
    useful for suppressing noisy output from external libraries.

    Yields:
        None: Context manager yield

    Example:
        ```python
        # Suppress noisy output
        with suppress_stdout():
            noisy_function()

        # Output is now restored
        print("This will show")
        ```

    Note:
        Properly restores stdout even if an exception occurs.
    """
    try:
        with open(os.devnull, "w") as devnull:
            old_stdout = os.dup(1)
            try:
                # Redirect stdout to devnull
                os.dup2(devnull.fileno(), 1)
                yield
            finally:
                # Restore stdout
                os.dup2(old_stdout, 1)
    except Exception as e:
        logger.warning(f"Error in stdout suppression: {str(e)}")
        yield


def setup_nltk():
    """Initialize NLTK with required data.

    Downloads and sets up the NLTK punkt tokenizer if not already available.
    Falls back to basic sentence splitting if setup fails.

    Example:
        ```python
        # Setup NLTK at application start
        setup_nltk()

        # Now NLTK functions can be used
        from nltk import sent_tokenize
        sentences = sent_tokenize(text)
        ```
    """
    try:
        # Check if punkt tokenizer is available
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            # Download required data
            logger.info("Downloading required NLTK data...")
            nltk.download("punkt", quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {str(e)}")
            logger.warning("Falling back to basic sentence splitting")
    except Exception as e:
        logger.warning(f"NLTK initialization error: {str(e)}")


def safe_import_xformers() -> bool:
    """Safely import xformers optimization library.

    Attempts to import xformers and its operations module with proper
    error handling and logging.

    Returns:
        bool: True if import successful, False otherwise

    Example:
        ```python
        if safe_import_xformers():
            # Use xformers optimizations
            model.enable_xformers_memory_efficient_attention()
        else:
            # Fall back to standard attention
            pass
        ```
    """
    try:
        import xformers
        import xformers.ops

        logger.info("Successfully imported xformers")
        return True
    except ImportError:
        logger.warning(
            "xformers not found. Consider installing for better performance."
        )
        return False
    except Exception as e:
        logger.warning(f"Unexpected error importing xformers: {str(e)}")
        return False


def safe_import_flux() -> Tuple[Any, Optional[str]]:
    """Safely import FluxPipeline with error handling.

    Returns:
        Tuple[Any, Optional[str]]: Tuple containing:
            - FluxPipeline class if import successful, None otherwise
            - Error message if import failed, None otherwise

    Example:
        ```python
        FluxPipeline, error = safe_import_flux()
        if error:
            print(f"Failed to import: {error}")
        else:
            pipeline = FluxPipeline.from_pretrained(model_id)
        ```
    """
    try:
        from diffusers import FluxPipeline

        logger.info("Successfully imported FluxPipeline")
        return FluxPipeline, None
    except Exception as e:
        logger.error(f"Error importing FluxPipeline: {str(e)}")
        return None, str(e)


def setup_workspace(base_path: Optional[Path] = None) -> Path:
    """Setup workspace directory structure.

    Creates a standardized directory structure for the project:
    - outputs/
        - images/
        - logs/
        - stats/
    - cache/

    Args:
        base_path (Optional[Path], optional): Base directory for workspace.
            Defaults to ./workspace.

    Returns:
        Path: Path to created workspace directory

    Raises:
        Exception: If directory creation fails

    Example:
        ```python
        # Use default workspace
        workspace = setup_workspace()

        # Use custom location
        workspace = setup_workspace(Path("/path/to/workspace"))
        ```
    """
    if base_path is None:
        base_path = Path.cwd() / "workspace"

    try:
        # Create standard directory structure
        directories = [
            "outputs",
            "outputs/images",
            "outputs/logs",
            "outputs/stats",
            "cache",
        ]

        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Workspace setup complete at {base_path}")
        return base_path

    except Exception as e:
        logger.error(f"Error setting up workspace: {str(e)}")
        raise


def suppress_warnings():
    """Suppress common warning messages.

    Filters out specific warning categories and messages that are
    known to be non-problematic:
    - UserWarnings
    - FutureWarnings
    - Specific tokenizer warnings

    Example:
        ```python
        # Call at application startup
        suppress_warnings()

        # Warnings will now be filtered
        noisy_function()  # Warnings suppressed
        ```
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*You set `add_prefix_space`.*")
    warnings.filterwarnings(
        "ignore", message=".*The tokenizer class you load from this checkpoint.*"
    )


def get_unique_filename(base_path: Path, prefix: str = "", suffix: str = "") -> Path:
    """Generate a unique filename with timestamp.

    Creates a unique filename by combining:
    - Optional prefix
    - Timestamp
    - Counter (if needed)
    - Optional suffix

    Args:
        base_path (Path): Directory for the file
        prefix (str, optional): Prefix for filename. Defaults to "".
        suffix (str, optional): Suffix for filename. Defaults to "".

    Returns:
        Path: Path to unique filename

    Example:
        ```python
        # Generate unique image filename
        path = get_unique_filename(
            Path("outputs"),
            prefix="image_",
            suffix=".png"
        )
        print(path)  # outputs/image_20230101_120000.png

        # Save file
        image.save(path)
        ```
    """
    from datetime import datetime

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1

    # Find unique filename
    while True:
        if counter == 1:
            filename = f"{prefix}{timestamp}{suffix}"
        else:
            filename = f"{prefix}{timestamp}_{counter}{suffix}"

        full_path = base_path / filename
        if not full_path.exists():
            return full_path
        counter += 1
