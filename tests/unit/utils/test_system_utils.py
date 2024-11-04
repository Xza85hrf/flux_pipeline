"""Unit tests for system utilities."""

import pytest
import sys
import os
from unittest.mock import patch, Mock, MagicMock
from utils.system_utils import suppress_stdout, safe_import_flux
import builtins


def test_suppress_stdout():
    """Test stdout suppression context manager."""
    original_stdout = sys.stdout
    test_output = "This should be suppressed"

    with suppress_stdout():
        print(test_output)

    assert sys.stdout == original_stdout


def test_suppress_stdout_error_handling(caplog):
    """Test error handling in suppress_stdout."""
    mock_file = Mock()
    mock_file.__enter__ = Mock(side_effect=OSError("Failed to redirect stdout"))
    mock_file.__exit__ = Mock()

    with patch("builtins.open", return_value=mock_file):
        with suppress_stdout():
            print("This should not raise an error")
    # Check that the warning was logged
    assert any(
        "Failed to redirect stdout" in record.message for record in caplog.records
    )


@pytest.mark.parametrize(
    "scenario_name,error",
    [
        ("read_error", OSError("Failed to redirect stdout")),
        ("write_error", OSError("Failed to redirect stdout")),
        ("general_error", Exception("Unexpected error")),
    ],
)
def test_suppress_stdout_error_scenarios(scenario_name, error, caplog):
    """Test various error scenarios in suppress_stdout."""
    if scenario_name == "read_error":
        with patch("os.dup", side_effect=error):
            with suppress_stdout():
                print("This should not raise an error")
        assert any(
            "Failed to redirect stdout" in record.message for record in caplog.records
        )

    elif scenario_name == "write_error":
        with patch("os.dup2", side_effect=error):
            with pytest.raises(OSError) as exc_info:
                with suppress_stdout():
                    print("This should raise an error")
            assert str(exc_info.value) == "Failed to redirect stdout"

    else:
        mock_file = Mock()
        mock_file.__enter__ = Mock(side_effect=error)
        mock_file.__exit__ = Mock()

        with patch("builtins.open", return_value=mock_file):
            with suppress_stdout():
                print("This should be handled gracefully")
        assert any("Unexpected error" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "import_scenario",
    [
        {
            "name": "successful_import",
            "expected_error": None,
        },
        {
            "name": "import_error",
            "raise_import_error": True,
            "expected_error": "No module named 'diffusers'",
        },
        {
            "name": "general_error",
            "raise_general_error": True,
            "expected_error": "General error",
        },
    ],
)
def test_safe_import_flux(import_scenario, caplog):
    """Test safe importing of Flux pipeline with various scenarios."""
    if "raise_import_error" in import_scenario:
        # Simulate ImportError during import
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'diffusers'"),
        ):
            result, error = safe_import_flux()
            assert error is not None
            assert result is None
            assert "Error importing FluxPipeline" in caplog.text
            assert "No module named 'diffusers'" in error
    elif "raise_general_error" in import_scenario:
        # Simulate general exception during import

        original_import = builtins.__import__

        def mock_import(name, globals, locals, fromlist, level):
            if name == "diffusers" and fromlist and "FluxPipeline" in fromlist:
                raise Exception("General error")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mock_import):
            result, error = safe_import_flux()
            assert str(error) == "General error"
            assert result is None
            assert "Error importing FluxPipeline: General error" in caplog.text
    else:
        # Successful import scenario
        mock_flux_pipeline = Mock()
        mock_diffusers = Mock(FluxPipeline=mock_flux_pipeline)
        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            result, error = safe_import_flux()
            assert error is None
            assert result is not None
