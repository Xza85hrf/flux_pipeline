"""
Unit tests for the SeedManager module.

This module contains tests that validate the functionality of the SeedManager class, which is responsible for managing seed profiles and generating seeds in a machine learning application. The tests cover various aspects such as seed generation, profile management, and history clearing.
"""

import pytest
from unittest.mock import patch, Mock, mock_open
from core.seed_manager import SeedProfile, SeedManager


@pytest.mark.parametrize(
    "clear_profile",
    [
        SeedProfile.CONSERVATIVE,
        SeedProfile.BALANCED,
        SeedProfile.EXPERIMENTAL,
    ],
)
def test_history_clearing(clear_profile):
    """
    Test history clearing functionality.

    Tests:
    - History clearing for different seed profiles
    - Verification that cleared profile's history is empty
    - Verification that other profiles' histories remain intact

    Args:
        clear_profile: The seed profile to clear history for
    """
    with patch("pathlib.Path.exists", return_value=True), patch(
        "builtins.open", mock_open()
    ), patch("pathlib.Path.rename") as mock_rename:

        manager = SeedManager()

        # Generate seeds for all profiles
        for profile in SeedProfile:
            manager.set_profile(profile)
            manager.generate_seed()

        # Clear history for specific profile
        manager.clear_history(clear_profile)

        # Verify cleared profile is empty
        assert not manager.used_seeds[clear_profile.value]

        # Verify other profiles still have seeds
        other_profiles = [p for p in SeedProfile if p != clear_profile]
        assert any(len(manager.used_seeds[p.value]) > 0 for p in other_profiles)
