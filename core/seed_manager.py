"""Seed management module for consistent and controlled image generation.

This module provides comprehensive seed management capabilities for AI image generation,
including:
- Multiple seed generation profiles (Conservative, Balanced, Experimental)
- Seed history tracking and persistence
- Seed validation and range management
- Usage statistics and analytics
- Profile-based seed generation strategies

The module ensures reproducible results while providing flexibility in seed selection
and management across different generation scenarios.

Example:
    Basic usage:
    ```python
    from core.seed_manager import SeedManager, SeedProfile
    
    # Initialize seed manager
    manager = SeedManager()
    
    # Set profile for specific use case
    manager.set_profile(SeedProfile.BALANCED)
    
    # Generate seed
    seed = manager.generate_seed()
    print(f"Generated seed: {seed}")
    
    # Get seed information
    info = manager.get_seed_info(seed)
    print(f"Seed info: {info}")
    ```

Note:
    Seeds are crucial for reproducible image generation. Different profiles offer
    varying levels of stability vs. experimentation in the generation process.
"""

import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from config.logging_config import logger


class SeedProfile(Enum):
    """Enumeration of seed generation profiles.
    
    Each profile represents a different strategy for seed generation:
    - CONSERVATIVE: Lower range for stable, consistent results
    - BALANCED: Middle range for general purpose use
    - EXPERIMENTAL: High range for unique variations
    - FULL_RANGE: Complete range for maximum diversity
    
    Example:
        ```python
        profile = SeedProfile.BALANCED
        print(f"Profile range: {profile.value}")
        ```
    """
    CONSERVATIVE = "conservative"  # Uses lower range seeds (0 to 999,999)
    BALANCED = "balanced"  # Uses middle range (1,000,000 to 999,999,999)
    EXPERIMENTAL = (
        "experimental"  # Uses high range seeds (1,000,000,000 to 4,294,967,295)
    )
    FULL_RANGE = "full_range"  # Uses the complete 64-bit range


@dataclass
class SeedRange:
    """Data structure for defining seed range parameters.
    
    Attributes:
        start (int): Starting value of the seed range
        end (int): Ending value of the seed range
        description (str): Human-readable description of the range
    
    Example:
        ```python
        range = SeedRange(
            start=0,
            end=999999,
            description="Conservative range for stable results"
        )
        ```
    """
    start: int
    end: int
    description: str


class SeedManager:
    """Manager class for seed generation and tracking.
    
    This class provides comprehensive seed management capabilities including:
    - Profile-based seed generation
    - Seed history tracking
    - Range validation
    - Usage statistics
    - Persistence management
    
    Attributes:
        MAX_32BIT_SEED (int): Maximum value for 32-bit seeds
        MAX_64BIT_SEED (int): Maximum value for 64-bit seeds
        profiles (Dict[SeedProfile, SeedRange]): Available seed profiles
        current_profile (SeedProfile): Active seed profile
        seed_history_file (Path): File for persisting seed history
        used_seeds (Dict[str, List[int]]): Tracking of used seeds by profile
    
    Example:
        ```python
        manager = SeedManager()
        
        # Generate seeds with different profiles
        conservative_seed = manager.generate_seed()
        
        manager.set_profile(SeedProfile.EXPERIMENTAL)
        experimental_seed = manager.generate_seed()
        
        # Get usage statistics
        stats = manager.get_seed_statistics()
        print(f"Seed usage stats: {stats}")
        ```
    """

    def __init__(self):
        """Initialize the seed manager with default settings."""
        self.MAX_32BIT_SEED = 2**32 - 1
        self.MAX_64BIT_SEED = 2**64 - 1

        # Define available seed profiles with their ranges
        self.profiles = {
            SeedProfile.CONSERVATIVE: SeedRange(
                start=0,
                end=999_999,
                description="Conservative range for stable results",
            ),
            SeedProfile.BALANCED: SeedRange(
                start=1_000_000,
                end=999_999_999,
                description="Balanced range for general use",
            ),
            SeedProfile.EXPERIMENTAL: SeedRange(
                start=1_000_000_000,
                end=self.MAX_32BIT_SEED,
                description="Experimental range for unique results",
            ),
            SeedProfile.FULL_RANGE: SeedRange(
                start=0,
                end=self.MAX_64BIT_SEED,
                description="Full 64-bit range for maximum variation",
            ),
        }

        # Set default profile and initialize history tracking
        self.current_profile = SeedProfile.BALANCED
        self.seed_history_file = Path("seed_history.txt")
        self.used_seeds: Dict[str, List[int]] = {
            profile.value: [] for profile in SeedProfile
        }
        self._load_seed_history()

    def _load_seed_history(self):
        """Load and parse seed usage history from file.
        
        This method:
        1. Reads the seed history file if it exists
        2. Parses each line into timestamp, profile, and seed
        3. Validates and stores the seed information
        4. Handles corrupted history gracefully
        
        Note:
            If the history file is corrupted, it will be archived and a new one created.
        """
        if self.seed_history_file.exists():
            try:
                with open(self.seed_history_file, "r") as f:
                    for line in f:
                        try:
                            # Parse history line (format: timestamp:profile:seed:info)
                            if ":" in line:
                                parts = line.strip().split(":", 3)
                                if len(parts) >= 3:
                                    timestamp, profile, seed = parts[:3]
                                    if profile in self.used_seeds:
                                        try:
                                            seed_val = int(seed)
                                            self.used_seeds[profile].append(seed_val)
                                        except ValueError:
                                            logger.warning(
                                                f"Invalid seed value in history: {seed}"
                                            )
                        except Exception as line_error:
                            logger.warning(f"Error parsing history line: {line_error}")
                            continue
            except Exception as e:
                logger.warning(f"Error loading seed history: {e}")
                # Archive corrupted file and start fresh
                self.seed_history_file.unlink(missing_ok=True)
                self.used_seeds = {profile.value: [] for profile in SeedProfile}

    def _save_seed_use(self, profile: str, seed: int, additional_info: str = ""):
        """Record seed usage to history file.
        
        Args:
            profile (str): Profile name the seed was generated with
            seed (int): The generated seed value
            additional_info (str, optional): Extra information about seed usage
        
        Note:
            Format: timestamp:profile:seed:additional_info
        """
        try:
            with open(self.seed_history_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp}:{profile}:{seed}:{additional_info}\n")
        except Exception as e:
            logger.warning(f"Error saving seed history: {e}")

    def set_profile(self, profile: SeedProfile):
        """Set the active seed generation profile.
        
        Args:
            profile (SeedProfile): The profile to set as active
        
        Example:
            ```python
            manager = SeedManager()
            
            # Switch to experimental profile
            manager.set_profile(SeedProfile.EXPERIMENTAL)
            
            # Generate seed with new profile
            seed = manager.generate_seed()
            ```
        """
        self.current_profile = profile
        logger.info(f"Set seed profile to {profile.value}")

    def generate_seed(self, manual_seed: Optional[int] = None) -> int:
        """Generate or validate a seed value.
        
        This method either generates a random seed within the current profile's
        range or validates and adjusts a manually provided seed.
        
        Args:
            manual_seed (Optional[int], optional): Specific seed to use. Defaults to None.
            
        Returns:
            int: Valid seed value within current profile's range
        
        Example:
            ```python
            # Generate random seed
            seed1 = manager.generate_seed()
            
            # Use specific seed
            seed2 = manager.generate_seed(manual_seed=12345)
            ```
        """
        if manual_seed is not None:
            seed = self._validate_seed(manual_seed)
        else:
            range_info = self.profiles[self.current_profile]
            seed = random.randint(range_info.start, range_info.end)

        # Track seed usage
        self.used_seeds[self.current_profile.value].append(seed)
        self._save_seed_use(
            self.current_profile.value, seed, f"Manual seed: {manual_seed is not None}"
        )
        return seed

    def _validate_seed(self, seed: int) -> int:
        """Validate and adjust seed to fit profile range.
        
        Args:
            seed (int): Seed value to validate
            
        Returns:
            int: Adjusted seed value within valid range
            
        Note:
            Negative seeds are converted to positive.
            Seeds outside range are wrapped to fit.
        """
        range_info = self.profiles[self.current_profile]

        # Handle negative seeds
        if seed < 0:
            logger.warning("Negative seed provided. Converting to positive.")
            seed = abs(seed)

        # Wrap seeds to valid range
        if seed > range_info.end:
            logger.warning(
                f"Seed exceeds current profile range. Wrapping to valid range."
            )
            seed = range_info.start + (seed % (range_info.end - range_info.start + 1))

        return seed

    def get_profile_info(self) -> Dict[str, str]:
        """Get detailed information about all seed profiles.
        
        Returns:
            Dict[str, str]: Profile information including ranges and usage
        
        Example:
            ```python
            info = manager.get_profile_info()
            for profile, details in info.items():
                print(f"{profile}:\n{details}\n")
            ```
        """
        return {
            profile.value: (
                f"Range: {range_info.start:,} to {range_info.end:,}\n"
                f"Description: {range_info.description}\n"
                f"Used seeds: {len(self.used_seeds[profile.value])}"
            )
            for profile, range_info in self.profiles.items()
        }

    def get_seed_info(self, seed: int) -> str:
        """Get information about a specific seed value.
        
        Args:
            seed (int): Seed value to analyze
            
        Returns:
            str: Information about the seed including profile and usage
        
        Example:
            ```python
            seed = manager.generate_seed()
            info = manager.get_seed_info(seed)
            print(f"Seed information: {info}")
            ```
        """
        # Find which profile range contains the seed
        for profile, range_info in self.profiles.items():
            if range_info.start <= seed <= range_info.end:
                usage_count = self.used_seeds[profile.value].count(seed)
                return (
                    f"Seed {seed:,} ({profile.value} profile)\n"
                    f"Times used: {usage_count}"
                )
        return f"Seed {seed:,} exceeds normal ranges"

    def clear_history(self, profile: Optional[SeedProfile] = None):
        """Clear seed usage history.
        
        Args:
            profile (Optional[SeedProfile], optional): Specific profile to clear.
                If None, clears all profiles. Defaults to None.
        
        Example:
            ```python
            # Clear specific profile
            manager.clear_history(SeedProfile.EXPERIMENTAL)
            
            # Clear all profiles
            manager.clear_history()
            ```
        """
        # Clear specified profile or all profiles
        if profile:
            self.used_seeds[profile.value] = []
        else:
            self.used_seeds = {profile.value: [] for profile in SeedProfile}

        # Archive existing history file
        if self.seed_history_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = self.seed_history_file.with_suffix(f".{timestamp}.bak")
            self.seed_history_file.rename(archive_path)

    def get_seed_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get comprehensive seed usage statistics.
        
        Returns:
            Dict[str, Dict[str, int]]: Statistics by profile including:
                - Total seeds used
                - Unique seeds used
                - Minimum seed value
                - Maximum seed value
        
        Example:
            ```python
            stats = manager.get_seed_statistics()
            for profile, profile_stats in stats.items():
                print(f"{profile} stats:")
                for stat, value in profile_stats.items():
                    print(f"  {stat}: {value}")
            ```
        """
        stats = {}
        for profile in SeedProfile:
            seeds = self.used_seeds[profile.value]
            stats[profile.value] = {
                "total_used": len(seeds),
                "unique_seeds": len(set(seeds)),
                "min_seed": min(seeds) if seeds else 0,
                "max_seed": max(seeds) if seeds else 0,
            }
        return stats
