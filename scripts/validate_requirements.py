#!/usr/bin/env python3
"""Validate that all requirements files are consistent.

This script checks that variant requirements files (CUDA, ROCm, Intel) contain
all the common packages from the base requirements.txt file.

Usage:
    python scripts/validate_requirements.py

Exit Codes:
    0: All requirements files are consistent
    1: Inconsistencies found
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_requirements(file_path: Path) -> Dict[str, str]:
    """Parse a requirements file and return package:version mapping.

    Args:
        file_path: Path to requirements file

    Returns:
        Dictionary mapping package names to version specifications
    """
    packages = {}

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return packages

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments, empty lines, and pip options
            if not line or line.startswith('#') or line.startswith('--'):
                continue

            # Handle package==version format
            if '==' in line:
                pkg, version = line.split('==', 1)
                pkg = pkg.strip().lower()
                version = version.strip()

                # Skip platform-specific packages
                if '+' in version:  # e.g., torch==2.5.1+cu124
                    base_version = version.split('+')[0]
                    packages[pkg] = base_version
                else:
                    packages[pkg] = version
            else:
                # Package without version pin
                pkg = line.strip().lower()
                packages[pkg] = ""

    return packages


def get_common_packages(base_packages: Dict[str, str]) -> Set[str]:
    """Extract common packages that should be in all variants.

    Args:
        base_packages: Package dictionary from base requirements.txt

    Returns:
        Set of package names that should be in all variants
    """
    # Exclude torch variants as they're platform-specific
    exclude_prefixes = ['torch', 'cuda-python', 'xformers']

    common = set()
    for pkg in base_packages.keys():
        if not any(pkg.startswith(prefix) for prefix in exclude_prefixes):
            common.add(pkg)

    return common


def validate_variant(
    variant_name: str,
    variant_packages: Dict[str, str],
    common_packages: Set[str],
    base_packages: Dict[str, str]
) -> List[str]:
    """Validate a variant requirements file.

    Args:
        variant_name: Name of the variant (e.g., "CUDA", "ROCm")
        variant_packages: Packages from variant file
        common_packages: Set of common packages that should exist
        base_packages: Base packages with versions

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check for missing packages
    variant_pkg_names = set(variant_packages.keys())
    missing = common_packages - variant_pkg_names

    if missing:
        errors.append(
            f"  ❌ {variant_name} missing packages: {', '.join(sorted(missing))}"
        )

    # Check version mismatches
    mismatches = []
    for pkg in common_packages & variant_pkg_names:
        base_ver = base_packages.get(pkg, "")
        variant_ver = variant_packages.get(pkg, "")

        if base_ver and variant_ver and base_ver != variant_ver:
            mismatches.append(
                f"    - {pkg}: base={base_ver}, {variant_name}={variant_ver}"
            )

    if mismatches:
        errors.append(f"  ⚠️  {variant_name} version mismatches:")
        errors.extend(mismatches)

    return errors


def main() -> int:
    """Main validation function.

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    print("🔍 Validating requirements files...\n")

    # Define file paths
    base_dir = Path(__file__).parent.parent
    base_file = base_dir / "requirements.txt"
    variant_files = {
        "CPU": base_dir / "requirements_cpu.txt",
        "CUDA": base_dir / "requirements_cuda.txt",
        "ROCm": base_dir / "requirements_rocm.txt",
        "Intel": base_dir / "requirements_intel.txt",
    }

    # Check base file exists
    if not base_file.exists():
        print(f"❌ Base requirements file not found: {base_file}")
        return 1

    # Parse base requirements
    print(f"📄 Reading base requirements from: {base_file.name}")
    base_packages = parse_requirements(base_file)
    common_packages = get_common_packages(base_packages)

    print(f"   Found {len(base_packages)} total packages")
    print(f"   {len(common_packages)} common packages to validate\n")

    # Validate each variant
    all_errors = []
    for variant_name, variant_file in variant_files.items():
        print(f"📦 Checking {variant_name}: {variant_file.name}")

        if not variant_file.exists():
            print(f"  ⚠️  File not found, skipping...")
            continue

        variant_packages = parse_requirements(variant_file)
        errors = validate_variant(
            variant_name, variant_packages, common_packages, base_packages
        )

        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(error)
        else:
            print(f"  ✅ All common packages present and versions match")

        print()

    # Summary
    if all_errors:
        print(f"❌ Validation failed with {len(all_errors)} issue(s)\n")
        print("To fix:")
        print("  1. Add missing packages to variant files")
        print("  2. Update version mismatches")
        print("  3. Re-run validation")
        return 1
    else:
        print("✅ All requirements files are consistent!\n")
        print(f"Validated files:")
        for variant_name, variant_file in variant_files.items():
            if variant_file.exists():
                print(f"  - {variant_file.name} ({variant_name})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
