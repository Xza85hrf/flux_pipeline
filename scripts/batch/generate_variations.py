#!/usr/bin/env python3
"""Generate multiple variations of a single prompt.

This script generates N variations of the same prompt using different seeds.
Useful for exploring different interpretations of a concept.

Usage:
    python scripts/batch/generate_variations.py --prompt "A serene landscape" --count 10

    python scripts/batch/generate_variations.py \
        --prompt "A futuristic city" \
        --count 20 \
        --profile creative \
        --size 1024x1024 \
        --output ./variations/
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline import FluxPipeline
from core import SeedProfile
from config import setup_environment, logger
from utils import setup_workspace, PerformanceMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple variations of a single prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt to generate variations for"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of variations to generate (default: 10)"
    )

    parser.add_argument(
        "--profile",
        type=str,
        choices=["conservative", "balanced", "creative"],
        default="balanced",
        help="Seed profile to use (default: balanced)"
    )

    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Image size in format WIDTHxHEIGHT (default: 1024x1024)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./workspace/variations_TIMESTAMP)"
    )

    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save metadata JSON file with generation info"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Parse size
    try:
        width, height = map(int, args.size.split('x'))
    except ValueError:
        logger.error(f"Invalid size format: {args.size}. Use WIDTHxHEIGHT (e.g., 1024x1024)")
        return 1

    # Setup
    logger.info("Initializing FluxPipeline...")
    setup_environment()
    workspace = setup_workspace()

    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = workspace / f"variations_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize pipeline
    pipeline = FluxPipeline(workspace=workspace)

    if not pipeline.load_model():
        logger.error("Failed to load model")
        return 1

    # Parse seed profile
    profile_map = {
        "conservative": SeedProfile.CONSERVATIVE,
        "balanced": SeedProfile.BALANCED,
        "creative": SeedProfile.CREATIVE,
    }
    seed_profile = profile_map[args.profile]

    # Initialize metrics
    metrics = PerformanceMetrics()

    # Generate variations
    logger.info(f"Generating {args.count} variations of: '{args.prompt}'")
    logger.info(f"Profile: {args.profile}, Size: {width}x{height}, Steps: {args.steps}")

    results = []

    for i in range(1, args.count + 1):
        logger.info(f"\n[{i}/{args.count}] Generating variation {i}...")

        with metrics.measure(f"variation_{i}", "generation"):
            image, seed = pipeline.generate_image(
                prompt=args.prompt,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                seed_profile=seed_profile
            )

        if image:
            # Save image
            filename = f"variation_{i:03d}_seed_{seed}.png"
            output_path = output_dir / filename
            image.save(output_path)

            results.append({
                "variation": i,
                "seed": seed,
                "filename": filename,
                "path": str(output_path)
            })

            logger.info(f"  ✅ Saved: {filename} (seed: {seed})")
        else:
            logger.error(f"  ❌ Failed to generate variation {i}")

    # Save metadata
    if args.save_metadata:
        metadata = {
            "prompt": args.prompt,
            "total_variations": args.count,
            "successful": len(results),
            "failed": args.count - len(results),
            "profile": args.profile,
            "size": {"width": width, "height": height},
            "steps": args.steps,
            "timestamp": datetime.now().isoformat(),
            "variations": results
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"\n✅ Metadata saved to: {metadata_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Total variations: {args.count}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Failed: {args.count - len(results)}")
    logger.info(f"Output directory: {output_dir}")

    # Performance report
    stats = metrics.get_operation_stats(category="generation")
    if stats['count'] > 0:
        logger.info(f"\nPerformance:")
        logger.info(f"  Average time: {stats['avg_time_seconds']:.2f}s per image")
        logger.info(f"  Total time: {stats['total_time_seconds']:.2f}s")
        logger.info(f"  Fastest: {stats['min_time_seconds']:.2f}s")
        logger.info(f"  Slowest: {stats['max_time_seconds']:.2f}s")

    logger.info(f"{'='*60}\n")

    return 0 if len(results) == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
