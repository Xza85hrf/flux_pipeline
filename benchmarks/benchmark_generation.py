#!/usr/bin/env python3
"""Benchmark image generation performance.

This script benchmarks FluxPipeline performance across different:
- Image sizes
- Inference steps
- Batch sizes
- GPU configurations

Results are saved to JSON and optionally displayed as tables.

Usage:
    python benchmarks/benchmark_generation.py

    python benchmarks/benchmark_generation.py --sizes 512 768 1024

    python benchmarks/benchmark_generation.py --output benchmark_results.json
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pipeline import FluxPipeline
from core import SeedProfile
from config import setup_environment, logger
from utils import setup_workspace, PerformanceMetrics


class GenerationBenchmark:
    """Benchmark suite for image generation performance."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.pipeline = None
        self.workspace = None
        self.metrics = PerformanceMetrics()
        self.results: List[Dict[str, Any]] = []

    def setup(self):
        """Setup pipeline for benchmarking."""
        logger.info("Setting up benchmark environment...")
        setup_environment()
        self.workspace = setup_workspace()

        logger.info("Loading model...")
        self.pipeline = FluxPipeline(workspace=self.workspace)

        if not self.pipeline.load_model():
            raise RuntimeError("Failed to load model")

        logger.info("✅ Setup complete")

    def cleanup_memory(self):
        """Clean up GPU memory between tests."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def benchmark_size(self, size: int, num_runs: int = 3) -> Dict[str, Any]:
        """Benchmark generation at specific size.

        Args:
            size: Image dimension (square)
            num_runs: Number of runs to average

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"\nBenchmarking {size}x{size}...")

        times = []
        memory_deltas = []
        prompt = "A test image for benchmarking performance"

        for run in range(num_runs):
            self.cleanup_memory()

            # Measure generation
            start_time = time.time()

            if TORCH_AVAILABLE and torch.cuda.is_available():
                start_mem = torch.cuda.memory_allocated() / 1024**3
            else:
                start_mem = 0

            image, seed = self.pipeline.generate_image(
                prompt=prompt,
                height=size,
                width=size,
                num_inference_steps=4,
                seed_profile=SeedProfile.BALANCED
            )

            elapsed = time.time() - start_time

            if TORCH_AVAILABLE and torch.cuda.is_available():
                end_mem = torch.cuda.memory_allocated() / 1024**3
                mem_delta = end_mem - start_mem
            else:
                mem_delta = 0

            if image:
                times.append(elapsed)
                memory_deltas.append(mem_delta)
                logger.info(f"  Run {run + 1}: {elapsed:.2f}s, {mem_delta:.2f}GB")
            else:
                logger.error(f"  Run {run + 1}: Failed")

        if not times:
            return {"size": size, "error": "All runs failed"}

        result = {
            "size": size,
            "num_runs": len(times),
            "avg_time_seconds": sum(times) / len(times),
            "min_time_seconds": min(times),
            "max_time_seconds": max(times),
            "avg_memory_gb": sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            "images_per_second": 1 / (sum(times) / len(times)),
        }

        logger.info(f"  Average: {result['avg_time_seconds']:.2f}s")
        return result

    def benchmark_steps(self, steps_list: List[int], size: int = 512) -> List[Dict[str, Any]]:
        """Benchmark different inference step counts.

        Args:
            steps_list: List of step counts to test
            size: Image size to use

        Returns:
            List of benchmark results
        """
        logger.info(f"\nBenchmarking inference steps at {size}x{size}...")

        results = []
        prompt = "A test image for benchmarking inference steps"

        for steps in steps_list:
            logger.info(f"\nTesting {steps} steps...")
            self.cleanup_memory()

            start_time = time.time()

            image, seed = self.pipeline.generate_image(
                prompt=prompt,
                height=size,
                width=size,
                num_inference_steps=steps,
                seed_profile=SeedProfile.BALANCED
            )

            elapsed = time.time() - start_time

            if image:
                result = {
                    "steps": steps,
                    "time_seconds": elapsed,
                    "time_per_step": elapsed / steps,
                }
                results.append(result)
                logger.info(f"  {steps} steps: {elapsed:.2f}s ({elapsed/steps:.3f}s/step)")
            else:
                logger.error(f"  {steps} steps: Failed")

        return results

    def benchmark_batch(self, batch_sizes: List[int], size: int = 512) -> List[Dict[str, Any]]:
        """Benchmark batch generation.

        Args:
            batch_sizes: List of batch sizes to test
            size: Image size to use

        Returns:
            List of benchmark results
        """
        logger.info(f"\nBenchmarking batch sizes at {size}x{size}...")

        results = []
        prompts = [f"Test image {i}" for i in range(max(batch_sizes))]

        for batch_size in batch_sizes:
            logger.info(f"\nTesting batch size {batch_size}...")
            self.cleanup_memory()

            start_time = time.time()

            for i in range(batch_size):
                image, seed = self.pipeline.generate_image(
                    prompt=prompts[i],
                    height=size,
                    width=size,
                    num_inference_steps=4,
                    seed_profile=SeedProfile.BALANCED
                )

                if not image:
                    logger.error(f"  Image {i+1}/{batch_size}: Failed")
                    break

            elapsed = time.time() - start_time

            result = {
                "batch_size": batch_size,
                "total_time_seconds": elapsed,
                "time_per_image": elapsed / batch_size,
                "images_per_second": batch_size / elapsed,
            }
            results.append(result)
            logger.info(f"  Batch {batch_size}: {elapsed:.2f}s ({elapsed/batch_size:.2f}s/img)")

        return results

    def run_all_benchmarks(self, args) -> Dict[str, Any]:
        """Run all benchmarks.

        Args:
            args: Command line arguments

        Returns:
            Complete benchmark results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.get_system_info(),
            "benchmarks": {}
        }

        # Benchmark sizes
        if args.benchmark_sizes:
            size_results = []
            for size in args.sizes:
                result = self.benchmark_size(size, args.runs_per_size)
                size_results.append(result)
            results["benchmarks"]["sizes"] = size_results

        # Benchmark steps
        if args.benchmark_steps:
            steps_results = self.benchmark_steps(args.steps, args.steps_size)
            results["benchmarks"]["steps"] = steps_results

        # Benchmark batch
        if args.benchmark_batch:
            batch_results = self.benchmark_batch(args.batch_sizes, args.batch_size_pixels)
            results["benchmarks"]["batch"] = batch_results

        return results

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarks.

        Returns:
            System information dictionary
        """
        info = {
            "python_version": sys.version.split()[0],
        }

        if TORCH_AVAILABLE:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()

            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return info

    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way.

        Args:
            results: Benchmark results dictionary
        """
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)

        # System info
        print("\nSystem Information:")
        for key, value in results["system_info"].items():
            print(f"  {key}: {value}")

        # Size benchmarks
        if "sizes" in results["benchmarks"]:
            print("\nImage Size Benchmarks:")
            print(f"{'Size':<12} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'Img/s':<10}")
            print("-" * 60)
            for result in results["benchmarks"]["sizes"]:
                if "error" not in result:
                    print(f"{result['size']}x{result['size']:<6} "
                          f"{result['avg_time_seconds']:<12.2f} "
                          f"{result['min_time_seconds']:<12.2f} "
                          f"{result['max_time_seconds']:<12.2f} "
                          f"{result['images_per_second']:<10.3f}")

        # Steps benchmarks
        if "steps" in results["benchmarks"]:
            print("\nInference Steps Benchmarks:")
            print(f"{'Steps':<10} {'Time (s)':<12} {'Time/Step':<12}")
            print("-" * 40)
            for result in results["benchmarks"]["steps"]:
                print(f"{result['steps']:<10} "
                      f"{result['time_seconds']:<12.2f} "
                      f"{result['time_per_step']:<12.3f}")

        # Batch benchmarks
        if "batch" in results["benchmarks"]:
            print("\nBatch Size Benchmarks:")
            print(f"{'Batch':<10} {'Total (s)':<12} {'Per Image':<12} {'Img/s':<10}")
            print("-" * 50)
            for result in results["benchmarks"]["batch"]:
                print(f"{result['batch_size']:<10} "
                      f"{result['total_time_seconds']:<12.2f} "
                      f"{result['time_per_image']:<12.2f} "
                      f"{result['images_per_second']:<10.3f}")

        print("\n" + "="*60 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark FluxPipeline performance")

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[512, 768, 1024],
        help="Image sizes to benchmark (default: 512 768 1024)"
    )

    parser.add_argument(
        "--runs-per-size",
        type=int,
        default=3,
        help="Number of runs per size (default: 3)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Inference steps to benchmark (default: 1 2 4 8)"
    )

    parser.add_argument(
        "--steps-size",
        type=int,
        default=512,
        help="Image size for steps benchmark (default: 512)"
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Batch sizes to benchmark (default: 1 5 10)"
    )

    parser.add_argument(
        "--batch-size-pixels",
        type=int,
        default=512,
        help="Image size for batch benchmark (default: 512)"
    )

    parser.add_argument(
        "--benchmark-sizes",
        action="store_true",
        default=True,
        help="Run size benchmarks"
    )

    parser.add_argument(
        "--benchmark-steps",
        action="store_true",
        help="Run inference steps benchmarks"
    )

    parser.add_argument(
        "--benchmark-batch",
        action="store_true",
        help="Run batch size benchmarks"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Initialize benchmark
    benchmark = GenerationBenchmark()

    try:
        benchmark.setup()

        # Run benchmarks
        results = benchmark.run_all_benchmarks(args)

        # Print results
        benchmark.print_results(results)

        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✅ Results saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
