#!/usr/bin/env python3
"""
Study 2 Pipeline Runner

Orchestrates the full data collection and analysis pipeline:
1. Collect responses (test_specification_levels.py)
2. Code misconceptions (code_misconceptions_llm.py)
3. Analyze results (analyze_study2.py)

Usage:
    # Full experiment (all tiers, 3 reps)
    python run_study2_pipeline.py --full

    # Quick pilot (mid tier, 1 rep)
    python run_study2_pipeline.py --pilot

    # Custom
    python run_study2_pipeline.py --tier mid --reps 2 --specs S1 S4
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


SCRIPTS_DIR = Path(__file__).parent
OUTPUT_BASE = Path("/Users/dereklomas/AIED/study2-materials/results")


def run_command(cmd: list, description: str) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def run_pipeline(
    tier: str = "mid",
    specs: list = None,
    reps: int = 1,
    skip_collection: bool = False,
    skip_coding: bool = False,
    skip_analysis: bool = False,
    output_name: str = None
):
    """Run the full pipeline."""

    if specs is None:
        specs = ["S1", "S2", "S3", "S4"]

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = output_name or f"{tier}_{len(specs)}specs_{reps}reps"
    output_dir = OUTPUT_BASE / f"{timestamp}_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"STUDY 2 PIPELINE")
    print(f"{'#'*60}")
    print(f"Tier: {tier}")
    print(f"Specs: {specs}")
    print(f"Reps: {reps}")
    print(f"Output: {output_dir}")
    print()

    responses_file = output_dir / "responses.jsonl"
    coded_file = output_dir / "coding_results" / "coded_errors.jsonl"

    # Step 1: Collect responses
    if not skip_collection:
        collect_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "test_specification_levels.py"),
            "--tier", tier,
            "--specs", *specs,
            "--reps", str(reps),
            "--output", str(output_dir)
        ]

        if not run_command(collect_cmd, "Data Collection"):
            return False

        # The collection script saves to spec_level_results.jsonl
        # Rename for consistency
        orig_file = output_dir / "spec_level_results.jsonl"
        if orig_file.exists():
            orig_file.rename(responses_file)
    else:
        print("\nSkipping collection (--skip-collection)")

    # Step 2: Code misconceptions
    if not skip_coding:
        if not responses_file.exists():
            print(f"ERROR: Responses file not found: {responses_file}")
            return False

        code_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "code_misconceptions_llm.py"),
            "--input", str(responses_file),
            "--output", str(output_dir / "coding_results")
        ]

        if not run_command(code_cmd, "Misconception Coding"):
            return False
    else:
        print("\nSkipping coding (--skip-coding)")

    # Step 3: Analyze results
    if not skip_analysis:
        if not coded_file.exists():
            print(f"ERROR: Coded file not found: {coded_file}")
            return False

        analyze_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "analyze_study2.py"),
            "--input", str(coded_file),
            "--output", str(output_dir / "analysis")
        ]

        if not run_command(analyze_cmd, "Statistical Analysis"):
            return False
    else:
        print("\nSkipping analysis (--skip-analysis)")

    # Summary
    print(f"\n{'#'*60}")
    print("PIPELINE COMPLETE")
    print(f"{'#'*60}")
    print(f"\nResults saved to: {output_dir}")
    print("\nKey files:")
    print(f"  - Responses: {responses_file}")
    print(f"  - Coded errors: {coded_file}")
    print(f"  - Analysis: {output_dir / 'analysis'}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run Study 2 pipeline")

    # Preset configurations
    parser.add_argument("--full", action="store_true",
                       help="Full experiment: all tiers, all specs, 3 reps")
    parser.add_argument("--pilot", action="store_true",
                       help="Quick pilot: mid tier, all specs, 1 rep")

    # Custom configuration
    parser.add_argument("--tier", type=str, default="mid",
                       choices=["frontier", "mid", "weak", "all"],
                       help="Model tier(s) to test")
    parser.add_argument("--specs", nargs="+", default=["S1", "S2", "S3", "S4"],
                       help="Specification levels to test")
    parser.add_argument("--reps", type=int, default=1,
                       help="Number of repetitions")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom name for output directory")

    # Skip stages
    parser.add_argument("--skip-collection", action="store_true",
                       help="Skip data collection (use existing responses)")
    parser.add_argument("--skip-coding", action="store_true",
                       help="Skip misconception coding")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip statistical analysis")

    args = parser.parse_args()

    # Apply presets
    if args.full:
        tier = "all"
        specs = ["S1", "S2", "S3", "S4"]
        reps = 3
        output_name = "full_experiment"
    elif args.pilot:
        tier = "mid"
        specs = ["S1", "S2", "S3", "S4"]
        reps = 1
        output_name = "pilot"
    else:
        tier = args.tier
        specs = args.specs
        reps = args.reps
        output_name = args.output_name

    # Calculate expected API calls
    tier_models = {
        "frontier": 2,
        "mid": 2,
        "weak": 2,
        "all": 6
    }
    n_models = tier_models.get(tier, 2)
    n_items = 6  # 5 near-transfer + 1 discriminant per misconception
    n_misc = 4
    total_collection = n_models * len(specs) * n_misc * n_items * reps

    print(f"\nExpected API calls:")
    print(f"  Collection: ~{total_collection} calls")
    print(f"  Coding: ~{total_collection // 2} calls (errors only, estimated 50% error rate)")

    confirm = input("\nProceed? [y/N]: ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    success = run_pipeline(
        tier=tier,
        specs=specs,
        reps=reps,
        skip_collection=args.skip_collection,
        skip_coding=args.skip_coding,
        skip_analysis=args.skip_analysis,
        output_name=output_name
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
