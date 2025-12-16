#!/usr/bin/env python3
"""Standalone nervaluate benchmark runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run nervaluate benchmark")
    parser.add_argument("--samples", "-n", type=int, default=500, help="Number of samples")
    parser.add_argument("--quick", action="store_true", help="Quick mode (100 samples)")
    args = parser.parse_args()
    
    if args.quick:
        args.samples = 100
    
    print("Initializing detection engine...")
    
    from privplay.engine.classifier import ClassificationEngine
    from privplay.benchmark.nervaluate_runner import NervaluateBenchmarkRunner
    
    engine = ClassificationEngine(
        use_meta_classifier=True,
        use_coreference=True,
        capture_signals=False,
    )
    
    print(f"  Meta-classifier: {engine._meta_classifier is not None}")
    print(f"  Coreference: {engine._coref_resolver is not None}")
    
    runner = NervaluateBenchmarkRunner(engine=engine)
    result = runner.run(dataset_name="ai4privacy", n_samples=args.samples)
    
    return result

if __name__ == "__main__":
    main()