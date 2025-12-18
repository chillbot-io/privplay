#!/usr/bin/env python3
"""
Step 2: Fit Platt scaling models for BERT calibration.

Takes calibration_data.json from Step 1 and fits logistic regression
models to convert raw BERT scores to calibrated probabilities.

Output: calibration_models.json with coefficients for:
- PHI-BERT: calibrated = sigmoid(a * raw + b)
- PII-BERT: calibrated = sigmoid(a * raw + b)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def fit_platt_scaling(raw_scores, actuals):
    """
    Fit Platt scaling using logistic regression.
    
    Returns (a, b) coefficients for: calibrated = sigmoid(a * raw + b)
    """
    from sklearn.linear_model import LogisticRegression
    
    X = np.array(raw_scores).reshape(-1, 1)
    y = np.array(actuals)
    
    # Fit logistic regression
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    
    # Extract coefficients
    a = model.coef_[0][0]
    b = model.intercept_[0]
    
    return a, b, model


def evaluate_calibration(raw_scores, actuals, a, b, name="Model"):
    """Evaluate calibration quality."""
    
    calibrated = sigmoid(a * np.array(raw_scores) + b)
    
    # Brier score (lower is better)
    brier = np.mean((calibrated - actuals) ** 2)
    
    # Expected Calibration Error (ECE)
    # Bin predictions and compare average predicted vs actual
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        in_bin = (calibrated >= bin_boundaries[i]) & (calibrated < bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            avg_pred = np.mean(calibrated[in_bin])
            avg_actual = np.mean(np.array(actuals)[in_bin])
            ece += np.abs(avg_pred - avg_actual) * np.sum(in_bin) / len(actuals)
    
    console.print(f"\n[bold]{name} Calibration:[/bold]")
    console.print(f"  Brier Score: {brier:.4f} (lower is better)")
    console.print(f"  ECE: {ece:.4f} (lower is better)")
    
    # Show calibration by bucket
    console.print(f"\n  Calibration by confidence bucket:")
    for low, high in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        mask = (np.array(raw_scores) >= low) & (np.array(raw_scores) < high)
        if np.sum(mask) > 0:
            raw_in_bucket = np.array(raw_scores)[mask]
            cal_in_bucket = calibrated[mask]
            actual_in_bucket = np.array(actuals)[mask]
            
            avg_raw = np.mean(raw_in_bucket)
            avg_cal = np.mean(cal_in_bucket)
            actual_rate = np.mean(actual_in_bucket)
            
            console.print(f"    Raw {low:.1f}-{high:.1f}: n={np.sum(mask)}, raw_avg={avg_raw:.2f}, cal_avg={avg_cal:.2f}, actual={actual_rate:.2f}")
    
    return brier, ece


def fit_calibration_models(input_path: str = "calibration_data.json", output_path: str = "calibration_models.json"):
    """Fit and save Platt scaling models."""
    
    console.print(f"\n[bold cyan]═══ Platt Scaling Calibration ═══[/bold cyan]\n")
    
    # Load calibration data
    console.print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)
    
    results = {
        "phi_bert": None,
        "pii_bert": None,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source_file": input_path,
        }
    }
    
    # Fit PHI-BERT calibration
    if data["phi_bert"]:
        console.print(f"\n[bold]Fitting PHI-BERT calibration ({len(data['phi_bert'])} signals)...[/bold]")
        
        raw_scores = [d["raw_score"] for d in data["phi_bert"]]
        actuals = [d["actual"] for d in data["phi_bert"]]
        
        # Check we have both classes
        if len(set(actuals)) < 2:
            console.print("[yellow]  Warning: Only one class present, cannot fit calibration[/yellow]")
        else:
            a, b, model = fit_platt_scaling(raw_scores, actuals)
            
            console.print(f"  Coefficients: a={a:.4f}, b={b:.4f}")
            console.print(f"  Formula: calibrated = sigmoid({a:.4f} * raw + {b:.4f})")
            
            # Evaluate
            brier, ece = evaluate_calibration(raw_scores, actuals, a, b, "PHI-BERT")
            
            results["phi_bert"] = {
                "a": a,
                "b": b,
                "n_samples": len(raw_scores),
                "brier_score": brier,
                "ece": ece,
            }
    
    # Fit PII-BERT calibration
    if data["pii_bert"]:
        console.print(f"\n[bold]Fitting PII-BERT calibration ({len(data['pii_bert'])} signals)...[/bold]")
        
        raw_scores = [d["raw_score"] for d in data["pii_bert"]]
        actuals = [d["actual"] for d in data["pii_bert"]]
        
        if len(set(actuals)) < 2:
            console.print("[yellow]  Warning: Only one class present, cannot fit calibration[/yellow]")
        else:
            a, b, model = fit_platt_scaling(raw_scores, actuals)
            
            console.print(f"  Coefficients: a={a:.4f}, b={b:.4f}")
            console.print(f"  Formula: calibrated = sigmoid({a:.4f} * raw + {b:.4f})")
            
            # Evaluate
            brier, ece = evaluate_calibration(raw_scores, actuals, a, b, "PII-BERT")
            
            results["pii_bert"] = {
                "a": a,
                "b": b,
                "n_samples": len(raw_scores),
                "brier_score": brier,
                "ece": ece,
            }
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]✓ Saved calibration models to {output_path}[/green]")
    
    # Show example conversions
    console.print(f"\n[bold]Example Conversions:[/bold]")
    table = Table(box=box.SIMPLE)
    table.add_column("Raw Score")
    table.add_column("PHI-BERT Calibrated")
    table.add_column("PII-BERT Calibrated")
    
    for raw in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        phi_cal = "-"
        pii_cal = "-"
        
        if results["phi_bert"]:
            phi_cal = f"{sigmoid(results['phi_bert']['a'] * raw + results['phi_bert']['b']):.2%}"
        if results["pii_bert"]:
            pii_cal = f"{sigmoid(results['pii_bert']['a'] * raw + results['pii_bert']['b']):.2%}"
        
        table.add_row(f"{raw:.0%}", phi_cal, pii_cal)
    
    console.print(table)
    console.print()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fit Platt scaling for BERT calibration")
    parser.add_argument("--input", type=str, default="calibration_data.json", help="Input calibration data")
    parser.add_argument("--output", type=str, default="calibration_models.json", help="Output model file")
    
    args = parser.parse_args()
    
    fit_calibration_models(input_path=args.input, output_path=args.output)
