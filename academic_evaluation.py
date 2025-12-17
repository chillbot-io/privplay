#!/usr/bin/env python3
"""
Academic-Grade NER Evaluation Script

Uses nervaluate (SemEval 2013 Task 9 methodology) for rigorous entity-level evaluation.

Evaluation Modes:
- Strict: Exact boundary + exact type match (most conservative)
- Exact: Exact boundary, type ignored
- Partial: Overlap gets 0.5 credit, type must match
- Type (ent_type): Type must match, boundaries can overlap

Outputs:
- Raw JSON with all metrics
- Formatted markdown report
- Per-entity-type breakdown
- Error analysis

Usage:
    python academic_evaluation.py [--samples N] [--output-dir DIR]

Requirements:
    pip install nervaluate datasets
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Add privplay to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TYPE MAPPING (AI4Privacy → Privplay canonical types)
# =============================================================================

# Map AI4Privacy labels to canonical categories for evaluation
TYPE_CATEGORY_MAP = {
    # NAME category
    "FIRSTNAME": "NAME", "LASTNAME": "NAME", "MIDDLENAME": "NAME",
    "PREFIX": "NAME", "SUFFIX": "NAME", "NAME_PERSON": "NAME",
    "NAME_PATIENT": "NAME", "NAME_PROVIDER": "NAME", "FULLNAME": "NAME",
    
    # CONTACT category
    "EMAIL": "CONTACT", "PHONENUMBER": "CONTACT", "PHONE": "CONTACT",
    "FAX": "CONTACT", "URL": "CONTACT", "TELEPHONENUM": "CONTACT",
    
    # LOCATION category  
    "ADDRESS": "LOCATION", "STREET": "LOCATION", "CITY": "LOCATION",
    "STATE": "LOCATION", "COUNTY": "LOCATION", "COUNTRY": "LOCATION",
    "ZIPCODE": "LOCATION", "ZIP": "LOCATION", "BUILDINGNUMBER": "LOCATION",
    "SECONDARYADDRESS": "LOCATION", "NEARBYGPSCOORDINATE": "LOCATION",
    "GPS_COORDINATE": "LOCATION", "POSTCODE": "LOCATION",
    
    # DATE category
    "DATE": "DATE", "DOB": "DATE", "DATE_DOB": "DATE", "TIME": "DATE",
    "DATEOFBIRTH": "DATE", "BIRTHDAY": "DATE",
    
    # IDNUM category (government/account IDs)
    "SSN": "IDNUM", "PASSPORTNUMBER": "IDNUM", "PASSPORT": "IDNUM",
    "DRIVERLICENSE": "IDNUM", "DRIVER_LICENSE": "IDNUM",
    "ACCOUNTNUMBER": "IDNUM", "ACCOUNT_NUMBER": "IDNUM",
    "IBAN": "IDNUM", "BIC": "IDNUM", "MRN": "IDNUM",
    "TAXID": "IDNUM", "NATIONALID": "IDNUM",
    
    # FINANCIAL category
    "CREDITCARDNUMBER": "FINANCIAL", "CREDIT_CARD": "FINANCIAL",
    "CREDITCARDCVV": "FINANCIAL", "PIN": "FINANCIAL",
    "BANK_ACCOUNT": "FINANCIAL", "AMOUNT": "FINANCIAL",
    
    # DIGITAL category
    "IP": "DIGITAL", "IPV4": "DIGITAL", "IPV6": "DIGITAL", "IP_ADDRESS": "DIGITAL",
    "MAC": "DIGITAL", "MACADDRESS": "DIGITAL", "MAC_ADDRESS": "DIGITAL",
    "USERNAME": "DIGITAL", "USERAGENT": "DIGITAL", "USER_AGENT": "DIGITAL",
    "PASSWORD": "DIGITAL", "USERID": "DIGITAL",
    
    # CRYPTO category
    "BITCOINADDRESS": "CRYPTO", "ETHEREUMADDRESS": "CRYPTO",
    "LITECOINADDRESS": "CRYPTO", "CRYPTO_ADDRESS": "CRYPTO",
    
    # VEHICLE category
    "VEHICLEVIN": "VEHICLE", "VEHICLEIDENTIFICATIONNUMBER": "VEHICLE",
    "VIN": "VEHICLE", "DEVICE_ID": "VEHICLE", "IMEI": "VEHICLE",
    "LICENSEPLATE": "VEHICLE", "VEHICLEVRM": "VEHICLE",
    
    # ORG category
    "COMPANYNAME": "ORG", "FACILITY": "ORG", "HOSPITAL": "ORG",
    "ORGANIZATION": "ORG", "EMPLOYER": "ORG",
    
    # OTHER
    "AGE": "OTHER", "GENDER": "OTHER", "OTHER": "OTHER",
    "JOBTITLE": "OTHER", "JOBAREA": "OTHER", "JOBTYPE": "OTHER",
    "CURRENCY": "OTHER", "CURRENCYCODE": "OTHER",
}


def normalize_type(entity_type: str) -> str:
    """Normalize entity type to canonical category."""
    if not entity_type:
        return "OTHER"
    
    # Normalize input: uppercase, strip whitespace and punctuation
    t = entity_type.upper().strip().replace("_", "").replace("-", "").replace(" ", "")
    
    # Direct lookup (original form)
    upper = entity_type.upper().strip()
    if upper in TYPE_CATEGORY_MAP:
        return TYPE_CATEGORY_MAP[upper]
    
    # Try normalized form against normalized keys
    for key, val in TYPE_CATEGORY_MAP.items():
        key_normalized = key.replace("_", "").replace("-", "").replace(" ", "")
        if t == key_normalized:
            return val
    
    # Substring matching for common patterns
    if "NAME" in t or "FIRST" in t or "LAST" in t or "MIDDLE" in t:
        return "NAME"
    if "EMAIL" in t or "PHONE" in t or "FAX" in t or "URL" in t or "TEL" in t:
        return "CONTACT"
    if "ADDRESS" in t or "STREET" in t or "CITY" in t or "STATE" in t or "ZIP" in t or "COUNTRY" in t or "GPS" in t or "COORDINATE" in t or "LOCATION" in t:
        return "LOCATION"
    if "DATE" in t or "DOB" in t or "BIRTH" in t or "TIME" in t:
        return "DATE"
    if "SSN" in t or "PASSPORT" in t or "LICENSE" in t or "ACCOUNT" in t or "IBAN" in t or "MRN" in t or "RECORD" in t:
        return "IDNUM"
    if "CREDIT" in t or "CARD" in t or "CVV" in t or "PIN" in t or "BANK" in t:
        return "FINANCIAL"
    if "IP" in t or "MAC" in t or "USER" in t or "PASSWORD" in t or "AGENT" in t:
        return "DIGITAL"
    if "BITCOIN" in t or "ETHEREUM" in t or "LITECOIN" in t or "CRYPTO" in t:
        return "CRYPTO"
    if "VIN" in t or "VEHICLE" in t or "DEVICE" in t or "IMEI" in t or "SERIAL" in t:
        return "VEHICLE"
    if "COMPANY" in t or "ORG" in t or "FACILITY" in t or "HOSPITAL" in t or "EMPLOYER" in t:
        return "ORG"
    
    return "OTHER"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    samples: int = 1000
    dataset: str = "ai4privacy"
    output_dir: str = "./evaluation_results"
    seed: int = 42
    include_type_breakdown: bool = True
    include_error_analysis: bool = True


@dataclass
class EntitySpan:
    """A single entity span for evaluation."""
    label: str
    start: int
    end: int
    text: str = ""


@dataclass 
class SampleResult:
    """Results for a single sample."""
    sample_id: int
    text: str
    ground_truth: List[Dict]
    predictions: List[Dict]
    true_positives: List[Dict]
    false_positives: List[Dict]
    false_negatives: List[Dict]


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all samples."""
    correct: int = 0
    incorrect: int = 0
    partial: int = 0
    missed: int = 0
    spurious: int = 0
    possible: int = 0
    actual: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_ai4privacy_dataset(max_samples: int = 1000, seed: int = 42) -> List[Dict]:
    """Load AI4Privacy dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        sys.exit(1)
    
    logger.info(f"Loading AI4Privacy dataset (max {max_samples} samples with entities)...")
    
    dataset = load_dataset(
        "ai4privacy/pii-masking-400k",
        split="train",
    )
    
    # Shuffle for variety
    dataset = dataset.shuffle(seed=seed)
    
    samples = []
    skipped_empty = 0
    
    for item in dataset:
        if len(samples) >= max_samples:
            break
        
        # Skip samples with no entities
        if not item.get("privacy_mask") or len(item["privacy_mask"]) == 0:
            skipped_empty += 1
            continue
        
        # Extract entities - note: text is in 'value' field
        entities = []
        for mask in item["privacy_mask"]:
            entities.append({
                "label": mask.get("label", "OTHER"),
                "start": mask.get("start", 0),
                "end": mask.get("end", 0),
                "text": mask.get("value", ""),  # AI4Privacy uses 'value' not 'text'
            })
        
        samples.append({
            "id": len(samples),
            "text": item.get("source_text", ""),
            "entities": entities,
        })
    
    logger.info(f"Loaded {len(samples)} samples with entities (skipped {skipped_empty} empty samples)")
    return samples


# =============================================================================
# DETECTION
# =============================================================================

def run_detection(text: str, engine) -> List[Dict]:
    """Run Privplay detection on text."""
    try:
        entities = engine.detect(text, verify=False)
        
        results = []
        for e in entities:
            etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
            results.append({
                "label": normalize_type(etype),
                "start": e.start,
                "end": e.end,
                "text": e.text,
                "confidence": e.confidence,
                "original_type": etype,
            })
        
        return results
    except Exception as ex:
        logger.warning(f"Detection failed: {ex}")
        return []


# =============================================================================
# NERVALUATE INTEGRATION
# =============================================================================

def run_nervaluate(
    ground_truth: List[List[Dict]],
    predictions: List[List[Dict]],
    tags: List[str],
) -> Tuple[Dict, Dict]:
    """
    Run nervaluate evaluation.
    
    Returns:
        (overall_results, results_by_tag)
    """
    try:
        from nervaluate import Evaluator
    except ImportError:
        logger.error("Please install nervaluate: pip install nervaluate")
        sys.exit(1)
    
    # Convert to nervaluate format
    true_entities = []
    pred_entities = []
    
    for gt, pred in zip(ground_truth, predictions):
        true_entities.append([
            {"label": normalize_type(e["label"]), "start": e["start"], "end": e["end"]}
            for e in gt
        ])
        pred_entities.append([
            {"label": e["label"], "start": e["start"], "end": e["end"]}
            for e in pred
        ])
    
    evaluator = Evaluator(true_entities, pred_entities, tags=tags)
    
    # Handle both old (2 values) and new (4 values) nervaluate versions
    eval_result = evaluator.evaluate()
    if len(eval_result) == 4:
        results, results_by_tag, _, _ = eval_result
    else:
        results, results_by_tag = eval_result
    
    return results, results_by_tag


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def analyze_errors(
    samples: List[Dict],
    predictions: List[List[Dict]],
) -> Dict[str, Any]:
    """Analyze false positives and false negatives."""
    
    fp_by_type = defaultdict(list)
    fn_by_type = defaultdict(list)
    type_confusion = defaultdict(lambda: defaultdict(int))
    
    for sample, preds in zip(samples, predictions):
        gt_entities = sample["entities"]
        gt_spans = {(e["start"], e["end"]): e for e in gt_entities}
        pred_spans = {(e["start"], e["end"]): e for e in preds}
        
        # Find FPs (predicted but not in ground truth)
        for span, pred in pred_spans.items():
            if span not in gt_spans:
                # Check for partial overlap
                overlapping = None
                for gt_span, gt_ent in gt_spans.items():
                    if span[0] < gt_span[1] and span[1] > gt_span[0]:
                        overlapping = gt_ent
                        break
                
                if overlapping:
                    # Type confusion
                    gt_type = normalize_type(overlapping["label"])
                    pred_type = pred["label"]
                    if gt_type != pred_type:
                        type_confusion[gt_type][pred_type] += 1
                else:
                    # Pure FP
                    fp_by_type[pred["label"]].append({
                        "text": pred.get("text", ""),
                        "sample_id": sample["id"],
                    })
        
        # Find FNs (in ground truth but not predicted)
        for span, gt in gt_spans.items():
            if span not in pred_spans:
                # Check for partial overlap
                has_overlap = False
                for pred_span in pred_spans:
                    if span[0] < pred_span[1] and span[1] > pred_span[0]:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    gt_type = normalize_type(gt["label"])
                    fn_by_type[gt_type].append({
                        "text": gt.get("text", gt.get("value", "")),
                        "sample_id": sample["id"],
                    })
    
    return {
        "false_positives_by_type": {k: len(v) for k, v in fp_by_type.items()},
        "false_negatives_by_type": {k: len(v) for k, v in fn_by_type.items()},
        "type_confusion_matrix": {k: dict(v) for k, v in type_confusion.items()},
        "top_fp_examples": {k: v[:5] for k, v in fp_by_type.items()},
        "top_fn_examples": {k: v[:5] for k, v in fn_by_type.items()},
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(
    results: Dict,
    results_by_tag: Dict,
    error_analysis: Dict,
    config: EvaluationConfig,
    runtime_seconds: float,
) -> str:
    """Generate formatted markdown evaluation report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Privplay NER Evaluation Report

**Generated:** {timestamp}  
**Methodology:** SemEval 2013 Task 9 (nervaluate)  
**Dataset:** {config.dataset}  
**Samples:** {config.samples:,}  
**Runtime:** {runtime_seconds:.1f}s  

---

## Executive Summary

| Metric | Strict | Exact | Partial | Type |
|--------|--------|-------|---------|------|
| **Precision** | {results['strict']['precision']:.1%} | {results['exact']['precision']:.1%} | {results['partial']['precision']:.1%} | {results['ent_type']['precision']:.1%} |
| **Recall** | {results['strict']['recall']:.1%} | {results['exact']['recall']:.1%} | {results['partial']['recall']:.1%} | {results['ent_type']['recall']:.1%} |
| **F1** | {results['strict']['f1']:.1%} | {results['exact']['f1']:.1%} | {results['partial']['f1']:.1%} | {results['ent_type']['f1']:.1%} |

### Evaluation Mode Definitions

- **Strict**: Exact boundary match AND exact type match (most conservative)
- **Exact**: Exact boundary match, entity type ignored
- **Partial**: Partial boundary overlap (0.5 credit), type must match
- **Type (ent_type)**: Entity type must match, boundaries can partially overlap

---

## Detailed Metrics

### Strict Evaluation (Gold Standard)

| Metric | Value |
|--------|-------|
| Correct | {results['strict']['correct']:,} |
| Incorrect | {results['strict']['incorrect']:,} |
| Partial | {results['strict']['partial']:,} |
| Missed (FN) | {results['strict']['missed']:,} |
| Spurious (FP) | {results['strict']['spurious']:,} |
| Possible (Total GT) | {results['strict']['possible']:,} |
| Actual (Total Pred) | {results['strict']['actual']:,} |

### Counts Summary

| Category | Strict | Exact | Partial | Type |
|----------|--------|-------|---------|------|
| Correct | {results['strict']['correct']:,} | {results['exact']['correct']:,} | {results['partial']['correct']:,} | {results['ent_type']['correct']:,} |
| Incorrect | {results['strict']['incorrect']:,} | {results['exact']['incorrect']:,} | {results['partial']['incorrect']:,} | {results['ent_type']['incorrect']:,} |
| Partial | {results['strict']['partial']:,} | {results['exact']['partial']:,} | {results['partial']['partial']:,} | {results['ent_type']['partial']:,} |
| Missed | {results['strict']['missed']:,} | {results['exact']['missed']:,} | {results['partial']['missed']:,} | {results['ent_type']['missed']:,} |
| Spurious | {results['strict']['spurious']:,} | {results['exact']['spurious']:,} | {results['partial']['spurious']:,} | {results['ent_type']['spurious']:,} |

---

## Per-Entity-Type Performance (Strict)

| Entity Type | Precision | Recall | F1 | Support |
|-------------|-----------|--------|-----|---------|
"""
    
    # Add per-type metrics
    for tag, metrics in sorted(results_by_tag.items()):
        strict = metrics.get('strict', {})
        p = strict.get('precision', 0)
        r = strict.get('recall', 0)
        f1 = compute_f1(p, r)
        support = strict.get('possible', 0)
        report += f"| {tag} | {p:.1%} | {r:.1%} | {f1:.1%} | {support:,} |\n"
    
    report += """
---

## Error Analysis

### False Positives by Type

| Type | Count |
|------|-------|
"""
    
    for etype, count in sorted(error_analysis["false_positives_by_type"].items(), key=lambda x: -x[1]):
        report += f"| {etype} | {count:,} |\n"
    
    report += """
### False Negatives by Type

| Type | Count |
|------|-------|
"""
    
    for etype, count in sorted(error_analysis["false_negatives_by_type"].items(), key=lambda x: -x[1]):
        report += f"| {etype} | {count:,} |\n"
    
    if error_analysis["type_confusion_matrix"]:
        report += """
### Type Confusion Matrix

Shows cases where entity was detected but assigned wrong type.

| Ground Truth | Predicted As | Count |
|--------------|--------------|-------|
"""
        for gt_type, predictions in error_analysis["type_confusion_matrix"].items():
            for pred_type, count in sorted(predictions.items(), key=lambda x: -x[1]):
                report += f"| {gt_type} | {pred_type} | {count:,} |\n"
    
    report += f"""
---

## Methodology Notes

### SemEval 2013 Task 9

This evaluation follows the methodology defined in:

> Segura-Bedmar, I., Martínez, P., & Herrero-Zazo, M. (2013). *SemEval-2013 Task 9: Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)*. In Proceedings of SemEval.

The evaluation considers entity-level (not token-level) matching with four scenarios:
1. **Strict**: Both boundaries and type must match exactly
2. **Exact**: Boundaries must match exactly, type ignored
3. **Partial**: Boundaries can overlap (partial credit), type must match  
4. **Type**: Type must match, boundaries can overlap

### Dataset

**AI4Privacy PII Masking 400K**: A large-scale PII detection benchmark containing
400k+ samples with 54 PII categories, created for privacy-preserving NLP research.

---

*Report generated by Privplay Academic Evaluation Suite*
"""
    
    return report


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(config: EvaluationConfig) -> Dict[str, Any]:
    """Run complete evaluation pipeline."""
    
    start_time = datetime.now()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    samples = load_ai4privacy_dataset(
        max_samples=config.samples,
        seed=config.seed,
    )
    
    # Initialize detection engine
    logger.info("Initializing Privplay detection engine...")
    try:
        from privplay.engine.classifier import ClassificationEngine
        engine = ClassificationEngine(
            use_coreference=False,  # Faster evaluation
            use_meta_classifier=True,
        )
    except ImportError as e:
        logger.error(f"Failed to import Privplay: {e}")
        logger.error("Make sure you're running from the privplay directory")
        sys.exit(1)
    
    # Run detection on all samples
    logger.info(f"Running detection on {len(samples)} samples...")
    all_predictions = []
    all_ground_truth = []
    
    # Debug: track label normalization
    gt_label_counts = defaultdict(int)
    pred_label_counts = defaultdict(int)
    
    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processing sample {i + 1}/{len(samples)}")
        
        # Run detection
        predictions = run_detection(sample["text"], engine)
        all_predictions.append(predictions)
        
        # Normalize ground truth types
        gt_normalized = []
        for e in sample["entities"]:
            normalized_label = normalize_type(e["label"])
            gt_label_counts[normalized_label] += 1
            gt_normalized.append({
                "label": normalized_label,
                "start": e["start"],
                "end": e["end"],
                "text": e.get("text", e.get("value", "")),
                "original_label": e["label"],
            })
        all_ground_truth.append(gt_normalized)
        
        # Track prediction labels
        for p in predictions:
            pred_label_counts[p["label"]] += 1
    
    # Debug output
    logger.info(f"Ground truth label distribution: {dict(gt_label_counts)}")
    logger.info(f"Prediction label distribution: {dict(pred_label_counts)}")
    
    # Get unique tags
    all_tags = set()
    for gt in all_ground_truth:
        for e in gt:
            all_tags.add(e["label"])
    for pred in all_predictions:
        for e in pred:
            all_tags.add(e["label"])
    tags = sorted(list(all_tags - {"OTHER"}))  # Exclude OTHER from evaluation
    
    logger.info(f"Evaluating {len(tags)} entity types: {tags}")
    
    # Run nervaluate
    logger.info("Running nervaluate evaluation...")
    results, results_by_tag = run_nervaluate(all_ground_truth, all_predictions, tags)
    
    # Compute F1 scores (nervaluate doesn't always include them)
    for mode in ['strict', 'exact', 'partial', 'ent_type']:
        p = results[mode].get('precision', 0)
        r = results[mode].get('recall', 0)
        results[mode]['f1'] = compute_f1(p, r)
    
    for tag in results_by_tag:
        for mode in ['strict', 'exact', 'partial', 'ent_type']:
            if mode in results_by_tag[tag]:
                p = results_by_tag[tag][mode].get('precision', 0)
                r = results_by_tag[tag][mode].get('recall', 0)
                results_by_tag[tag][mode]['f1'] = compute_f1(p, r)
    
    # Error analysis
    logger.info("Running error analysis...")
    error_analysis = analyze_errors(samples, all_predictions)
    
    # Calculate runtime
    end_time = datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()
    
    # Compile full results
    full_results = {
        "metadata": {
            "timestamp": start_time.isoformat(),
            "dataset": config.dataset,
            "samples": config.samples,
            "seed": config.seed,
            "runtime_seconds": runtime_seconds,
            "entity_types_evaluated": tags,
        },
        "overall": results,
        "by_entity_type": results_by_tag,
        "error_analysis": error_analysis,
    }
    
    # Save raw JSON
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    logger.info(f"Raw results saved to: {json_path}")
    
    # Generate and save markdown report
    report = generate_markdown_report(
        results, results_by_tag, error_analysis, config, runtime_seconds
    )
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Markdown report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nSamples: {config.samples:,}")
    print(f"Runtime: {runtime_seconds:.1f}s")
    print(f"\nSTRICT EVALUATION (Gold Standard):")
    print(f"  Precision: {results['strict']['precision']:.1%}")
    print(f"  Recall:    {results['strict']['recall']:.1%}")
    print(f"  F1:        {results['strict']['f1']:.1%}")
    print(f"\nOutputs:")
    print(f"  JSON: {json_path}")
    print(f"  Report: {report_path}")
    
    return full_results


def main():
    parser = argparse.ArgumentParser(
        description="Academic-grade NER evaluation using SemEval 2013 methodology"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        samples=args.samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    run_evaluation(config)


if __name__ == "__main__":
    main()
