#!/usr/bin/env python3
"""
Pipeline Tracer - Traces examples through detection pipeline to diagnose issues.

Run from your privplay directory:
    python scripts/trace_pipeline.py --samples 100
    python scripts/trace_pipeline.py --samples 50 --verbose
    python scripts/trace_pipeline.py --samples 100 --failures-only
    python scripts/trace_pipeline.py --samples 100 --output trace_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict
import logging

# Add parent directory to path so we can import privplay
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    """Result from a single detector."""
    detector: str
    entities: List[Dict] = field(default_factory=list)
    
    def add(self, text: str, start: int, end: int, entity_type: str, confidence: float):
        self.entities.append({
            'text': text,
            'start': start,
            'end': end,
            'type': entity_type,
            'confidence': round(confidence, 3),
        })


@dataclass 
class TraceResult:
    """Complete trace of one sample through the pipeline."""
    sample_id: int
    text: str
    text_preview: str
    
    ground_truth: List[Dict] = field(default_factory=list)
    ground_truth_canonical: List[Dict] = field(default_factory=list)
    
    phi_bert: DetectorResult = field(default_factory=lambda: DetectorResult("phi_bert"))
    pii_bert: DetectorResult = field(default_factory=lambda: DetectorResult("pii_bert"))
    presidio: DetectorResult = field(default_factory=lambda: DetectorResult("presidio"))
    rules: DetectorResult = field(default_factory=lambda: DetectorResult("rules"))
    
    final_output: List[Dict] = field(default_factory=list)
    final_canonical: List[Dict] = field(default_factory=list)
    
    true_positives: List[Dict] = field(default_factory=list)
    false_positives: List[Dict] = field(default_factory=list)
    false_negatives: List[Dict] = field(default_factory=list)
    
    issues: List[str] = field(default_factory=list)


# =============================================================================
# TYPE MAPPING (from nervaluate_runner.py)
# =============================================================================

CANONICAL_TYPE_MAP = {
    # Names
    "FIRSTNAME": "NAME", "LASTNAME": "NAME", "MIDDLENAME": "NAME",
    "NAME_PERSON": "NAME", "NAME_PATIENT": "NAME", "NAME_PROVIDER": "NAME",
    "NAME_RELATIVE": "NAME", "PERSON": "NAME",
    
    # Contact
    "EMAIL": "CONTACT", "PHONENUMBER": "CONTACT", "PHONE": "CONTACT",
    "FAX": "CONTACT", "URL": "CONTACT",
    
    # Dates
    "DATE": "DATE", "DOB": "DATE", "DATE_DOB": "DATE",
    "DATE_ADMISSION": "DATE", "DATE_DISCHARGE": "DATE", "TIME": "DATE",
    
    # Location
    "ADDRESS": "LOCATION", "STREET": "LOCATION", "CITY": "LOCATION",
    "STATE": "LOCATION", "COUNTY": "LOCATION", "ZIPCODE": "LOCATION",
    "ZIP": "LOCATION", "BUILDINGNUMBER": "LOCATION", "LOCATION": "LOCATION",
    "SECONDARYADDRESS": "LOCATION", "NEARBYGPSCOORDINATE": "LOCATION",
    "GPS_COORDINATE": "LOCATION",
    
    # Identifiers
    "SSN": "IDNUM", "US_SSN": "IDNUM", "ACCOUNTNUMBER": "IDNUM",
    "ACCOUNT_NUMBER": "IDNUM", "BANK_ACCOUNT": "IDNUM", "MRN": "IDNUM",
    "HEALTH_PLAN_ID": "IDNUM", "DRIVER_LICENSE": "IDNUM",
    "PASSPORT": "IDNUM", "IBAN": "IDNUM",
    
    # Financial
    "CREDITCARDNUMBER": "FINANCIAL", "CREDIT_CARD": "FINANCIAL",
    "CREDITCARDCVV": "FINANCIAL", "PIN": "FINANCIAL",
    
    # Digital
    "IP_ADDRESS": "DIGITAL", "MAC_ADDRESS": "DIGITAL", "USERNAME": "DIGITAL",
    "PASSWORD": "DIGITAL", "USER_AGENT": "DIGITAL", "USERAGENT": "DIGITAL",
    "IMEI": "DIGITAL", "VEHICLEVIN": "DIGITAL", "VIN": "DIGITAL",
    
    # Crypto
    "CRYPTO_ADDRESS": "CRYPTO", "BITCOINADDRESS": "CRYPTO",
    "LITECOINADDRESS": "CRYPTO", "ETHEREUMADDRESS": "CRYPTO",
    
    # Organization
    "COMPANY": "ORG", "COMPANYNAME": "ORG", "FACILITY": "ORG",
    
    # Excluded (not PII) - map to None
    "GENDER": None, "SEX": None, "JOBTITLE": None, "JOBTYPE": None,
    "JOBAREA": None, "JOBDESCRIPTOR": None, "HEIGHT": None, "WEIGHT": None,
    "HAIRCOLOR": None, "EYECOLOR": None, "BLOODTYPE": None,
    "CURRENCYSYMBOL": None, "CURRENCYCODE": None, "CURRENCYNAME": None,
    "CURRENCY": None, "AMOUNT": None, "NUMBER": None,
    "VEHICLEVRM": None, "VEHICLEMODEL": None, "VEHICLECOLOR": None,
    "VEHICLETYPE": None, "ORDINALDIRECTION": None, "MASKEDNUMBER": None,
    "CREDITCARDISSUER": None,
}

def get_canonical_type(entity_type: str) -> Optional[str]:
    """Map entity type to canonical category."""
    return CANONICAL_TYPE_MAP.get(entity_type.upper(), "OTHER")


def spans_overlap(s1: Tuple[int, int], s2: Tuple[int, int], threshold: float = 0.5) -> bool:
    """Check if two spans overlap by at least threshold proportion."""
    start1, end1 = s1
    start2, end2 = s2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return False
    
    overlap_len = overlap_end - overlap_start
    min_len = min(end1 - start1, end2 - start2)
    
    return (overlap_len / min_len) >= threshold if min_len > 0 else False


def evaluate_sample(
    predictions: List[Dict], 
    ground_truth: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Evaluate predictions against ground truth."""
    true_positives = []
    false_positives = []
    false_negatives = []
    
    matched_gt = set()
    matched_pred = set()
    
    for i, pred in enumerate(predictions):
        pred_span = (pred['start'], pred['end'])
        pred_type = pred.get('canonical_type', pred.get('type', ''))
        
        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue
                
            gt_span = (gt['start'], gt['end'])
            gt_type = gt.get('canonical_type', gt.get('type', ''))
            
            if spans_overlap(pred_span, gt_span):
                true_positives.append({
                    'pred': pred,
                    'gt': gt,
                    'type_match': pred_type == gt_type,
                    'pred_type': pred_type,
                    'gt_type': gt_type,
                })
                matched_gt.add(j)
                matched_pred.add(i)
                break
    
    for i, pred in enumerate(predictions):
        if i not in matched_pred:
            false_positives.append(pred)
    
    for j, gt in enumerate(ground_truth):
        if j not in matched_gt:
            false_negatives.append(gt)
    
    return true_positives, false_positives, false_negatives


def diagnose_issues(trace: TraceResult) -> List[str]:
    """Analyze trace to identify specific issues."""
    issues = []
    
    for fn in trace.false_negatives:
        gt_type = fn.get('original_type', fn.get('type', ''))
        gt_canonical = fn.get('canonical_type', '')
        gt_text = fn.get('text', '')[:40]
        gt_start, gt_end = fn['start'], fn['end']
        
        # Check which detectors saw it
        detectors_saw = []
        
        for det_name, det_result in [
            ('phi_bert', trace.phi_bert),
            ('pii_bert', trace.pii_bert),
            ('presidio', trace.presidio),
            ('rules', trace.rules),
        ]:
            for e in det_result.entities:
                if spans_overlap((e['start'], e['end']), (gt_start, gt_end)):
                    detectors_saw.append(f"{det_name}:{e['type']}")
        
        if detectors_saw:
            issues.append(
                f"FN: '{gt_text}' ({gt_type}→{gt_canonical}) - "
                f"Seen by [{', '.join(detectors_saw)}] but LOST IN MERGE"
            )
        else:
            issues.append(
                f"FN: '{gt_text}' ({gt_type}→{gt_canonical}) - "
                f"NO DETECTOR found this"
            )
    
    for fp in trace.false_positives:
        fp_type = fp.get('type', '')
        fp_text = fp.get('text', '')[:40]
        source = fp.get('source', 'unknown')
        issues.append(f"FP: '{fp_text}' ({fp_type}) - Source: {source}")
    
    for tp in trace.true_positives:
        if not tp.get('type_match', True):
            pred_type = tp.get('pred_type', '')
            gt_type = tp.get('gt_type', '')
            text = tp['pred'].get('text', '')[:40]
            issues.append(f"TYPE_MISMATCH: '{text}' - Predicted {pred_type}, Expected {gt_type}")
    
    return issues


class PipelineTracer:
    """Traces samples through detection pipeline."""
    
    def __init__(self, use_meta_classifier: bool = True):
        self.use_meta_classifier = use_meta_classifier
        self._classifier = None
        self._phi_model = None
        self._pii_model = None
        self._presidio = None
        self._rules = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialize all components."""
        if self._classifier is not None:
            return
            
        print("Initializing detection pipeline...")
        
        # Try to import classifier - this is the main thing we need
        try:
            from privplay.engine.classifier import ClassificationEngine
            classifier_available = True
        except ImportError as e:
            print(f"  ✗ ClassificationEngine import failed: {e}")
            print("    Check that privplay.engine.models exists")
            classifier_available = False
        
        # Try individual detectors for tracing
        try:
            from privplay.engine.models.transformer import get_model
            self._phi_model = get_model()
            print("  ✓ PHI BERT loaded")
        except ImportError as e:
            print(f"  ✗ PHI BERT import failed: {e}")
        except Exception as e:
            print(f"  ✗ PHI BERT failed: {e}")
        
        try:
            from privplay.engine.models.pii_transformer import get_pii_model
            self._pii_model = get_pii_model()
            print("  ✓ PII BERT loaded")
        except ImportError as e:
            print(f"  ✗ PII BERT import failed: {e}")
        except Exception as e:
            print(f"  ✗ PII BERT failed: {e}")
        
        try:
            from privplay.engine.models.presidio_detector import get_presidio_detector
            self._presidio = get_presidio_detector()
            print("  ✓ Presidio loaded")
        except ImportError as e:
            print(f"  ✗ Presidio import failed: {e}")
        except Exception as e:
            print(f"  ✗ Presidio failed: {e}")
        
        try:
            from privplay.engine.rules.engine import RuleEngine
            self._rules = RuleEngine()
            print("  ✓ Rules engine loaded")
        except ImportError as e:
            print(f"  ✗ Rules engine import failed: {e}")
        except Exception as e:
            print(f"  ✗ Rules engine failed: {e}")
        
        if classifier_available:
            try:
                self._classifier = ClassificationEngine(
                    use_meta_classifier=self.use_meta_classifier,
                    use_coreference=False,
                    capture_signals=True,
                )
                print("  ✓ Classifier loaded")
            except Exception as e:
                print(f"  ✗ Classifier init failed: {e}")
        
        if self._classifier is None:
            print("\n  ⚠ Running in LIMITED MODE - only individual detectors available")
            print("    Final merged output will be empty\n")
    
    def trace_sample(self, sample_id: int, text: str, annotations: List[Dict]) -> TraceResult:
        """Trace a single sample through the pipeline."""
        self._lazy_init()
        
        trace = TraceResult(
            sample_id=sample_id,
            text=text,
            text_preview=text[:100] + "..." if len(text) > 100 else text,
        )
        
        # Store ground truth
        for ann in annotations:
            gt_text = text[ann['start']:ann['end']] if ann['start'] < len(text) else ""
            original_type = ann.get('original_type', ann['type'])
            canonical = get_canonical_type(original_type)
            
            if canonical is None:  # Skip excluded types
                continue
                
            trace.ground_truth.append({
                'text': gt_text,
                'start': ann['start'],
                'end': ann['end'],
                'type': ann['type'],
                'original_type': original_type,
            })
            
            trace.ground_truth_canonical.append({
                'text': gt_text,
                'start': ann['start'],
                'end': ann['end'],
                'type': ann['type'],
                'canonical_type': canonical,
                'original_type': original_type,
            })
        
        # Run individual detectors
        if self._phi_model:
            try:
                entities = self._phi_model.detect(text)
                for e in entities:
                    etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                    trace.phi_bert.add(e.text, e.start, e.end, etype, e.confidence)
            except Exception as ex:
                trace.issues.append(f"PHI BERT error: {ex}")
        
        if self._pii_model:
            try:
                entities = self._pii_model.detect(text)
                for e in entities:
                    etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                    trace.pii_bert.add(e.text, e.start, e.end, etype, e.confidence)
            except Exception as ex:
                trace.issues.append(f"PII BERT error: {ex}")
        
        if self._presidio:
            try:
                entities = self._presidio.detect(text)
                for e in entities:
                    etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                    trace.presidio.add(e.text, e.start, e.end, etype, e.confidence)
            except Exception as ex:
                trace.issues.append(f"Presidio error: {ex}")
        
        if self._rules:
            try:
                entities = self._rules.detect(text)
                for e in entities:
                    etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                    trace.rules.add(e.text, e.start, e.end, etype, e.confidence)
            except Exception as ex:
                trace.issues.append(f"Rules error: {ex}")
        
        # Run full classifier
        if self._classifier:
            try:
                entities = self._classifier.detect(text, verify=False)
                for e in entities:
                    etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                    canonical = get_canonical_type(etype)
                    
                    trace.final_output.append({
                        'text': e.text,
                        'start': e.start,
                        'end': e.end,
                        'type': etype,
                        'confidence': round(e.confidence, 3),
                        'source': e.source.value if hasattr(e.source, 'value') else str(e.source),
                    })
                    
                    if canonical:  # Only include if it maps to a canonical type
                        trace.final_canonical.append({
                            'text': e.text,
                            'start': e.start,
                            'end': e.end,
                            'type': etype,
                            'canonical_type': canonical,
                            'confidence': round(e.confidence, 3),
                            'source': e.source.value if hasattr(e.source, 'value') else str(e.source),
                        })
            except Exception as ex:
                trace.issues.append(f"Classifier error: {ex}")
        
        # Evaluate
        trace.true_positives, trace.false_positives, trace.false_negatives = evaluate_sample(
            trace.final_canonical,
            trace.ground_truth_canonical,
        )
        
        # Diagnose
        trace.issues.extend(diagnose_issues(trace))
        
        return trace


def print_trace(trace: TraceResult, verbose: bool = False):
    """Pretty print a trace result."""
    print(f"\n{'='*80}")
    print(f"SAMPLE {trace.sample_id}")
    print(f"{'='*80}")
    print(f"Text: {trace.text_preview}")
    print()
    
    print(f"GROUND TRUTH ({len(trace.ground_truth_canonical)} entities):")
    for gt in trace.ground_truth_canonical:
        print(f"  [{gt['start']:4d}:{gt['end']:4d}] {gt.get('original_type',''):20s} → {gt['canonical_type']:12s} | '{gt['text'][:30]}'")
    print()
    
    if verbose:
        print(f"PHI BERT ({len(trace.phi_bert.entities)}):")
        for e in trace.phi_bert.entities:
            print(f"  [{e['start']:4d}:{e['end']:4d}] {e['type']:20s} ({e['confidence']:.2f}) | '{e['text'][:30]}'")
        
        print(f"\nPII BERT ({len(trace.pii_bert.entities)}):")
        for e in trace.pii_bert.entities:
            print(f"  [{e['start']:4d}:{e['end']:4d}] {e['type']:20s} ({e['confidence']:.2f}) | '{e['text'][:30]}'")
        
        print(f"\nPresidio ({len(trace.presidio.entities)}):")
        for e in trace.presidio.entities:
            print(f"  [{e['start']:4d}:{e['end']:4d}] {e['type']:20s} ({e['confidence']:.2f}) | '{e['text'][:30]}'")
        
        print(f"\nRules ({len(trace.rules.entities)}):")
        for e in trace.rules.entities:
            print(f"  [{e['start']:4d}:{e['end']:4d}] {e['type']:20s} ({e['confidence']:.2f}) | '{e['text'][:30]}'")
        print()
    
    print(f"FINAL OUTPUT ({len(trace.final_canonical)} entities):")
    for e in trace.final_canonical:
        print(f"  [{e['start']:4d}:{e['end']:4d}] {e['type']:20s} → {e['canonical_type']:12s} ({e['confidence']:.2f}) | '{e['text'][:30]}'")
    print()
    
    tp = len(trace.true_positives)
    fp = len(trace.false_positives)
    fn = len(trace.false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"EVALUATION: TP={tp}, FP={fp}, FN={fn}")
    print(f"            P={precision:.1%}, R={recall:.1%}, F1={f1:.1%}")
    print()
    
    if trace.issues:
        print("ISSUES:")
        for issue in trace.issues:
            print(f"  • {issue}")


def print_summary(traces: List[TraceResult]):
    """Print aggregate summary."""
    total_tp = sum(len(t.true_positives) for t in traces)
    total_fp = sum(len(t.false_positives) for t in traces)
    total_fn = sum(len(t.false_negatives) for t in traces)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*80}")
    print(f"Samples: {len(traces)}")
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1 Score:  {f1:.1%}")
    print()
    
    # Issue breakdown
    issue_counts = defaultdict(int)
    for trace in traces:
        for issue in trace.issues:
            if issue.startswith("FN:"):
                if "NO DETECTOR" in issue:
                    issue_counts["FN - No detector found"] += 1
                elif "LOST IN MERGE" in issue:
                    issue_counts["FN - Lost in merge"] += 1
                else:
                    issue_counts["FN - Other"] += 1
            elif issue.startswith("FP:"):
                if "Source:" in issue:
                    source = issue.split("Source:")[-1].strip()
                    issue_counts[f"FP - {source}"] += 1
            elif issue.startswith("TYPE_MISMATCH:"):
                issue_counts["Type mismatch"] += 1
    
    print("ISSUE BREAKDOWN:")
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d} | {issue_type}")
    
    # FN by ground truth type
    fn_by_type = defaultdict(int)
    fn_lost_in_merge_by_type = defaultdict(int)
    fn_no_detector_by_type = defaultdict(int)
    
    for trace in traces:
        for fn in trace.false_negatives:
            fn_type = fn.get('original_type', fn.get('type', 'UNKNOWN'))
            fn_by_type[fn_type] += 1
        
        for issue in trace.issues:
            if "LOST IN MERGE" in issue:
                # Extract type from issue
                if "→" in issue:
                    type_part = issue.split("(")[1].split("→")[0]
                    fn_lost_in_merge_by_type[type_part] += 1
            elif "NO DETECTOR" in issue:
                if "→" in issue:
                    type_part = issue.split("(")[1].split("→")[0]
                    fn_no_detector_by_type[type_part] += 1
    
    print("\nFALSE NEGATIVES BY TYPE (top 15):")
    for fn_type, count in sorted(fn_by_type.items(), key=lambda x: -x[1])[:15]:
        print(f"  {count:4d} | {fn_type}")
    
    if fn_lost_in_merge_by_type:
        print("\nLOST IN MERGE BY TYPE (top 10):")
        for fn_type, count in sorted(fn_lost_in_merge_by_type.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:4d} | {fn_type}")
    
    if fn_no_detector_by_type:
        print("\nNO DETECTOR FOUND BY TYPE (top 10):")
        for fn_type, count in sorted(fn_no_detector_by_type.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:4d} | {fn_type}")
    
    # FP by type
    fp_by_type = defaultdict(int)
    for trace in traces:
        for fp in trace.false_positives:
            fp_type = fp.get('type', 'UNKNOWN')
            fp_by_type[fp_type] += 1
    
    print("\nFALSE POSITIVES BY TYPE (top 10):")
    for fp_type, count in sorted(fp_by_type.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:4d} | {fp_type}")


def main():
    parser = argparse.ArgumentParser(description="Trace samples through detection pipeline")
    parser.add_argument("--samples", "-n", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all detector outputs")
    parser.add_argument("--failures-only", "-f", action="store_true", help="Only show samples with errors")
    parser.add_argument("--no-meta", action="store_true", help="Disable meta-classifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Loading AI4Privacy dataset...")
    try:
        from privplay.benchmark.datasets import load_ai4privacy_dataset
    except ModuleNotFoundError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure you're running from the privplay directory:")
        print("  cd /mnt/d/privplay")
        print("  python scripts/trace_pipeline.py --samples 100")
        print("\nOr that privplay is installed:")
        print("  pip install -e .")
        sys.exit(1)
    
    dataset = load_ai4privacy_dataset(max_samples=args.samples)
    print(f"Loaded {len(dataset.samples)} samples")
    
    tracer = PipelineTracer(use_meta_classifier=not args.no_meta)
    
    traces = []
    for i, sample in enumerate(dataset.samples):
        print(f"\rProcessing sample {i+1}/{len(dataset.samples)}...", end="", flush=True)
        
        # Convert AnnotatedEntity objects to dicts
        annotations = []
        for e in sample.entities:
            annotations.append({
                'start': e.start,
                'end': e.end,
                'type': e.normalized_type,
                'original_type': e.entity_type,
            })
        
        trace = tracer.trace_sample(i, sample.text, annotations)
        traces.append(trace)
    
    print("\n")
    
    for trace in traces:
        has_issues = len(trace.false_positives) > 0 or len(trace.false_negatives) > 0
        if args.failures_only and not has_issues:
            continue
        print_trace(trace, verbose=args.verbose)
    
    print_summary(traces)
    
    if args.output:
        output_data = []
        for trace in traces:
            output_data.append({
                'sample_id': trace.sample_id,
                'text': trace.text,
                'ground_truth': trace.ground_truth_canonical,
                'phi_bert': trace.phi_bert.entities,
                'pii_bert': trace.pii_bert.entities,
                'presidio': trace.presidio.entities,
                'rules': trace.rules.entities,
                'final_output': trace.final_canonical,
                'true_positives': trace.true_positives,
                'false_positives': trace.false_positives,
                'false_negatives': trace.false_negatives,
                'issues': trace.issues,
            })
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nTraces saved to {args.output}")


if __name__ == "__main__":
    main()
