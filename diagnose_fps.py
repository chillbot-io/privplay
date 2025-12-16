#!/usr/bin/env python3
"""
Diagnose False Positives - Find what's causing low precision.

Run on your machine:
    python diagnose_fps.py

This will:
1. Load 200 AI4Privacy samples
2. Run detection
3. Show exactly which detections are FPs and why
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

# Add privplay to path if needed
sys.path.insert(0, '/mnt/d/privplay')

from privplay.engine.classifier import ClassificationEngine
from privplay.benchmark import get_dataset
from privplay.config import get_config


@dataclass
class FPAnalysis:
    """Analysis of a false positive."""
    text: str
    detected_type: str
    source: str  # rule, presidio, phi_bert, pii_bert
    confidence: float
    context_before: str
    context_after: str
    reason: str  # Why it's likely a FP


def get_context(full_text: str, start: int, end: int, window: int = 40) -> Tuple[str, str]:
    """Get context before and after a span."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(full_text), end + window)
    return full_text[ctx_start:start], full_text[end:ctx_end]


def analyze_sample(engine, sample, ground_truth_spans: Set[Tuple[int, int]]) -> List[FPAnalysis]:
    """Analyze false positives in a single sample."""
    fps = []
    
    detected = engine.detect(sample.text, verify=False)
    
    for entity in detected:
        # Check if this is a true positive (overlaps with ground truth)
        is_tp = False
        for gt_start, gt_end in ground_truth_spans:
            # Check overlap
            overlap_start = max(entity.start, gt_start)
            overlap_end = min(entity.end, gt_end)
            if overlap_start < overlap_end:
                overlap_len = overlap_end - overlap_start
                min_len = min(entity.end - entity.start, gt_end - gt_start)
                if min_len > 0 and overlap_len / min_len >= 0.5:
                    is_tp = True
                    break
        
        if not is_tp:
            ctx_before, ctx_after = get_context(sample.text, entity.start, entity.end)
            
            # Try to guess why it's a FP
            reason = "unknown"
            text_lower = entity.text.lower()
            
            if len(entity.text) <= 2:
                reason = "too_short"
            elif entity.text.isdigit() and len(entity.text) < 5:
                reason = "short_number"
            elif text_lower in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}:
                reason = "common_word"
            elif entity.entity_type.value in ['ADDRESS', 'LOCATION'] and len(entity.text.split()) == 1:
                reason = "single_word_address"
            elif entity.entity_type.value in ['NAME_PERSON', 'NAME_PATIENT'] and len(entity.text.split()) == 1:
                reason = "single_word_name"
            elif any(c.isdigit() for c in entity.text) and any(c.isalpha() for c in entity.text):
                reason = "mixed_alphanumeric"
            
            fps.append(FPAnalysis(
                text=entity.text,
                detected_type=entity.entity_type.value,
                source=entity.source.value,
                confidence=entity.confidence,
                context_before=ctx_before,
                context_after=ctx_after,
                reason=reason,
            ))
    
    return fps


def main():
    print("=" * 70)
    print("FALSE POSITIVE DIAGNOSIS")
    print("=" * 70)
    print()
    
    # Load dataset
    print("Loading AI4Privacy dataset (200 samples)...")
    dataset = get_dataset("ai4privacy", max_samples=200)
    samples = list(dataset)
    print(f"Loaded {len(samples)} samples")
    print()
    
    # Initialize engine
    print("Initializing detection engine...")
    config = get_config()
    engine = ClassificationEngine(config=config)
    print("Engine ready")
    print()
    
    # Analyze all samples
    print("Analyzing samples for false positives...")
    print()
    
    all_fps: List[FPAnalysis] = []
    
    for i, sample in enumerate(samples):
        # Build ground truth set
        gt_spans = {(e.start, e.end) for e in sample.entities}
        
        fps = analyze_sample(engine, sample, gt_spans)
        all_fps.extend(fps)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples, {len(all_fps)} FPs so far")
    
    print()
    print(f"Total false positives: {len(all_fps)}")
    print()
    
    # Aggregate by type
    print("=" * 70)
    print("FPs BY ENTITY TYPE")
    print("=" * 70)
    by_type = defaultdict(list)
    for fp in all_fps:
        by_type[fp.detected_type].append(fp)
    
    for entity_type, fps in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n{entity_type}: {len(fps)} FPs")
        print("-" * 50)
        
        # Show examples
        for fp in fps[:5]:
            print(f"  '{fp.text}' (conf={fp.confidence:.2f}, src={fp.source})")
            print(f"    Context: ...{fp.context_before}[{fp.text}]{fp.context_after}...")
            print(f"    Reason: {fp.reason}")
    
    # Aggregate by source
    print()
    print("=" * 70)
    print("FPs BY SOURCE COMPONENT")
    print("=" * 70)
    by_source = defaultdict(list)
    for fp in all_fps:
        by_source[fp.source].append(fp)
    
    for source, fps in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"\n{source}: {len(fps)} FPs")
        
        # Break down by type within source
        source_by_type = defaultdict(int)
        for fp in fps:
            source_by_type[fp.detected_type] += 1
        
        for t, count in sorted(source_by_type.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")
    
    # Aggregate by reason
    print()
    print("=" * 70)
    print("FPs BY SUSPECTED REASON")
    print("=" * 70)
    by_reason = defaultdict(list)
    for fp in all_fps:
        by_reason[fp.reason].append(fp)
    
    for reason, fps in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        print(f"\n{reason}: {len(fps)} FPs")
        
        # Show examples
        for fp in fps[:3]:
            print(f"  '{fp.text}' -> {fp.detected_type} (src={fp.source})")
    
    # Top FP texts (repeated false positives)
    print()
    print("=" * 70)
    print("MOST COMMON FP TEXTS (candidates for allowlist)")
    print("=" * 70)
    text_counts = defaultdict(int)
    for fp in all_fps:
        text_counts[fp.text.lower()].append(fp)
    
    # Sort by count
    text_counts_list = [(text, fps) for text, fps in text_counts.items() if len(fps) > 1]
    text_counts_list.sort(key=lambda x: -len(x[1]))
    
    for text, fps in text_counts_list[:20]:
        types = set(fp.detected_type for fp in fps)
        sources = set(fp.source for fp in fps)
        print(f"  '{text}' x{len(fps)} -> {types} from {sources}")
    
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Generate recommendations based on findings
    if by_source.get('presidio', []):
        presidio_fps = len(by_source['presidio'])
        print(f"\n1. PRESIDIO ({presidio_fps} FPs)")
        print("   - Consider raising score_threshold from 0.5 to 0.7")
        print("   - Add more terms to GLOBAL_ALLOW_LIST")
        
        presidio_types = defaultdict(int)
        for fp in by_source['presidio']:
            presidio_types[fp.detected_type] += 1
        worst_type = max(presidio_types.items(), key=lambda x: x[1])
        print(f"   - Worst type: {worst_type[0]} ({worst_type[1]} FPs)")
    
    if by_source.get('rule', []):
        rule_fps = len(by_source['rule'])
        print(f"\n2. RULES ({rule_fps} FPs)")
        print("   - Review patterns that are too broad")
        
        rule_types = defaultdict(int)
        for fp in by_source['rule']:
            rule_types[fp.detected_type] += 1
        worst_type = max(rule_types.items(), key=lambda x: x[1])
        print(f"   - Worst type: {worst_type[0]} ({worst_type[1]} FPs)")
    
    if by_reason.get('single_word_name', []):
        print(f"\n3. SINGLE-WORD NAMES ({len(by_reason['single_word_name'])} FPs)")
        print("   - Require names to have 2+ words, or")
        print("   - Require context like 'Mr.', 'Dr.', 'Patient:'")
    
    if by_reason.get('single_word_address', []):
        print(f"\n4. SINGLE-WORD ADDRESSES ({len(by_reason['single_word_address'])} FPs)")
        print("   - Require addresses to have street number + name, or")
        print("   - Require multiple components (city + state)")


if __name__ == "__main__":
    main()
