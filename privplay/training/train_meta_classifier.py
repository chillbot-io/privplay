"""Meta-classifier training pipeline with coreference and document-level features.

Generates training data from multiple sources:
1. Synthetic clinical notes (templated, perfect span tracking)
2. Adversarial cases (hard negatives and edge cases)
3. AI4Privacy samples (general PII dataset)
4. MTSamples (REAL clinical notes with re-injected PHI)

Features include:
- Detector signals (PHI BERT, PII BERT, Presidio, Rules)
- Coreference signals (cluster membership, anchor info, pronouns)
- Document-level context (has_ssn, has_dates, medical_terms, entity_count, pii_density)
- Span-level computed features (length, character composition)

Usage:
    python -m privplay.training.train_meta_classifier \
        --synthetic 2000 \
        --adversarial 1000 \
        --ai4privacy 1000 \
        --mtsamples 1000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def generate_training_data(
    n_synthetic: int = 2000,
    n_adversarial: int = 1000,
    n_ai4privacy: int = 1000,
    n_mtsamples: int = 1000,
) -> Tuple[List, List, List, List]:
    """Generate all training data from multiple sources."""
    from .synthetic_generator import generate_synthetic_dataset, validate_dataset, get_coreference_stats
    from .adversarial_cases import generate_adversarial_dataset, get_adversarial_stats
    
    console.print("\n[bold]Step 1: Generate Training Data[/bold]")
    console.print("─" * 50)
    
    # 1. Synthetic clinical notes
    synthetic_docs = []
    if n_synthetic > 0:
        console.print(f"\nGenerating {n_synthetic} synthetic clinical notes...")
        synthetic_docs = generate_synthetic_dataset(n_synthetic)
        valid, invalid = validate_dataset(synthetic_docs)
        console.print(f"  ✓ Generated {valid} valid documents ({invalid} invalid)")
        
        # Show coreference stats
        coref_stats = get_coreference_stats(synthetic_docs)
        console.print(f"  Coreference: {coref_stats['docs_with_coreference']} docs, {coref_stats['reference_entities']} references")
    
    # 2. Adversarial cases
    adversarial_docs = []
    if n_adversarial > 0:
        console.print(f"\nGenerating {n_adversarial} adversarial cases...")
        adversarial_docs = generate_adversarial_dataset(n_adversarial)
        stats = get_adversarial_stats(adversarial_docs)
        console.print(f"  ✓ Generated {len(adversarial_docs)} adversarial cases")
        console.print("  Distribution:")
        for adv_type, count in sorted(stats.items(), key=lambda x: -x[1])[:5]:
            console.print(f"    {adv_type}: {count}")
    
    # 3. AI4Privacy samples
    ai4privacy_docs = []
    if n_ai4privacy > 0:
        console.print(f"\nLoading {n_ai4privacy} AI4Privacy samples...")
        try:
            ai4privacy_docs = load_ai4privacy_sample(n_ai4privacy)
            console.print(f"  ✓ Loaded {len(ai4privacy_docs)} samples")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not load AI4Privacy: {e}[/yellow]")
            console.print("  Continuing without AI4Privacy data...")
    
    # 4. MTSamples
    mtsamples_docs = []
    if n_mtsamples > 0:
        console.print(f"\nLoading {n_mtsamples} MTSamples documents (with PHI re-injection)...")
        try:
            mtsamples_docs = load_mtsamples_sample(n_mtsamples)
            console.print(f"  ✓ Loaded {len(mtsamples_docs)} MTSamples documents")
            
            # Show specialty distribution
            from collections import Counter
            specialties = Counter(
                doc.metadata.get("medical_specialty", "unknown") 
                for doc in mtsamples_docs
            )
            console.print("  Specialties:")
            for specialty, count in specialties.most_common(5):
                console.print(f"    {specialty}: {count}")
                
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not load MTSamples: {e}[/yellow]")
            console.print("  Continuing without MTSamples data...")
            import traceback
            logger.debug(traceback.format_exc())
    
    return synthetic_docs, adversarial_docs, ai4privacy_docs, mtsamples_docs


def load_ai4privacy_sample(n: int) -> List:
    """Load sample from AI4Privacy dataset."""
    from .synthetic_generator import LabeledDocument, LabeledEntity
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")
    
    # Load dataset
    dataset = load_dataset("ai4privacy/pii-masking-200k", split="train")
    
    # Sample
    indices = list(range(min(n * 2, len(dataset))))
    import random
    random.shuffle(indices)
    
    docs = []
    for idx in indices[:n]:
        row = dataset[idx]
        
        # Parse entities from the dataset format
        entities = []
        if "privacy_mask" in row and row["privacy_mask"]:
            text = row.get("source_text", "")
            masks = row["privacy_mask"]
            
            if isinstance(masks, list):
                for mask in masks:
                    if isinstance(mask, dict):
                        entities.append(LabeledEntity(
                            start=mask.get("start", 0),
                            end=mask.get("end", 0),
                            text=mask.get("value", ""),
                            entity_type=normalize_ai4privacy_type(mask.get("label", "UNKNOWN")),
                        ))
        
        if entities:
            docs.append(LabeledDocument(
                id=f"ai4privacy_{idx}",
                text=row.get("source_text", ""),
                entities=entities,
                doc_type="ai4privacy",
                metadata={"source": "ai4privacy"},
            ))
    
    return docs


def load_mtsamples_sample(n: int) -> List:
    """Load sample from MTSamples with PHI re-injection."""
    from .mtsamples_loader import load_mtsamples_dataset, validate_dataset, get_dataset_stats
    
    # Load and process MTSamples
    docs = load_mtsamples_dataset(
        max_samples=n,
        min_entities=0,  # Include docs without detected placeholders
        validate=True,   # Strict validation
    )
    
    # Log stats
    stats = get_dataset_stats(docs)
    logger.info(f"MTSamples stats: {stats}")
    
    return docs


def normalize_ai4privacy_type(label: str) -> str:
    """Normalize AI4Privacy entity types to our types."""
    mapping = {
        "FIRSTNAME": "NAME_PERSON",
        "LASTNAME": "NAME_PERSON",
        "FULLNAME": "NAME_PERSON",
        "NAME": "NAME_PERSON",
        "EMAIL": "EMAIL",
        "PHONE": "PHONE",
        "PHONENUMBER": "PHONE",
        "SSN": "SSN",
        "SOCIALSECURITYNUMBER": "SSN",
        "DATE": "DATE",
        "DOB": "DATE_DOB",
        "DATEOFBIRTH": "DATE_DOB",
        "ADDRESS": "ADDRESS",
        "STREETADDRESS": "ADDRESS",
        "CITY": "LOCATION",
        "STATE": "LOCATION",
        "ZIP": "ZIP",
        "ZIPCODE": "ZIP",
        "CREDITCARD": "CREDIT_CARD",
        "CREDITCARDNUMBER": "CREDIT_CARD",
        "IP": "IP_ADDRESS",
        "IPADDRESS": "IP_ADDRESS",
        "URL": "URL",
        "USERNAME": "USERNAME",
        "PASSWORD": "PASSWORD",
    }
    return mapping.get(label.upper().replace("_", ""), label.upper())


def capture_signals_for_documents(
    docs: List,
    engine,
    use_coreference: bool = True,
) -> List[Dict]:
    """Run detection and capture signals for each document.
    
    NEW: Enriches signals with coreference information.
    """
    from ..engine.classifier import SpanSignals
    
    console.print("\n[bold]Step 2: Capture Detection Signals[/bold]")
    console.print("─" * 50)
    
    # Import coreference if available
    coref_resolver = None
    if use_coreference:
        try:
            from ..engine.coreference import get_coreference_resolver
            coref_resolver = get_coreference_resolver(device='cpu')
            console.print("  Coreference resolver loaded ✓")
        except Exception as e:
            console.print(f"  [yellow]⚠ Coreference not available: {e}[/yellow]")
            use_coreference = False
    
    all_signals = []
    
    engine.capture_signals = True
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing documents...", total=len(docs))
        
        for doc in docs:
            # Clear previous signals
            engine.clear_captured_signals()
            
            # Run detection
            try:
                engine.detect(doc.text, verify=False)
            except Exception as e:
                logger.warning(f"Detection failed for {doc.id}: {e}")
                progress.update(task, advance=1)
                continue
            
            # Get captured signals
            signals = engine.get_captured_signals()
            
            # Enrich with coreference (NEW)
            if use_coreference and coref_resolver:
                signals = enrich_signals_with_coreference(
                    doc.text, signals, coref_resolver, doc
                )
            
            # Label signals against ground truth
            labeled_signals = label_signals(signals, doc)
            all_signals.extend(labeled_signals)
            
            progress.update(task, advance=1)
    
    engine.capture_signals = False
    
    # Stats
    total = len(all_signals)
    if total > 0:
        positive = sum(1 for s in all_signals if s["ground_truth_type"] != "NONE")
        negative = total - positive
        
        console.print(f"\n  Total signals captured: {total}")
        console.print(f"  Positive (real entities): {positive} ({positive/total*100:.1f}%)")
        console.print(f"  Negative (false positives): {negative} ({negative/total*100:.1f}%)")
        
        # Coreference stats
        coref_signals = sum(1 for s in all_signals if s.get("in_coref_cluster", False))
        console.print(f"  Signals in coref clusters: {coref_signals} ({coref_signals/total*100:.1f}%)")
        
        # Breakdown by source
        from collections import Counter
        sources = Counter(s.get("doc_type", "unknown") for s in all_signals)
        console.print(f"\n  By source:")
        for source, count in sources.most_common():
            console.print(f"    {source}: {count}")
    else:
        console.print("\n  [yellow]No signals captured![/yellow]")
    
    return all_signals


def enrich_signals_with_coreference(
    text: str,
    signals: List,
    coref_resolver,
    doc,
) -> List:
    """Enrich signals with coreference information and add new coref-derived signals."""
    
    # Common pronouns
    PRONOUNS = {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their'}
    
    try:
        # Run coreference
        coref_result = coref_resolver.resolve(text)
        
        # Build detected spans lookup
        detected_spans = [
            {
                'start': s.span_start,
                'end': s.span_end,
                'text': s.span_text,
                'entity_type': s.merged_type,
                'confidence': s.merged_conf,
            }
            for s in signals
        ]
        
        # Enrich with PHI info
        coref_result = coref_resolver.enrich_with_phi(coref_result, detected_spans)
        
        # Update existing signals with coref info
        for signal in signals:
            key = (signal.span_start, signal.span_end)
            
            if key in coref_result.span_to_cluster:
                cluster_id = coref_result.span_to_cluster[key]
                cluster = coref_result.clusters[cluster_id]
                
                signal.in_coref_cluster = True
                signal.coref_cluster_id = cluster_id
                signal.coref_cluster_size = len(cluster.mentions)
                signal.coref_anchor_type = cluster.anchor_type
                signal.coref_anchor_conf = cluster.anchor_confidence
                
                # Check if this is the anchor
                if cluster.anchor_span == key:
                    signal.coref_is_anchor = True
                
                # Check if pronoun
                if signal.span_text.lower() in PRONOUNS:
                    signal.coref_is_pronoun = True
        
        # Get NEW spans from coreference (pronouns, references not originally detected)
        new_coref_signals = []
        detected_set = {(s.span_start, s.span_end) for s in signals}
        
        for cluster in coref_result.clusters:
            # Skip clusters without PHI anchor
            if cluster.anchor_type is None:
                continue
            
            for mention_start, mention_end, mention_text in cluster.mentions:
                key = (mention_start, mention_end)
                
                # Skip if already detected
                if key in detected_set:
                    continue
                
                # Skip if this IS the anchor
                if cluster.anchor_span == key:
                    continue
                
                # Create new signal for this coref mention
                from ..engine.classifier import SpanSignals
                new_signal = SpanSignals(
                    span_start=mention_start,
                    span_end=mention_end,
                    span_text=mention_text,
                    # No detector fired on this
                    phi_bert_detected=False,
                    pii_bert_detected=False,
                    presidio_detected=False,
                    rule_detected=False,
                    # But it's in a coref cluster
                    in_coref_cluster=True,
                    coref_cluster_id=cluster.cluster_id,
                    coref_cluster_size=len(cluster.mentions),
                    coref_anchor_type=cluster.anchor_type,
                    coref_anchor_conf=cluster.anchor_confidence,
                    coref_is_anchor=False,
                    coref_is_pronoun=mention_text.lower() in PRONOUNS,
                    # Computed features
                    span_length=len(mention_text),
                    has_digits=any(c.isdigit() for c in mention_text),
                    has_letters=any(c.isalpha() for c in mention_text),
                    all_caps=mention_text.isupper() and any(c.isalpha() for c in mention_text),
                    all_digits=mention_text.replace("-", "").replace(" ", "").isdigit(),
                    mixed_case=(
                        any(c.isupper() for c in mention_text) and 
                        any(c.islower() for c in mention_text)
                    ),
                    # Inherit type from anchor
                    merged_type=cluster.anchor_type or "",
                    merged_conf=0.0,  # No direct detection
                    merged_source="coreference",
                )
                new_coref_signals.append(new_signal)
        
        # Combine original signals with new coref signals
        signals = list(signals) + new_coref_signals
        
    except Exception as e:
        logger.warning(f"Coreference enrichment failed: {e}")
    
    return signals


def label_signals(signals: List, doc) -> List[Dict]:
    """Label captured signals against document ground truth.
    
    Also computes document-level features for all signals in the document.
    """
    labeled = []
    
    # Build ground truth lookup with tolerance
    gt_lookup = {}
    gt_coref_lookup = {}  # For coreference clusters
    
    for entity in doc.entities:
        for offset in range(-2, 3):  # ±2 char tolerance
            gt_lookup[(entity.start + offset, entity.end + offset)] = entity.entity_type
        
        # Track coreference cluster info if available
        if hasattr(entity, 'coref_cluster_id') and entity.coref_cluster_id is not None:
            if entity.coref_cluster_id not in gt_coref_lookup:
                gt_coref_lookup[entity.coref_cluster_id] = entity.entity_type
    
    # Compute document-level features once for all signals in this doc
    doc_features = compute_document_features(signals, doc)
    
    for signal in signals:
        signal_dict = signal_to_dict(signal)
        signal_dict["document_id"] = doc.id
        signal_dict["doc_type"] = doc.doc_type
        
        # Add document-level features
        signal_dict.update(doc_features)
        
        # Check exact match first
        span_key = (signal.span_start, signal.span_end)
        if span_key in gt_lookup:
            signal_dict["ground_truth_type"] = gt_lookup[span_key]
            signal_dict["ground_truth_source"] = "exact_match"
        else:
            # Check overlap match
            matched = False
            for entity in doc.entities:
                overlap = calculate_overlap(
                    signal.span_start, signal.span_end,
                    entity.start, entity.end
                )
                if overlap > 0.5:
                    signal_dict["ground_truth_type"] = entity.entity_type
                    signal_dict["ground_truth_source"] = f"overlap_{overlap:.2f}"
                    matched = True
                    break
            
            if not matched:
                # This is a false positive (or a coref mention not in ground truth)
                signal_dict["ground_truth_type"] = "NONE"
                signal_dict["ground_truth_source"] = "no_match"
        
        labeled.append(signal_dict)
    
    return labeled


def compute_document_features(signals: List, doc) -> Dict:
    """Compute document-level features from all signals in a document.
    
    These features provide context - e.g., if a document already has SSNs,
    a 9-digit number is more likely to be another SSN.
    """
    # Entity type detection
    detected_types = set()
    for signal in signals:
        if signal.merged_type:
            detected_types.add(signal.merged_type.upper())
        if signal.rule_type:
            detected_types.add(signal.rule_type.upper())
        if signal.phi_bert_type:
            detected_types.add(signal.phi_bert_type.upper())
        if signal.pii_bert_type:
            detected_types.add(signal.pii_bert_type.upper())
    
    # Check for SSN
    doc_has_ssn = "SSN" in detected_types
    
    # Check for dates
    date_types = {"DATE", "DATE_DOB", "DATE_ADMISSION", "DATE_DISCHARGE"}
    doc_has_dates = bool(detected_types & date_types)
    
    # Check for medical terms (heuristic based on doc_type and detected types)
    medical_types = {"DRUG", "DIAGNOSIS", "LAB_TEST", "FACILITY", "NAME_PROVIDER", "MRN"}
    doc_has_medical_terms = bool(detected_types & medical_types)
    
    # Also check doc_type
    clinical_doc_types = {
        "admission_note", "discharge_summary", "progress_note", "lab_report",
        "consultation_note", "referral_letter", "clinical_note", "mtsamples"
    }
    if hasattr(doc, 'doc_type') and doc.doc_type in clinical_doc_types:
        doc_has_medical_terms = True
    
    # Entity count
    doc_entity_count = len(signals)
    
    # PII density: ratio of PII character span to total document length
    total_pii_chars = 0
    for signal in signals:
        total_pii_chars += signal.span_end - signal.span_start
    
    doc_length = len(doc.text) if hasattr(doc, 'text') and doc.text else 1
    doc_pii_density = min(1.0, total_pii_chars / doc_length)
    
    return {
        "doc_has_ssn": doc_has_ssn,
        "doc_has_dates": doc_has_dates,
        "doc_has_medical_terms": doc_has_medical_terms,
        "doc_entity_count": doc_entity_count,
        "doc_pii_density": doc_pii_density,
    }


def signal_to_dict(signal) -> Dict:
    """Convert SpanSignals to dict, including coreference fields."""
    return {
        "id": signal.id,
        "span_start": signal.span_start,
        "span_end": signal.span_end,
        "span_text": signal.span_text,
        
        # Detector signals
        "phi_bert_detected": signal.phi_bert_detected,
        "phi_bert_conf": signal.phi_bert_conf,
        "phi_bert_type": signal.phi_bert_type,
        
        "pii_bert_detected": signal.pii_bert_detected,
        "pii_bert_conf": signal.pii_bert_conf,
        "pii_bert_type": signal.pii_bert_type,
        
        "presidio_detected": signal.presidio_detected,
        "presidio_conf": signal.presidio_conf,
        "presidio_type": signal.presidio_type,
        
        "rule_detected": signal.rule_detected,
        "rule_conf": signal.rule_conf,
        "rule_type": signal.rule_type,
        "rule_has_checksum": signal.rule_has_checksum,
        
        "llm_verified": signal.llm_verified,
        "llm_decision": signal.llm_decision,
        "llm_conf": signal.llm_conf,
        
        # Coreference signals (NEW)
        "in_coref_cluster": getattr(signal, 'in_coref_cluster', False),
        "coref_cluster_id": getattr(signal, 'coref_cluster_id', None),
        "coref_cluster_size": getattr(signal, 'coref_cluster_size', 0),
        "coref_anchor_type": getattr(signal, 'coref_anchor_type', None),
        "coref_anchor_conf": getattr(signal, 'coref_anchor_conf', 0.0),
        "coref_is_anchor": getattr(signal, 'coref_is_anchor', False),
        "coref_is_pronoun": getattr(signal, 'coref_is_pronoun', False),
        
        # Computed features
        "sources_agree_count": signal.sources_agree_count,
        "span_length": signal.span_length,
        "has_digits": signal.has_digits,
        "has_letters": signal.has_letters,
        "all_caps": signal.all_caps,
        "all_digits": signal.all_digits,
        "mixed_case": signal.mixed_case,
        
        "merged_type": signal.merged_type,
        "merged_conf": signal.merged_conf,
        "merged_source": signal.merged_source,
    }


def calculate_overlap(start1: int, end1: int, start2: int, end2: int) -> float:
    """Calculate overlap ratio between two spans."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_len = overlap_end - overlap_start
    min_len = min(end1 - start1, end2 - start2)
    
    if min_len == 0:
        return 0.0
    
    return overlap_len / min_len


def train_meta_classifier(
    signals: List[Dict],
    output_dir: Path,
    use_xgboost: bool = False,
    test_size: float = 0.2,
) -> Dict:
    """Train the meta-classifier on labeled signals.
    
    NOW INCLUDES coreference features.
    """
    console.print("\n[bold]Step 3: Train Meta-Classifier[/bold]")
    console.print("─" * 50)
    
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Prepare features - NOW INCLUDING COREFERENCE AND DOCUMENT-LEVEL
    feature_names = [
        # Detector signals
        "phi_bert_detected", "phi_bert_conf",
        "pii_bert_detected", "pii_bert_conf",
        "presidio_detected", "presidio_conf",
        "rule_detected", "rule_conf", "rule_has_checksum",
        
        # Coreference signals
        "in_coref_cluster", "coref_cluster_size", "coref_anchor_conf",
        "coref_is_anchor", "coref_is_pronoun", "coref_has_phi_anchor",
        
        # Document-level features (context signals)
        "doc_has_ssn", "doc_has_dates", "doc_has_medical_terms",
        "doc_entity_count", "doc_pii_density",
        
        # Computed span features
        "sources_agree_count", "span_length",
        "has_digits", "has_letters", "all_caps", "all_digits", "mixed_case",
    ]
    
    X = []
    y_is_entity = []
    y_entity_type = []
    
    for s in signals:
        features = [
            # Detector signals
            int(s["phi_bert_detected"]),
            s["phi_bert_conf"],
            int(s["pii_bert_detected"]),
            s["pii_bert_conf"],
            int(s["presidio_detected"]),
            s["presidio_conf"],
            int(s["rule_detected"]),
            s["rule_conf"],
            int(s["rule_has_checksum"]),
            
            # Coreference signals
            int(s.get("in_coref_cluster", False)),
            s.get("coref_cluster_size", 0),
            s.get("coref_anchor_conf", 0.0),
            int(s.get("coref_is_anchor", False)),
            int(s.get("coref_is_pronoun", False)),
            int(s.get("coref_anchor_type") is not None),  # coref_has_phi_anchor
            
            # Document-level features (context signals)
            int(s.get("doc_has_ssn", False)),
            int(s.get("doc_has_dates", False)),
            int(s.get("doc_has_medical_terms", False)),
            s.get("doc_entity_count", 0),
            s.get("doc_pii_density", 0.0),
            
            # Computed span features
            s["sources_agree_count"],
            s["span_length"],
            int(s["has_digits"]),
            int(s["has_letters"]),
            int(s["all_caps"]),
            int(s["all_digits"]),
            int(s["mixed_case"]),
        ]
        X.append(features)
        
        is_entity = 1 if s["ground_truth_type"] != "NONE" else 0
        y_is_entity.append(is_entity)
        y_entity_type.append(s["ground_truth_type"])
    
    X = np.array(X)
    y_is_entity = np.array(y_is_entity)
    
    console.print(f"\n  Features: {len(feature_names)}")
    console.print(f"  Samples: {len(X)}")
    console.print(f"  Positive: {sum(y_is_entity)} ({sum(y_is_entity)/len(y_is_entity)*100:.1f}%)")
    console.print(f"  Negative: {len(y_is_entity) - sum(y_is_entity)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_is_entity, test_size=test_size, random_state=42, stratify=y_is_entity
    )
    
    console.print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train is_entity classifier
    console.print("\n  Training is_entity classifier...")
    
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        except ImportError:
            console.print("  [yellow]XGBoost not available, using RandomForest[/yellow]")
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    console.print(f"\n  [bold]Results (is_entity):[/bold]")
    console.print(f"    F1:        {f1:.1%}")
    console.print(f"    Precision: {precision:.1%}")
    console.print(f"    Recall:    {recall:.1%}")
    
    # Feature importance
    importances = [(name, float(imp)) for name, imp in ...]
    importances.sort(key=lambda x: -x[1])
    
    console.print(f"\n  [bold]Top Features:[/bold]")
    for name, imp in importances[:8]:
        console.print(f"    {name}: {imp:.3f}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    model_path = output_dir / "is_entity_model.pkl"
    joblib.dump(clf, model_path)
    console.print(f"\n  Model saved: {model_path}")
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "importances": importances,
        }, f, indent=2)
    
    # Save metadata
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "use_xgboost": use_xgboost,
        "feature_count": len(feature_names),
        "includes_coreference": True,
        "includes_document_features": True,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train meta-classifier")
    parser.add_argument("--synthetic", type=int, default=2000, help="Number of synthetic docs")
    parser.add_argument("--adversarial", type=int, default=1000, help="Number of adversarial cases")
    parser.add_argument("--ai4privacy", type=int, default=1000, help="Number of AI4Privacy samples")
    parser.add_argument("--mtsamples", type=int, default=1000, help="Number of MTSamples docs")
    parser.add_argument("--xgboost", action="store_true", help="Use XGBoost instead of RandomForest")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock model (for testing)")
    parser.add_argument("--no-coref", action="store_true", help="Disable coreference enrichment")
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]═══ Meta-Classifier Training Pipeline ═══[/bold cyan]\n")
    console.print("[dim]With coreference + document-level features![/dim]\n")
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        from ..config import get_config
        output_dir = get_config().data_dir / "meta_classifier"
    
    # Generate data from all sources
    synthetic_docs, adversarial_docs, ai4privacy_docs, mtsamples_docs = generate_training_data(
        n_synthetic=args.synthetic,
        n_adversarial=args.adversarial,
        n_ai4privacy=args.ai4privacy,
        n_mtsamples=args.mtsamples,
    )
    
    all_docs = synthetic_docs + adversarial_docs + ai4privacy_docs + mtsamples_docs
    console.print(f"\n  Total documents: {len(all_docs)}")
    console.print(f"    Synthetic: {len(synthetic_docs)}")
    console.print(f"    Adversarial: {len(adversarial_docs)}")
    console.print(f"    AI4Privacy: {len(ai4privacy_docs)}")
    console.print(f"    MTSamples: {len(mtsamples_docs)}")
    
    if len(all_docs) == 0:
        console.print("[red]No documents to process![/red]")
        return
    
    # Initialize engine
    console.print("\n  Initializing detection engine...")
    from ..config import get_config
    from ..engine.classifier import ClassificationEngine
    
    config = get_config()
    engine = ClassificationEngine(
        use_mock_model=args.mock, 
        config=config,
        use_coreference=not args.no_coref,
    )
    
    # Capture signals (with coreference enrichment)
    signals = capture_signals_for_documents(
        all_docs, 
        engine,
        use_coreference=not args.no_coref,
    )
    
    if len(signals) == 0:
        console.print("[red]No signals captured! Check your detectors.[/red]")
        return
    
    # Save signals for analysis
    signals_path = output_dir / "training_signals.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(signals_path, "w") as f:
        json.dump(signals, f)
    console.print(f"\n  Signals saved: {signals_path}")
    
    # Train
    metrics = train_meta_classifier(
        signals,
        output_dir,
        use_xgboost=args.xgboost,
    )
    
    console.print("\n[bold green]═══ Training Complete ═══[/bold green]\n")
    
    # Summary table
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Documents", str(len(all_docs)))
    table.add_row("  - Synthetic", str(len(synthetic_docs)))
    table.add_row("  - Adversarial", str(len(adversarial_docs)))
    table.add_row("  - AI4Privacy", str(len(ai4privacy_docs)))
    table.add_row("  - MTSamples", str(len(mtsamples_docs)))
    table.add_row("Signals", str(len(signals)))
    table.add_row("Features", str(metrics['feature_count']))
    table.add_row("F1 Score", f"{metrics['f1']:.1%}")
    table.add_row("Precision", f"{metrics['precision']:.1%}")
    table.add_row("Recall", f"{metrics['recall']:.1%}")
    table.add_row("Coreference", "Enabled" if not args.no_coref else "Disabled")
    table.add_row("Output", str(output_dir))
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
