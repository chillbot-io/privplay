#!/usr/bin/env python3
"""
Privplay Smoke Test Suite
Run this to verify all components are working.

Usage:
    python test_smoke.py           # Run all tests
    python test_smoke.py --quick   # Skip slow tests (model loading)
    python test_smoke.py --verbose # Show detailed output
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable, Optional
import traceback

# Test result tracking
@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    details: Optional[str] = None

results: List[TestResult] = []

def test(name: str):
    """Decorator to register and run a test."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                if result is None or result is True:
                    print(f"✓ PASSED ({duration:.2f}s)")
                    results.append(TestResult(name, True, duration))
                else:
                    print(f"✗ FAILED: {result}")
                    results.append(TestResult(name, False, duration, str(result)))
                    
            except Exception as e:
                duration = time.time() - start
                print(f"✗ ERROR: {e}")
                if VERBOSE:
                    traceback.print_exc()
                results.append(TestResult(name, False, duration, str(e), traceback.format_exc()))
        
        wrapper._test_name = name
        wrapper._test_func = func
        return wrapper
    return decorator


# =============================================================================
# IMPORTS TEST
# =============================================================================

@test("Import Core Modules")
def test_imports_core():
    from privplay.types import Entity, EntityType, SourceType, DecisionType
    from privplay.config import get_config, Config
    from privplay.db import Database, get_db, set_db
    print(f"  EntityType values: {len(EntityType)} types")
    print(f"  SourceType values: {[s.value for s in SourceType]}")
    return True

@test("Import Training Modules")
def test_imports_training():
    from privplay.training.faker_gen import generate_documents
    from privplay.training.scanner import scan_documents
    from privplay.training.reviewer import run_review_session
    from privplay.training.rules import RuleManager, CustomRule
    from privplay.training.finetune import finetune
    print("  All training modules imported")
    return True

@test("Import Engine Modules")
def test_imports_engine():
    from privplay.engine.rules.engine import RuleEngine, Rule
    from privplay.engine.classifier import ClassificationEngine
    print("  All engine modules imported")
    return True

@test("Import Benchmark Modules")
def test_imports_benchmark():
    from privplay.benchmark import (
        list_datasets,
        get_dataset,
        BenchmarkRunner,
        BenchmarkStorage,
    )
    from privplay.benchmark.datasets import (
        load_synthetic_phi_dataset,
        BenchmarkDataset,
        BenchmarkSample,
    )
    print("  All benchmark modules imported")
    return True

@test("Import Dictionary Modules")
def test_imports_dictionaries():
    from privplay.dictionaries import (
        load_payers,
        load_lab_tests,
        get_dictionary_status,
    )
    status = get_dictionary_status()
    print(f"  Bundled: {status['bundled']}")
    print(f"  Downloaded: {status['downloaded']}")
    return True


# =============================================================================
# TYPES & CONFIG TESTS
# =============================================================================

@test("EntityType Enum")
def test_entity_types():
    from privplay.types import EntityType
    
    # Check key types exist
    required = ['SSN', 'EMAIL', 'PHONE', 'NAME_PERSON', 'MRN', 'DATE', 'ADDRESS']
    for t in required:
        assert hasattr(EntityType, t), f"Missing EntityType.{t}"
    
    print(f"  Total entity types: {len(EntityType)}")
    print(f"  Sample: {', '.join(required)}")
    return True

@test("Configuration")
def test_config():
    from privplay.config import get_config, Config
    
    config = get_config()
    print(f"  Data dir: {config.data_dir}")
    print(f"  DB path: {config.db_path}")
    print(f"  Presidio enabled: {config.presidio.enabled}")
    
    # Test custom config
    custom = Config(data_dir=Path("/tmp/test_privplay"))
    assert custom.data_dir == Path("/tmp/test_privplay")
    print("  Custom config works")
    return True


# =============================================================================
# DATABASE TESTS
# =============================================================================

@test("Database Creation")
def test_db_creation():
    from privplay.db import Database
    from privplay.types import Document
    import tempfile
    import uuid
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        
        # Add document
        doc = Document(
            id=str(uuid.uuid4()),
            content="Test document with SSN 123-45-6789",
            source="test",
        )
        db.add_document(doc)
        
        # Retrieve
        retrieved = db.get_document(doc.id)
        assert retrieved is not None
        assert retrieved.content == doc.content
        
        print(f"  Created DB at {db_path}")
        print(f"  Added and retrieved document")
    return True

@test("Database Entities & Corrections")
def test_db_entities():
    from privplay.db import Database
    from privplay.types import Document, Entity, EntityType, SourceType, DecisionType
    import tempfile
    import uuid
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Database(Path(tmpdir) / "test.db")
        
        # Add document
        doc_id = str(uuid.uuid4())
        db.add_document(Document(id=doc_id, content="Test", source="test"))
        
        # Add entity
        entity = Entity(
            text="123-45-6789",
            start=0,
            end=11,
            entity_type=EntityType.SSN,
            confidence=0.95,
            source=SourceType.RULE,
        )
        db.add_entity(entity, doc_id)
        
        # Get entities
        entities = db.get_entities_for_document(doc_id)
        assert len(entities) == 1
        print(f"  Added entity: {entities[0].text} ({entities[0].entity_type.value})")
        
        # Add correction
        from privplay.types import Correction
        import uuid
        
        correction = Correction(
            id=str(uuid.uuid4()),
            entity_id=entity.id,
            document_id=doc_id,
            entity_text=entity.text,
            entity_start=entity.start,
            entity_end=entity.end,
            detected_type=entity.entity_type,
            decision=DecisionType.CONFIRMED,
            context_before="",
            context_after="",
            ner_confidence=entity.confidence,
        )
        db.add_correction(correction)
        
        corrections = db.get_corrections()
        assert len(corrections) == 1
        print(f"  Added correction: {corrections[0].decision.value}")
    return True


# =============================================================================
# RULE ENGINE TESTS
# =============================================================================

@test("Rule Engine - Built-in Rules")
def test_rule_engine_builtin():
    from privplay.engine.rules.engine import RuleEngine
    
    engine = RuleEngine()
    print(f"  Loaded {len(engine.rules)} built-in rules")
    
    # List some rules
    rule_names = [r.name for r in engine.rules[:5]]
    print(f"  Sample rules: {rule_names}")
    return True

@test("Rule Engine - SSN Detection")
def test_rule_engine_ssn():
    from privplay.engine.rules.engine import RuleEngine
    
    engine = RuleEngine()
    
    text = "Patient SSN is 123-45-6789 and phone is 555-123-4567"
    entities = engine.detect(text)
    
    ssn_found = any(e.entity_type.value == "SSN" for e in entities)
    phone_found = any(e.entity_type.value == "PHONE" for e in entities)
    
    print(f"  Text: {text}")
    print(f"  Found {len(entities)} entities")
    for e in entities:
        print(f"    - {e.text}: {e.entity_type.value} ({e.confidence:.0%})")
    
    assert ssn_found, "SSN not detected"
    assert phone_found, "Phone not detected"
    return True

@test("Rule Engine - Email & Date Detection")
def test_rule_engine_email_date():
    from privplay.engine.rules.engine import RuleEngine
    
    engine = RuleEngine()
    
    text = "Contact john@example.com by 03/15/2024 regarding appointment"
    entities = engine.detect(text)
    
    email_found = any(e.entity_type.value == "EMAIL" for e in entities)
    date_found = any(e.entity_type.value == "DATE" for e in entities)
    
    print(f"  Text: {text}")
    for e in entities:
        print(f"    - {e.text}: {e.entity_type.value}")
    
    assert email_found, "Email not detected"
    assert date_found, "Date not detected"
    return True

@test("Rule Engine - HIPAA Identifiers")
def test_rule_engine_hipaa():
    from privplay.engine.rules.engine import RuleEngine
    
    engine = RuleEngine()
    
    text = """
    MRN: 12345678
    DEA: AB1234567
    NPI: 1234567890
    Member ID: ABC123456789
    """
    
    entities = engine.detect(text)
    types_found = set(e.entity_type.value for e in entities)
    
    print(f"  Types detected: {types_found}")
    for e in entities:
        print(f"    - {e.text}: {e.entity_type.value}")
    
    assert "MRN" in types_found, "MRN not detected"
    assert "DEA_NUMBER" in types_found, "DEA not detected"
    return True


# =============================================================================
# CUSTOM RULES TESTS
# =============================================================================

@test("Custom Rule Manager")
def test_custom_rules():
    from privplay.training.rules import RuleManager, CustomRule
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = Path(tmpdir) / "custom_rules.json"
        manager = RuleManager(rules_path)
        
        # Add a custom rule
        rule = CustomRule(
            name="company_id",
            pattern=r'\bCO\d{6}\b',
            entity_type="ACCOUNT_NUMBER",
            confidence=0.90,
            description="Company ID format",
        )
        
        success = manager.add_rule(rule)
        assert success, "Failed to add rule"
        print(f"  Added rule: {rule.name}")
        
        # List rules
        rules = manager.list_rules()
        assert len(rules) == 1
        print(f"  Total rules: {len(rules)}")
        
        # Test pattern
        matches = manager.test_pattern(r'\bCO\d{6}\b', "Order CO123456 confirmed")
        assert len(matches) == 1
        print(f"  Pattern test found: {matches[0]['text']}")
        
        # Remove rule
        manager.remove_rule("company_id")
        assert len(manager.list_rules()) == 0
        print("  Rule removed successfully")
    return True


# =============================================================================
# FAKER / SYNTHETIC DATA TESTS
# =============================================================================

@test("Faker Document Generation")
def test_faker_generation():
    from privplay.training.faker_gen import generate_documents, generate_simple_pii_examples
    
    # Generate clinical docs
    docs = generate_documents(5)
    assert len(docs) == 5
    print(f"  Generated {len(docs)} clinical documents")
    print(f"  Sample length: {len(docs[0].content)} chars")
    
    # Generate simple PII
    simple = generate_simple_pii_examples(3)
    assert len(simple) == 3
    print(f"  Generated {len(simple)} simple PII examples")
    
    # Check content has PHI-like patterns
    sample = docs[0].content
    has_date = "/" in sample or "-" in sample
    print(f"  Sample has date-like patterns: {has_date}")
    return True


# =============================================================================
# BENCHMARK DATASET TESTS
# =============================================================================

@test("Synthetic PHI Dataset")
def test_synthetic_phi_dataset():
    from privplay.benchmark.datasets import load_synthetic_phi_dataset
    
    dataset = load_synthetic_phi_dataset(num_samples=10, seed=42)
    
    print(f"  Name: {dataset.name}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Entity types: {len(dataset.entity_types)}")
    
    # Check first sample
    sample = dataset.samples[0]
    print(f"  First sample entities: {len(sample.entities)}")
    
    # Verify ground truth
    total_entities = sum(len(s.entities) for s in dataset.samples)
    print(f"  Total ground truth entities: {total_entities}")
    
    assert len(dataset) == 10
    assert total_entities > 0
    return True

@test("List Available Datasets")
def test_list_datasets():
    from privplay.benchmark import list_datasets
    
    datasets = list_datasets()
    print(f"  Available datasets: {len(datasets)}")
    for ds in datasets:
        print(f"    - {ds['name']}: {ds['description'][:50]}...")
    
    assert len(datasets) >= 2  # At least synthetic_phi and ai4privacy
    return True


# =============================================================================
# CLASSIFICATION ENGINE TESTS (uses mock model by default)
# =============================================================================

@test("Classification Engine - Mock Model")
def test_classification_engine_mock():
    from privplay.engine.classifier import ClassificationEngine
    from privplay.config import get_config
    
    config = get_config()
    config.presidio.enabled = False  # Skip presidio for this test
    
    engine = ClassificationEngine(use_mock_model=True, config=config)
    
    text = "Patient John Smith, SSN 123-45-6789, email john@test.com"
    entities = engine.detect(text, verify=False)
    
    print(f"  Text: {text}")
    print(f"  Detected {len(entities)} entities:")
    for e in entities:
        print(f"    - {e.text}: {e.entity_type.value} ({e.source.value})")
    
    # Should at least find SSN and email via rules
    types = set(e.entity_type.value for e in entities)
    assert "SSN" in types or "EMAIL" in types, "No entities detected"
    return True


# =============================================================================
# BENCHMARK RUNNER TESTS
# =============================================================================

@test("Benchmark Runner - Mock")
def test_benchmark_runner_mock():
    from privplay.benchmark import BenchmarkRunner, get_dataset
    from privplay.engine.classifier import ClassificationEngine
    from privplay.config import get_config
    
    # Load small synthetic dataset
    dataset = get_dataset("synthetic_phi", max_samples=5)
    
    # Create engine with mock model
    config = get_config()
    config.presidio.enabled = False
    engine = ClassificationEngine(use_mock_model=True, config=config)
    
    # Run benchmark (no storage)
    runner = BenchmarkRunner(engine, storage=None)
    result = runner.run(dataset, verify=False)
    
    print(f"  Samples: {result.num_samples}")
    print(f"  Precision: {result.precision:.1%}")
    print(f"  Recall: {result.recall:.1%}")
    print(f"  F1: {result.f1:.1%}")
    print(f"  TP/FP/FN: {result.true_positives}/{result.false_positives}/{result.false_negatives}")
    
    return True


# =============================================================================
# BENCHMARK STORAGE TESTS
# =============================================================================

@test("Benchmark Storage")
def test_benchmark_storage():
    from privplay.benchmark import BenchmarkStorage, BenchmarkRunner, get_dataset
    from privplay.engine.classifier import ClassificationEngine
    from privplay.config import get_config
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "benchmarks.db"
        storage = BenchmarkStorage(db_path)
        
        # Create a mock result
        dataset = get_dataset("synthetic_phi", max_samples=3)
        config = get_config()
        config.presidio.enabled = False
        engine = ClassificationEngine(use_mock_model=True, config=config)
        
        runner = BenchmarkRunner(engine, storage=storage)
        result = runner.run(dataset, verify=False)
        
        # Check it was saved
        history = storage.get_history(limit=5)
        assert len(history) >= 1
        print(f"  Saved {len(history)} benchmark runs")
        
        # Retrieve by ID
        run = storage.get_run(history[0].run_id)
        assert run is not None
        print(f"  Retrieved run: {run.run_id[:8]}... F1={run.f1:.1%}")
    return True


# =============================================================================
# DICTIONARIES TESTS
# =============================================================================

@test("Bundled Dictionaries - Payers")
def test_dict_payers():
    from privplay.dictionaries import load_payers
    
    payers = load_payers()
    print(f"  Loaded {len(payers)} payers")
    
    # Check some known payers exist
    sample = list(payers)[:5]
    print(f"  Sample: {sample}")
    
    assert len(payers) > 50
    assert "medicare" in payers or "Medicare" in payers or any("medicare" in p.lower() for p in payers)
    return True

@test("Bundled Dictionaries - Lab Tests")
def test_dict_lab_tests():
    from privplay.dictionaries import load_lab_tests
    
    tests = load_lab_tests()
    print(f"  Loaded {len(tests)} lab tests")
    
    sample = list(tests)[:5]
    print(f"  Sample: {sample}")
    
    assert len(tests) > 100
    return True


# =============================================================================
# FULL WORKFLOW TEST
# =============================================================================

@test("Full HIL Workflow Simulation")
def test_full_workflow():
    """Simulate the complete human-in-loop workflow."""
    from privplay.db import Database
    from privplay.types import Document, DecisionType
    from privplay.training.faker_gen import generate_documents
    from privplay.engine.classifier import ClassificationEngine
    from privplay.config import get_config
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        db = Database(Path(tmpdir) / "workflow.db")
        config = get_config()
        config.presidio.enabled = False
        engine = ClassificationEngine(use_mock_model=True, config=config)
        
        print("  Step 1: Generate documents")
        docs = generate_documents(3)
        for doc in docs:
            db.add_document(doc)
        print(f"    Generated {len(docs)} docs")
        
        # 2. Scan
        print("  Step 2: Scan for entities")
        total_entities = 0
        for doc in db.get_unscanned_documents():
            entities = engine.detect(doc.content, verify=False)
            for entity in entities:
                db.add_entity(entity, doc.id)
                total_entities += 1
            db.mark_document_scanned(doc.id)
        print(f"    Found {total_entities} entities")
        
        # 3. Simulate review (auto-approve high confidence)
        print("  Step 3: Simulate review")
        from privplay.types import Correction
        import uuid as uuid_mod
        
        reviewed = 0
        for doc in docs:
            entities = db.get_entities_for_document(doc.id)
            for entity in entities:
                # Simulate human review - approve if high confidence
                decision = DecisionType.CONFIRMED if entity.confidence > 0.8 else DecisionType.REJECTED
                correction = Correction(
                    id=str(uuid_mod.uuid4()),
                    entity_id=entity.id,
                    document_id=doc.id,
                    entity_text=entity.text,
                    entity_start=entity.start,
                    entity_end=entity.end,
                    detected_type=entity.entity_type,
                    decision=decision,
                    context_before=doc.content[max(0, entity.start-20):entity.start],
                    context_after=doc.content[entity.end:entity.end+20],
                    ner_confidence=entity.confidence,
                )
                db.add_correction(correction)
                reviewed += 1
        print(f"    Reviewed {reviewed} entities")
        
        # 4. Check corrections
        corrections = db.get_corrections()
        confirmed = sum(1 for c in corrections if c.decision == DecisionType.CONFIRMED)
        rejected = sum(1 for c in corrections if c.decision == DecisionType.REJECTED)
        print(f"    Confirmed: {confirmed}, Rejected: {rejected}")
        
        # 5. Export check
        print("  Step 4: Export ready")
        print(f"    {len(corrections)} corrections available for export")
        
    return True


# =============================================================================
# SLOW TESTS (require model download)
# =============================================================================

@test("Classification Engine - Real Model")
def test_classification_engine_real():
    """Test with real transformer model. SLOW - downloads model."""
    from privplay.engine.classifier import ClassificationEngine
    from privplay.config import get_config
    
    config = get_config()
    config.presidio.enabled = False
    
    print("  Loading real model (this may download ~500MB on first run)...")
    engine = ClassificationEngine(use_mock_model=False, config=config)
    
    text = "Patient John Smith (DOB: 03/15/1985) was admitted on 12/01/2024. SSN: 123-45-6789"
    entities = engine.detect(text, verify=False)
    
    print(f"  Text: {text[:60]}...")
    print(f"  Detected {len(entities)} entities:")
    for e in entities[:10]:  # Limit output
        print(f"    - {e.text}: {e.entity_type.value} ({e.source.value}, {e.confidence:.0%})")
    
    assert len(entities) > 0, "Real model detected nothing"
    return True


# =============================================================================
# MAIN
# =============================================================================

VERBOSE = False
QUICK = False

def run_all_tests():
    """Run all registered tests."""
    
    # Collect all test functions
    tests_to_run = [
        # Imports
        test_imports_core,
        test_imports_training,
        test_imports_engine,
        test_imports_benchmark,
        test_imports_dictionaries,
        
        # Types & Config
        test_entity_types,
        test_config,
        
        # Database
        test_db_creation,
        test_db_entities,
        
        # Rule Engine
        test_rule_engine_builtin,
        test_rule_engine_ssn,
        test_rule_engine_email_date,
        test_rule_engine_hipaa,
        
        # Custom Rules
        test_custom_rules,
        
        # Faker
        test_faker_generation,
        
        # Benchmark Datasets
        test_synthetic_phi_dataset,
        test_list_datasets,
        
        # Classification Engine (mock)
        test_classification_engine_mock,
        
        # Benchmark Runner
        test_benchmark_runner_mock,
        test_benchmark_storage,
        
        # Dictionaries
        test_dict_payers,
        test_dict_lab_tests,
        
        # Full Workflow
        test_full_workflow,
    ]
    
    # Add slow tests if not quick mode
    if not QUICK:
        tests_to_run.append(test_classification_engine_real)
    
    # Run tests
    for test_func in tests_to_run:
        test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration for r in results)
    
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.name} ({r.duration:.2f}s)")
        if not r.passed and r.error:
            print(f"      Error: {r.error[:80]}")
    
    print()
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Total time: {total_time:.2f}s")
    
    if failed > 0:
        print("\n⚠️  SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privplay Smoke Tests")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    QUICK = args.quick
    VERBOSE = args.verbose
    
    print("="*60)
    print("PRIVPLAY SMOKE TEST SUITE")
    print("="*60)
    print(f"Mode: {'QUICK' if QUICK else 'FULL'}")
    print(f"Verbose: {VERBOSE}")
    
    run_all_tests()
