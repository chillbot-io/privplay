#!/usr/bin/env python3
"""
Privplay Diagnostic Health Check v2
===================================
Comprehensive test of all system components with verbose output.
Designed to catch real issues, not pass easily.

Run: python3 diagnostic.py
Output: diagnostic_report.txt (and stdout)
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
from io import StringIO
import traceback

# Capture all output
report = StringIO()

def log(msg="", indent=0):
    """Log to both stdout and report."""
    prefix = "  " * indent
    line = f"{prefix}{msg}"
    print(line)
    report.write(line + "\n")

def section(title):
    """Print section header."""
    log()
    log("=" * 60)
    log(f" {title}")
    log("=" * 60)
    log()

def check(name, condition, detail=""):
    """Log a check result."""
    status = "✓ PASS" if condition else "✗ FAIL"
    log(f"[{status}] {name}")
    if detail:
        log(f"         {detail}")
    return condition

def warn(name, detail=""):
    """Log a warning."""
    log(f"[⚠ WARN] {name}")
    if detail:
        log(f"         {detail}")

def main():
    start_time = datetime.now()
    
    log(f"Privplay Diagnostic Report v2")
    log(f"Generated: {start_time.isoformat()}")
    log(f"Python: {sys.version}")
    log(f"CWD: {os.getcwd()}")
    
    passes = 0
    fails = 0
    warnings = 0
    
    # Store config for later use
    config = None
    
    # =========================================================================
    section("1. ENVIRONMENT & IMPORTS")
    # =========================================================================
    
    # Core imports
    try:
        from privplay.types import Entity, EntityType, SourceType
        if check("Core types import", True):
            passes += 1
        
        # Actually verify the enums have expected values
        expected_types = ["SSN", "CREDIT_CARD", "EMAIL", "PHONE"]
        type_names = [e.name for e in EntityType]
        missing = [t for t in expected_types if t not in type_names]
        if missing:
            check("Expected EntityTypes exist", False, f"Missing: {missing}")
            fails += 1
        else:
            check("Expected EntityTypes exist", True, f"Found all: {expected_types}")
            passes += 1
        
        log(f"All EntityTypes: {type_names}")
        
        expected_sources = ["RULE", "PRESIDIO", "MODEL", "MERGED"]
        source_names = [s.name for s in SourceType]
        missing = [s for s in expected_sources if s not in source_names]
        if missing:
            check("Expected SourceTypes exist", False, f"Missing: {missing}")
            fails += 1
        else:
            check("Expected SourceTypes exist", True, f"Found all: {expected_sources}")
            passes += 1
            
        log(f"All SourceTypes: {source_names}")
        
    except Exception as e:
        check("Core types import", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # Config
    try:
        from privplay.config import get_config
        config = get_config()
        if check("Config load", True):
            passes += 1
        log(f"  confidence_threshold: {config.confidence_threshold}")
        log(f"  presidio.enabled: {config.presidio.enabled}")
        log(f"  verification.provider: {config.verification.provider}")
    except Exception as e:
        check("Config load", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # =========================================================================
    section("2. FILE SYSTEM & PATHS")
    # =========================================================================
    
    home_privplay = Path.home() / ".privplay"
    
    paths_to_check = [
        (home_privplay, "~/.privplay directory", True),  # required
        (home_privplay / "dictionaries", "Dictionaries directory", True),
        (home_privplay / "dictionaries" / "drugs.txt", "Drugs dictionary", True),
        (home_privplay / "dictionaries" / "hospitals.txt", "Hospitals dictionary", False),  # optional - CMS broken
        (home_privplay / "npi", "NPI directory", False),  # optional
        (home_privplay / "npi" / "npi.db", "NPI database", False),  # optional
        (home_privplay / "models", "Models directory", True),
        (home_privplay / "privplay.db", "Main database", True),
    ]
    
    for path, name, required in paths_to_check:
        exists = path.exists()
        if exists:
            if path.is_file():
                size = path.stat().st_size
                detail = f"{path} ({size:,} bytes)"
            else:
                try:
                    count = len(list(path.iterdir()))
                    detail = f"{path} ({count} items)"
                except:
                    detail = str(path)
            if check(name, True, detail):
                passes += 1
        else:
            if required:
                check(name, False, f"{path} NOT FOUND")
                fails += 1
            else:
                warn(name, f"{path} not found (optional)")
                warnings += 1
    
    # =========================================================================
    section("3. DICTIONARY STATUS")
    # =========================================================================
    
    try:
        from privplay.dictionaries import get_download_status
        
        dl_status = get_download_status()
        log("Download status:")
        for name, available in dl_status.items():
            status = "✓" if available else "✗"
            log(f"  {status} {name}: {available}")
        
        # Verify drugs is actually downloaded (critical for drug name detection)
        if dl_status.get('drugs', False):
            check("Drugs dictionary downloaded", True)
            passes += 1
        else:
            check("Drugs dictionary downloaded", False, "Run: download_fda_drugs()")
            fails += 1
        
    except ImportError as e:
        check("Dictionary module import", False, str(e))
        fails += 1
    except Exception as e:
        check("Dictionary status", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # Try to get actual counts
    try:
        from privplay.dictionaries import get_dictionary_status
        dict_status = get_dictionary_status()
        log()
        log("Dictionary entry counts:")
        for name, count in dict_status.items():
            log(f"  {name}: {count:,} entries")
            if name == "drugs" and count < 1000:
                warn(f"Drugs dictionary seems small", f"Only {count} entries")
                warnings += 1
    except Exception as e:
        warn("Could not get dictionary counts", str(e))
        warnings += 1
    
    # =========================================================================
    section("4. TRANSFORMER MODEL")
    # =========================================================================
    
    try:
        from privplay.engine.models.transformer import get_model
        
        log("Loading transformer model...")
        model = get_model()
        
        model_name = getattr(model, 'name', getattr(model, 'model_name', 'unknown'))
        if check("Model load", True, f"Model: {model_name}"):
            passes += 1
        
        # Test detection with KNOWN expected output
        test_text = "Patient John Smith was born on 01/15/1980"
        results = model.detect(test_text)
        
        log(f"Test input: '{test_text}'")
        log(f"Raw detections ({len(results)}):")
        for r in results:
            etype = r.entity_type.name if hasattr(r.entity_type, 'name') else str(r.entity_type)
            log(f"  - {etype}: '{r.text}' @ [{r.start}:{r.end}] (conf: {r.confidence:.3f})")
        
        # Verify we found SOMETHING in the name area (even if typed as OTHER)
        name_area_hits = [r for r in results if r.start < 25]  # "John Smith" is before position 25
        if name_area_hits:
            check("Model detects entities in name region", True, f"Found {len(name_area_hits)} entities")
            passes += 1
        else:
            check("Model detects entities in name region", False, "No entities found near 'John Smith'")
            fails += 1
        
        # Check if date was found
        date_hits = [r for r in results if "1980" in r.text or "01/15" in r.text]
        if date_hits:
            check("Model detects date", True)
            passes += 1
        else:
            warn("Model did not detect date", "May be expected depending on model")
            warnings += 1
            
    except ImportError as e:
        check("Transformer import", False, str(e))
        fails += 1
    except Exception as e:
        check("Transformer model", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # =========================================================================
    section("5. PRESIDIO DETECTOR")
    # =========================================================================
    
    try:
        from privplay.engine.models.presidio_detector import get_presidio_detector
        
        log("Loading Presidio...")
        presidio = get_presidio_detector()
        
        # Check if it loaded properly
        if hasattr(presidio, 'load_error') and presidio.load_error:
            check("Presidio load", False, f"Load error: {presidio.load_error}")
            fails += 1
        elif hasattr(presidio, 'is_available'):
            if presidio.is_available():
                check("Presidio load", True)
                passes += 1
            else:
                check("Presidio load", False, "is_available() returned False")
                fails += 1
        else:
            check("Presidio load", True, "(no is_available method, assuming OK)")
            passes += 1
        
        # Test with SPECIFIC expected outputs
        test_text = "Email: john.doe@example.com SSN: 123-45-6789 Phone: 555-123-4567"
        results = presidio.detect(test_text)
        
        log(f"Test input: '{test_text}'")
        log(f"Detections ({len(results)}):")
        for r in results:
            etype = r.entity_type.name if hasattr(r.entity_type, 'name') else str(r.entity_type)
            log(f"  - {etype}: '{r.text}' (conf: {r.confidence:.2f})")
        
        # Check for specific entities
        texts_found = [r.text for r in results]
        types_found = [r.entity_type.name if hasattr(r.entity_type, 'name') else str(r.entity_type) for r in results]
        
        # Email check
        email_found = any("john.doe@example.com" in t for t in texts_found)
        if check("Presidio finds email", email_found, f"Looking for john.doe@example.com"):
            passes += 1
        else:
            fails += 1
        
        # SSN check  
        ssn_found = any("123-45-6789" in t for t in texts_found)
        if check("Presidio finds SSN", ssn_found, f"Looking for 123-45-6789"):
            passes += 1
        else:
            fails += 1
            
    except ImportError as e:
        check("Presidio import", False, str(e))
        fails += 1
    except Exception as e:
        check("Presidio detector", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # =========================================================================
    section("6. RULES ENGINE")
    # =========================================================================
    
    try:
        from privplay.engine.rules.engine import RuleEngine
        
        rules = RuleEngine()
        
        # Get rule count - we know it's self.rules from code inspection
        rule_count = len(rules.rules) if hasattr(rules, 'rules') else 0
        if rule_count > 0:
            check("Rules engine load", True, f"Rule count: {rule_count}")
            passes += 1
        else:
            check("Rules engine load", False, "No rules loaded!")
            fails += 1
        
        # Test SPECIFIC patterns with EXACT expected outputs
        test_cases = [
            ("Credit card (Visa)", "4111111111111111", "CREDIT_CARD"),
            ("Credit card (MC)", "5500000000000004", "CREDIT_CARD"),
            ("Credit card (16-digit)", "6381973478101820", "CREDIT_CARD"),  # The one from ai4privacy
            ("SSN (dashed)", "123-45-6789", "SSN"),
            ("SSN (no dash)", "123456789", "SSN"),
            ("Email", "test@example.com", "EMAIL"),
            ("Phone (parens)", "(555) 123-4567", "PHONE"),
            ("Phone (dashed)", "555-123-4567", "PHONE"),
            ("IP Address", "192.168.1.100", "IP_ADDRESS"),
            ("MAC Address", "00:1A:2B:3C:4D:5E", "MAC_ADDRESS"),
        ]
        
        log()
        log("Rule detection tests (CRITICAL):")
        for name, test_value, expected_type in test_cases:
            results = rules.detect(f"Value: {test_value}")
            
            # Get types found
            types_found = []
            for r in results:
                if hasattr(r.entity_type, 'name'):
                    types_found.append(r.entity_type.name)
                elif hasattr(r.entity_type, 'value'):
                    types_found.append(r.entity_type.value)
                else:
                    types_found.append(str(r.entity_type))
            
            # Exact match required
            found = expected_type in types_found
            status = "✓" if found else "✗"
            log(f"  {status} {name}: '{test_value}' -> {types_found} (expected: {expected_type})")
            
            if found:
                passes += 1
            else:
                fails += 1
                # Show what we got for debugging
                for r in results:
                    log(f"      Got: {r.entity_type} for '{r.text}'")
                if not results:
                    log(f"      NO RESULTS - rule not matching!")
                
    except ImportError as e:
        check("Rules engine import", False, str(e))
        fails += 1
    except Exception as e:
        check("Rules engine", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # =========================================================================
    section("7. CLASSIFICATION ENGINE - MERGE LOGIC (CRITICAL)")
    # =========================================================================
    
    try:
        from privplay.engine.classifier import ClassificationEngine
        from privplay.types import SourceType, EntityType
        
        log("Loading classification engine...")
        engine = ClassificationEngine()
        
        if check("Engine load", True):
            passes += 1
        
        # Get and display stack status
        if hasattr(engine, 'get_stack_status'):
            status = engine.get_stack_status()
            log("Stack status:")
            for component, info in status.items():
                log(f"  {component}: {info}")
        log()
        
        # =====================================================================
        # CRITICAL TEST 1: Credit card should be CREDIT_CARD, not OTHER
        # This tests the tiered merge logic fix
        # =====================================================================
        log("=" * 40)
        log("CRITICAL TEST 1: Tiered merge - CC type")
        log("=" * 40)
        
        test_text = "My credit card number is 6381973478101820"
        results = engine.detect(test_text, verify=False)
        
        log(f"Input: '{test_text}'")
        log(f"Results:")
        for r in results:
            etype = r.entity_type.name if hasattr(r.entity_type, 'name') else str(r.entity_type)
            source = r.source.name if hasattr(r.source, 'name') else str(r.source)
            log(f"  - {etype}: '{r.text}' (conf: {r.confidence:.3f}, source: {source})")
        
        cc_results = [r for r in results if "6381973478101820" in r.text]
        if cc_results:
            cc = cc_results[0]
            etype = cc.entity_type.name if hasattr(cc.entity_type, 'name') else str(cc.entity_type)
            source = cc.source.name if hasattr(cc.source, 'name') else str(cc.source)
            
            # Must be CREDIT_CARD
            type_correct = etype == "CREDIT_CARD"
            if check("CC typed as CREDIT_CARD (not OTHER)", type_correct, f"Got type: {etype}"):
                passes += 1
            else:
                fails += 1
                log("*** MERGE BUG: Transformer's OTHER is overriding Rules' CREDIT_CARD ***")
            
            # Should come from RULE or MERGED (with RULE winning)
            source_correct = source in ["RULE", "MERGED"]
            if check("CC source is RULE or MERGED", source_correct, f"Got source: {source}"):
                passes += 1
            else:
                fails += 1
                
            # Confidence should be >= 0.9 (rules give 0.95, merge might boost)
            conf_ok = cc.confidence >= 0.9
            if check("CC confidence >= 0.9", conf_ok, f"Got: {cc.confidence:.3f}"):
                passes += 1
            else:
                warnings += 1
        else:
            check("Credit card detected at all", False, "No CC found in results!")
            fails += 1
        
        log()
        
        # =====================================================================
        # CRITICAL TEST 2: Multiple entities, check all get correct types
        # =====================================================================
        log("=" * 40)
        log("CRITICAL TEST 2: Multi-entity detection")
        log("=" * 40)
        
        test_text = "SSN: 123-45-6789, Email: test@test.com, Card: 4111111111111111"
        results = engine.detect(test_text, verify=False)
        
        log(f"Input: '{test_text}'")
        log(f"Results ({len(results)}):")
        for r in results:
            etype = r.entity_type.name if hasattr(r.entity_type, 'name') else str(r.entity_type)
            source = r.source.name if hasattr(r.source, 'name') else str(r.source)
            log(f"  - {etype}: '{r.text}' (source: {source})")
        
        # Check each expected entity - use .name for enum comparison
        def get_type_name(r):
            if hasattr(r.entity_type, 'name'):
                return r.entity_type.name
            return str(r.entity_type)
        
        found_ssn = any(r for r in results if "123-45-6789" in r.text and get_type_name(r) == "SSN")
        found_email = any(r for r in results if "test@test.com" in r.text and get_type_name(r) == "EMAIL")
        found_cc = any(r for r in results if "4111111111111111" in r.text and get_type_name(r) == "CREDIT_CARD")
        
        if check("SSN detected with correct type", found_ssn):
            passes += 1
        else:
            fails += 1
            
        if check("Email detected with correct type", found_email):
            passes += 1
        else:
            fails += 1
            
        if check("Credit card detected with correct type", found_cc):
            passes += 1
        else:
            fails += 1
            
    except ImportError as e:
        check("Classification engine import", False, str(e))
        fails += 1
    except Exception as e:
        check("Classification engine", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # =========================================================================
    section("8. LLM VERIFIER (Optional)")
    # =========================================================================
    
    try:
        from privplay.verification.verifier import get_verifier
        
        verifier = get_verifier()
        available = verifier.is_available() if hasattr(verifier, 'is_available') else False
        
        provider = config.verification.provider if config else "unknown"
        
        if available:
            check("LLM Verifier available", True, f"Provider: {provider}")
            passes += 1
        else:
            warn("LLM Verifier not available", f"Provider: {provider} (Ollama not running?)")
            warnings += 1
            log("  This is OK - detection works without verification")
            
    except Exception as e:
        warn("LLM Verifier", str(e))
        warnings += 1
    
    # =========================================================================
    section("9. DATABASE")
    # =========================================================================
    
    try:
        from privplay.db import get_db
        
        db = get_db()
        
        db_path = home_privplay / "privplay.db"
        if db_path.exists():
            size = db_path.stat().st_size
            check("Database file exists", True, f"{size:,} bytes")
            passes += 1
        else:
            check("Database file exists", False, str(db_path))
            fails += 1
        
        # Try to introspect the database
        log()
        log("Database introspection:")
        
        # Check what methods are available
        db_methods = [m for m in dir(db) if not m.startswith('_') and callable(getattr(db, m, None))]
        log(f"  Available methods: {db_methods[:15]}...")
        
        # Try common method names
        for method_name in ['get_stats', 'stats', 'get_correction_count', 'count_corrections']:
            if hasattr(db, method_name):
                try:
                    result = getattr(db, method_name)()
                    log(f"  {method_name}(): {result}")
                except Exception as e:
                    log(f"  {method_name}(): ERROR - {e}")
                break
                
    except Exception as e:
        check("Database", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # =========================================================================
    section("10. BENCHMARK PIPELINE")
    # =========================================================================
    
    try:
        from privplay.benchmark.datasets import get_dataset, list_datasets
        from privplay.benchmark.runner import BenchmarkRunner
        from privplay.engine.classifier import ClassificationEngine
        
        log("Available datasets:")
        for ds in list_datasets():
            log(f"  - {ds['name']}: {ds['description'][:60]}...")
        
        log()
        log("Loading ai4privacy (10 samples for quick test)...")
        dataset = get_dataset("ai4privacy", max_samples=10)
        
        if len(dataset) > 0:
            check("Dataset load", True, f"Loaded {len(dataset)} samples")
            passes += 1
        else:
            check("Dataset load", False, "No samples loaded")
            fails += 1
        
        # Show sample structure
        if len(dataset) > 0:
            sample = dataset.samples[0]
            log(f"Sample structure:")
            log(f"  id: {sample.id}")
            log(f"  text length: {len(sample.text)}")
            log(f"  entities: {len(sample.entities)}")
            if sample.entities:
                e = sample.entities[0]
                log(f"  first entity: {e.entity_type} -> {e.normalized_type} ('{e.text[:30]}...')")
        
        # Run mini benchmark
        log()
        log("Running benchmark on 10 samples...")
        engine = ClassificationEngine()
        runner = BenchmarkRunner(engine)
        result = runner.run(dataset, verify=False, show_progress=False)
        
        log(f"Results:")
        log(f"  Precision: {result.precision:.1%}")
        log(f"  Recall: {result.recall:.1%}")
        log(f"  F1: {result.f1:.1%}")
        log(f"  TP: {result.true_positives}, FP: {result.false_positives}, FN: {result.false_negatives}")
        
        # Sanity check - should have SOME true positives on 10 samples
        if result.true_positives > 0:
            check("Benchmark produces TPs", True, f"{result.true_positives} true positives")
            passes += 1
        else:
            check("Benchmark produces TPs", False, "Zero true positives is suspicious")
            fails += 1
        
        # Show per-entity breakdown if available
        if result.by_entity_type:
            log()
            log("Per-entity breakdown:")
            for etype, metrics in sorted(result.by_entity_type.items(), key=lambda x: -x[1].get('tp', 0))[:10]:
                tp = metrics.get('tp', 0)
                fp = metrics.get('fp', 0)
                fn = metrics.get('fn', 0)
                f1 = metrics.get('f1', 0)
                log(f"  {etype}: F1={f1:.0%} (TP={tp}, FP={fp}, FN={fn})")
            
    except ImportError as e:
        check("Benchmark import", False, str(e))
        fails += 1
    except Exception as e:
        check("Benchmark pipeline", False, str(e))
        fails += 1
        log(traceback.format_exc())
    
    # =========================================================================
    section("11. ENTITY LABEL MAPPING")
    # =========================================================================
    
    try:
        from privplay.benchmark.datasets import AI4PRIVACY_MAPPING, _normalize_label
        
        log("Checking ai4privacy -> internal label mapping:")
        
        critical_mappings = [
            ("CREDITCARDNUMBER", "CREDIT_CARD"),
            ("SSN", "SSN"),
            ("EMAIL", "EMAIL"),
            ("PHONENUMBER", "PHONE"),
            ("IPV4", "IP_ADDRESS"),
            ("DRIVERLICENSE", "DRIVER_LICENSE"),
        ]
        
        for source, expected in critical_mappings:
            actual = AI4PRIVACY_MAPPING.get(source, "NOT_FOUND")
            matches = actual == expected
            status = "✓" if matches else "✗"
            log(f"  {status} {source} -> {actual} (expected: {expected})")
            if matches:
                passes += 1
            else:
                fails += 1
        
        # Test the normalize function
        log()
        log("Testing _normalize_label():")
        test_labels = ["CREDITCARDNUMBER", "B-CREDITCARDNUMBER", "I-SSN", "UNKNOWN_LABEL"]
        for label in test_labels:
            result = _normalize_label(label)
            log(f"  '{label}' -> '{result}'")
            
    except ImportError as e:
        warn("Label mapping import", str(e))
        warnings += 1
    except Exception as e:
        warn("Label mapping check", str(e))
        warnings += 1
    
    # =========================================================================
    section("12. TRAINING PIPELINE")
    # =========================================================================
    
    try:
        from privplay.db import get_db
        
        db = get_db()
        log("Checking training data infrastructure:")
        
        # List all methods to find the right ones
        methods = [m for m in dir(db) if 'correct' in m.lower() or 'train' in m.lower() or 'export' in m.lower()]
        log(f"  Relevant methods found: {methods}")
        
        # get_corrections() takes no arguments
        try:
            corrections = db.get_corrections()
            correction_count = len(corrections)
            log(f"  Corrections in DB: {correction_count}")
            check("Can access corrections", True, f"{correction_count} corrections")
            passes += 1
        except Exception as e:
            warn("Could not get corrections", str(e))
            warnings += 1
            
    except Exception as e:
        warn("Training pipeline", str(e))
        warnings += 1
    
    # =========================================================================
    section("SUMMARY")
    # =========================================================================
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log(f"Completed in {duration:.1f} seconds")
    log()
    log(f"  ✓ PASSES:   {passes}")
    log(f"  ✗ FAILS:    {fails}")
    log(f"  ⚠ WARNINGS: {warnings}")
    log()
    
    if fails == 0:
        log("🎉 STATUS: ALL CRITICAL CHECKS PASSED")
        log()
        log("Next steps:")
        log("  1. Run full benchmark: phi-train benchmark run ai4privacy -n 500")
        log("  2. Check F1 improvement on CREDIT_CARD, SSN, etc.")
        log("  3. If good, capture training data: phi-train benchmark run ai4privacy -n 50000 --capture-errors")
    else:
        log(f"🚨 STATUS: {fails} CRITICAL ISSUES NEED ATTENTION")
        log()
        log("Review the FAIL items above before proceeding.")
    
    log()
    log("=" * 60)
    
    # Write report to file
    report_path = Path("diagnostic_report.txt")
    with open(report_path, "w") as f:
        f.write(report.getvalue())
    
    print(f"\nReport saved to: {report_path.absolute()}")
    
    return fails

if __name__ == "__main__":
    sys.exit(main())
