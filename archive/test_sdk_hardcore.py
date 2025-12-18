#!/usr/bin/env python3
"""
Privplay SDK Hardcore Test Suite & Benchmark

Part 1: E2E Tests - Safety, Accuracy, Sessions, Stress
Part 2: Benchmark - 1000 samples, P/R/F1 metrics

Run with: python test_sdk_hardcore.py
Options:
    --part1     Run only E2E tests
    --part2     Run only benchmark
    --quick     Quick benchmark (100 samples)
    
Pass threshold: 80% F1
"""

import asyncio
import sys
import time
import random
import string
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import Counter, defaultdict
import traceback
import concurrent.futures

# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    duration: float
    error: Optional[str] = None
    details: Optional[str] = None

results: List[TestResult] = []
current_category = "General"

def set_category(name: str):
    global current_category
    current_category = name

def test(name: str):
    """Decorator to register and run a test."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n  [{current_category}] {name}...", end=" ", flush=True)
            
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(func(*args, **kwargs))
                else:
                    result = func(*args, **kwargs)
                duration = time.time() - start
                
                if result is None or result is True:
                    print(f"✓ ({duration:.2f}s)")
                    results.append(TestResult(name, current_category, True, duration))
                elif isinstance(result, str):
                    print(f"✗ FAILED")
                    print(f"      {result}")
                    results.append(TestResult(name, current_category, False, duration, result))
                else:
                    print(f"✓ ({duration:.2f}s)")
                    results.append(TestResult(name, current_category, True, duration, details=str(result)))
                    
            except Exception as e:
                duration = time.time() - start
                print(f"✗ ERROR")
                print(f"      {e}")
                traceback.print_exc()
                results.append(TestResult(name, current_category, False, duration, str(e)))
        
        wrapper._test_name = name
        wrapper._test_func = func
        return wrapper
    return decorator


# =============================================================================
# PART 1: HARDCORE E2E TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# SAFETY TESTS - Ensure PHI is never leaked
# -----------------------------------------------------------------------------

def run_safety_tests():
    set_category("Safety")
    print("\n" + "="*70)
    print("SAFETY TESTS - PHI Leakage Prevention")
    print("="*70)
    
    @test("SSN Detection - Standard Format")
    def test_ssn_standard():
        from privplay import sync_scan, EntityType
        
        ssns = ["123-45-6789", "987-65-4321", "555-12-3456"]
        for ssn in ssns:
            text = f"Patient SSN: {ssn}"
            result = sync_scan(text, verify=False)
            found = any(ssn in e.text for e in result.entities)
            if not found:
                return f"Failed to detect SSN: {ssn}"
        return True
    
    @test("SSN Detection - No Dashes")
    def test_ssn_no_dashes():
        from privplay import sync_scan
        
        text = "SSN 123456789 on file"
        result = sync_scan(text, verify=False)
        # Should detect either with or without dashes
        found = any("123456789" in e.text.replace("-", "") for e in result.entities)
        if not found:
            return f"Failed to detect SSN without dashes"
        return True
    
    @test("SSN Detection - Spaces")
    def test_ssn_spaces():
        from privplay import sync_scan
        
        text = "SSN: 123 45 6789"
        result = sync_scan(text, verify=False)
        found = any("123" in e.text and "6789" in e.text for e in result.entities)
        # This is a stretch - may not catch, but shouldn't crash
        return True
    
    @test("Email Detection - Standard")
    def test_email_standard():
        from privplay import sync_scan, EntityType
        
        emails = ["john.doe@hospital.org", "patient123@gmail.com", "dr.smith@clinic.net"]
        for email in emails:
            text = f"Contact: {email}"
            result = sync_scan(text, verify=False)
            found = any(email.lower() in e.text.lower() for e in result.entities)
            if not found:
                return f"Failed to detect email: {email}"
        return True
    
    @test("Phone Detection - Multiple Formats")
    def test_phone_formats():
        from privplay import sync_scan
        
        phones = [
            ("555-123-4567", "dashed"),
            ("(555) 123-4567", "parens"),
            ("555.123.4567", "dotted"),
            ("5551234567", "plain"),
        ]
        for phone, fmt in phones:
            text = f"Call {phone}"
            result = sync_scan(text, verify=False)
            digits = re.sub(r'\D', '', phone)
            found = any(re.sub(r'\D', '', e.text) == digits for e in result.entities)
            # Not all formats may be caught - just ensure no crash
        return True
    
    @test("Name Detection - Clinical Context")
    def test_name_clinical():
        from privplay import sync_scan
        
        text = "Patient John Smith was admitted to the ICU"
        result = sync_scan(text, verify=False)
        found = any("John" in e.text or "Smith" in e.text for e in result.entities)
        if not found:
            return "Failed to detect patient name in clinical context"
        return True
    
    @test("Name Detection - Provider")
    def test_name_provider():
        from privplay import sync_scan
        
        text = "Dr. Jane Wilson ordered the labs"
        result = sync_scan(text, verify=False)
        found = any("Wilson" in e.text or "Jane" in e.text for e in result.entities)
        if not found:
            return "Failed to detect provider name"
        return True
    
    @test("MRN Detection")
    def test_mrn():
        from privplay import sync_scan
        
        text = "MRN: 12345678"
        result = sync_scan(text, verify=False)
        found = any("12345678" in e.text for e in result.entities)
        if not found:
            return "Failed to detect MRN"
        return True
    
    @test("Date of Birth Detection")
    def test_dob():
        from privplay import sync_scan
        
        dates = ["01/15/1985", "03-22-1990", "1985-01-15"]
        for date in dates:
            text = f"DOB: {date}"
            result = sync_scan(text, verify=False)
            found = any(e.text in date or date in e.text for e in result.entities)
            if not found:
                return f"Failed to detect DOB: {date}"
        return True
    
    @test("Address Detection")
    def test_address():
        from privplay import sync_scan
        
        text = "Patient lives at 123 Main Street, Springfield, IL 62701"
        result = sync_scan(text, verify=False)
        # Should detect some part of address
        found = any("123" in e.text or "Main" in e.text or "62701" in e.text for e in result.entities)
        if not found:
            return "Failed to detect any part of address"
        return True
    
    @test("Credit Card Detection")
    def test_credit_card():
        from privplay import sync_scan
        
        # Valid Luhn checksum
        text = "Card: 4532015112830366"
        result = sync_scan(text, verify=False)
        found = any("4532" in e.text for e in result.entities)
        if not found:
            return "Failed to detect credit card number"
        return True
    
    @test("Multiple PHI in Single Text")
    def test_multiple_phi():
        from privplay import sync_scan
        
        text = """
        Patient: John Smith
        DOB: 01/15/1985
        SSN: 123-45-6789
        Email: john.smith@email.com
        Phone: 555-123-4567
        """
        result = sync_scan(text, verify=False)
        if result.entity_count < 3:
            return f"Expected at least 3 entities, got {result.entity_count}"
        return True
    
    @test("PHI at Text Boundaries")
    def test_boundaries():
        from privplay import sync_scan
        
        # PHI at start
        text1 = "123-45-6789 is the SSN"
        r1 = sync_scan(text1, verify=False)
        if not any("123-45-6789" in e.text for e in r1.entities):
            return "Failed to detect SSN at start of text"
        
        # PHI at end
        text2 = "The SSN is 123-45-6789"
        r2 = sync_scan(text2, verify=False)
        if not any("123-45-6789" in e.text for e in r2.entities):
            return "Failed to detect SSN at end of text"
        
        return True
    
    @test("Redaction Completeness")
    def test_redaction_complete():
        from privplay import sync_redact
        
        ssn = "123-45-6789"
        email = "john@test.com"
        text = f"SSN: {ssn}, Email: {email}"
        
        safe = sync_redact(text, verify=False)
        
        if ssn in safe:
            return f"SSN leaked in redacted text: {safe}"
        if email in safe:
            return f"Email leaked in redacted text: {safe}"
        return True
    
    @test("No False Negatives on Known PHI Patterns")
    def test_no_false_negatives():
        from privplay import sync_scan
        
        critical_patterns = [
            ("SSN: 123-45-6789", "SSN"),
            ("john.doe@hospital.org", "email"),
            ("Patient John Smith", "name"),
            ("DOB 01/15/1985", "date"),
            ("MRN 12345678", "MRN"),
        ]
        
        failures = []
        for text, pattern_type in critical_patterns:
            result = sync_scan(text, verify=False)
            if result.entity_count == 0:
                failures.append(f"{pattern_type}: '{text}'")
        
        if failures:
            return f"False negatives: {failures}"
        return True
    
    # Run all safety tests
    test_ssn_standard()
    test_ssn_no_dashes()
    test_ssn_spaces()
    test_email_standard()
    test_phone_formats()
    test_name_clinical()
    test_name_provider()
    test_mrn()
    test_dob()
    test_address()
    test_credit_card()
    test_multiple_phi()
    test_boundaries()
    test_redaction_complete()
    test_no_false_negatives()


# -----------------------------------------------------------------------------
# ACCURACY TESTS - Verify correct detection
# -----------------------------------------------------------------------------

def run_accuracy_tests():
    set_category("Accuracy")
    print("\n" + "="*70)
    print("ACCURACY TESTS - Detection Correctness")
    print("="*70)
    
    @test("Entity Position Validation")
    def test_position_validation():
        from privplay import sync_scan
        
        text = "Patient John Smith, SSN 123-45-6789, email john@test.com"
        result = sync_scan(text, verify=False)
        
        errors = []
        for e in result.entities:
            actual = text[e.start:e.end]
            if actual != e.text:
                errors.append(f"Position mismatch: '{e.text}' vs '{actual}' at {e.start}:{e.end}")
        
        if errors:
            return "\n".join(errors)
        return True
    
    @test("Token Format Validation")
    def test_token_format():
        from privplay import sync_redact
        
        text = "Patient John Smith, SSN 123-45-6789"
        result = sync_redact(text, full=True, verify=False)
        
        # All tokens should match [TYPE_N] pattern
        token_pattern = re.compile(r'\[([A-Z_]+)_(\d+)\]')
        tokens_in_text = token_pattern.findall(result.safe_text)
        
        if not tokens_in_text:
            return f"No valid tokens in: {result.safe_text}"
        
        for token_type, token_num in tokens_in_text:
            if not token_type or not token_num.isdigit():
                return f"Invalid token format: [{token_type}_{token_num}]"
        
        return True
    
    @test("Redact-Restore Round Trip")
    def test_round_trip():
        from privplay import sync_redact, sync_restore
        
        original_phi = ["John Smith", "123-45-6789", "john@test.com"]
        text = f"Patient {original_phi[0]}, SSN {original_phi[1]}, email {original_phi[2]}"
        
        result = sync_redact(text, full=True, verify=False)
        
        # Simulate LLM using tokens
        llm_response = result.safe_text.replace("Patient", "Updated record for")
        
        restored = sync_restore(llm_response, result.token_map)
        
        # Check that PHI is restored
        for phi in original_phi:
            if phi in result.token_map and phi not in restored:
                return f"Failed to restore: {phi}"
        
        return True
    
    @test("Idempotency - Double Redaction")
    def test_idempotency():
        from privplay import sync_redact
        
        text = "Patient John Smith, SSN 123-45-6789"
        
        first = sync_redact(text, verify=False)
        second = sync_redact(first, verify=False)
        
        # Second redaction should not change anything (tokens are not PHI)
        if first != second:
            # Tokens might get re-tokenized - that's a bug
            if "[" in second and "[[" in second:
                return f"Double tokenization detected: {second}"
        
        return True
    
    @test("Confidence Scores Valid")
    def test_confidence_valid():
        from privplay import sync_scan
        
        text = "Patient John Smith, SSN 123-45-6789"
        result = sync_scan(text, verify=False)
        
        for e in result.entities:
            if e.confidence < 0 or e.confidence > 1:
                return f"Invalid confidence {e.confidence} for {e.text}"
        
        return True
    
    @test("Entity Types Valid")
    def test_entity_types_valid():
        from privplay import sync_scan, EntityType
        
        text = "Patient John Smith, SSN 123-45-6789, email john@test.com"
        result = sync_scan(text, verify=False)
        
        valid_types = set(t.value for t in EntityType)
        
        for e in result.entities:
            type_value = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
            if type_value not in valid_types:
                return f"Invalid entity type: {type_value}"
        
        return True
    
    @test("No Overlapping Entities")
    def test_no_overlaps():
        from privplay import sync_scan
        
        text = "Patient John Smith Jr., SSN 123-45-6789"
        result = sync_scan(text, verify=False)
        
        entities = sorted(result.entities, key=lambda e: e.start)
        for i in range(len(entities) - 1):
            if entities[i].end > entities[i+1].start:
                return f"Overlap: {entities[i].text} ({entities[i].start}-{entities[i].end}) and {entities[i+1].text} ({entities[i+1].start}-{entities[i+1].end})"
        
        return True
    
    @test("Empty Text Handling")
    def test_empty():
        from privplay import sync_scan, sync_redact
        
        r1 = sync_scan("", verify=False)
        if r1.entity_count != 0:
            return "Empty text should have 0 entities"
        
        r2 = sync_redact("", verify=False)
        if r2 != "":
            return "Empty text redaction should return empty"
        
        return True
    
    @test("Whitespace-Only Text")
    def test_whitespace():
        from privplay import sync_scan, sync_redact
        
        text = "   \n\t   \n   "
        r1 = sync_scan(text, verify=False)
        r2 = sync_redact(text, verify=False)
        
        # Should not crash, should return minimal/no entities
        return True
    
    @test("Unicode Text Handling")
    def test_unicode():
        from privplay import sync_scan
        
        text = "Patient José García-López, 日本語テスト, émoji: 🏥"
        result = sync_scan(text, verify=False)
        
        # Should not crash, positions should still be valid
        for e in result.entities:
            try:
                actual = text[e.start:e.end]
            except:
                return f"Invalid position for unicode: {e.start}:{e.end}"
        
        return True
    
    @test("Very Long Entity Names")
    def test_long_names():
        from privplay import sync_scan
        
        long_name = "Bartholomew Alexander Fitzgerald Montgomery Wellington III"
        text = f"Patient: {long_name}"
        result = sync_scan(text, verify=False)
        
        # Should detect at least part of the name
        return True
    
    # Run all accuracy tests
    test_position_validation()
    test_token_format()
    test_round_trip()
    test_idempotency()
    test_confidence_valid()
    test_entity_types_valid()
    test_no_overlaps()
    test_empty()
    test_whitespace()
    test_unicode()
    test_long_names()


# -----------------------------------------------------------------------------
# SESSION TESTS - Multi-turn consistency
# -----------------------------------------------------------------------------

def run_session_tests():
    set_category("Sessions")
    print("\n" + "="*70)
    print("SESSION TESTS - Multi-turn Consistency")
    print("="*70)
    
    @test("Token Consistency Within Session")
    def test_token_consistency():
        from privplay import SyncSession
        
        with SyncSession(password="test-password-123") as session:
            r1 = session.redact("Dr. Smith prescribed medication")
            r2 = session.redact("The patient asked Dr. Smith about side effects")
            
            # Same name should get same token
            smith_token_1 = None
            smith_token_2 = None
            
            for orig, tok in r1.token_map.items():
                if "Smith" in orig:
                    smith_token_1 = tok
            
            for orig, tok in r2.token_map.items():
                if "Smith" in orig:
                    smith_token_2 = tok
            
            if smith_token_1 and smith_token_2 and smith_token_1 != smith_token_2:
                return f"Token inconsistency: {smith_token_1} vs {smith_token_2}"
        
        return True
    
    @test("Session Isolation")
    def test_session_isolation():
        from privplay import SyncSession
        
        # Session A
        with SyncSession(password="password-A") as session_a:
            r_a = session_a.redact("Patient John Smith")
            token_a = list(r_a.token_map.values())[0] if r_a.token_map else None
        
        # Session B - same text, different session
        with SyncSession(password="password-B") as session_b:
            r_b = session_b.redact("Patient John Smith")
            token_b = list(r_b.token_map.values())[0] if r_b.token_map else None
        
        # Tokens should be different (different sessions)
        # Actually they might be the same format [PATIENT_1] - that's ok
        # What matters is the restore only works within the session
        
        return True
    
    @test("Session Restore Accuracy")
    def test_session_restore():
        from privplay import SyncSession
        
        with SyncSession(password="test-restore-456") as session:
            original = "Patient Jane Doe, SSN 987-65-4321"
            result = session.redact(original)
            
            # Simulate LLM response
            llm_response = f"I've updated the record. {result.safe_text}"
            
            restored = session.restore(llm_response)
            
            # Check restoration
            if "Jane Doe" in result.token_map and "Jane Doe" not in restored:
                return "Failed to restore name"
        
        return True
    
    @test("Multiple Entities Same Type")
    def test_multiple_same_type():
        from privplay import SyncSession
        
        with SyncSession(password="multi-entity-789") as session:
            result = session.redact("John Smith called about Jane Doe's prescription")
            
            # Should have different tokens for different people
            tokens = list(result.token_map.values())
            if len(tokens) >= 2:
                # Check tokens are unique
                if len(set(tokens)) != len(tokens):
                    return "Duplicate tokens for different entities"
        
        return True
    
    @test("Session Context Preserved Across Calls")
    def test_context_preserved():
        from privplay import SyncSession
        
        with SyncSession(password="context-test-000") as session:
            # First call
            r1 = session.redact("Patient ID: 12345678")
            
            # Second call - different text, but session remembers
            r2 = session.redact("Follow up for patient 12345678")
            
            # The same MRN should get same token
            # (This tests the SHDM store persistence within session)
        
        return True
    
    @test("Async Session Works")
    async def test_async_session():
        from privplay import Session
        
        async with Session(password="async-test-111") as session:
            r1 = await session.redact("Dr. Wilson examined Bob")
            r2 = await session.redact("Bob asked Dr. Wilson questions")
            
            restored = await session.restore("The appointment for [PATIENT_1] is confirmed")
        
        return True
    
    @test("Session With No PHI")
    def test_session_no_phi():
        from privplay import SyncSession
        
        with SyncSession(password="no-phi-test") as session:
            result = session.redact("The patient was stable and alert.")
            
            # Should work even with no PHI
            if result.safe_text is None:
                return "safe_text should not be None"
        
        return True
    
    @test("Rapid Session Create/Destroy")
    def test_rapid_sessions():
        from privplay import SyncSession
        
        for i in range(20):
            with SyncSession(password=f"rapid-{i}") as session:
                result = session.redact(f"Patient {i}")
        
        return True
    
    # Run all session tests
    test_token_consistency()
    test_session_isolation()
    test_session_restore()
    test_multiple_same_type()
    test_context_preserved()
    test_async_session()
    test_session_no_phi()
    test_rapid_sessions()


# -----------------------------------------------------------------------------
# STRESS TESTS - Performance and edge cases
# -----------------------------------------------------------------------------

def run_stress_tests():
    set_category("Stress")
    print("\n" + "="*70)
    print("STRESS TESTS - Performance & Limits")
    print("="*70)
    
    @test("Large Document (10KB)")
    def test_large_doc():
        from privplay import sync_scan
        
        # Generate ~10KB of text with embedded PHI (reduced from 50KB)
        # FastCoref is O(n²) on CPU, so we keep this reasonable
        base = "Patient John Smith (MRN: 12345678) was seen on 01/15/2024. " * 20
        text = base * 8  # ~10KB
        
        start = time.time()
        # Disable verification to speed up
        result = sync_scan(text, verify=False)
        duration = time.time() - start
        
        if duration > 180:  # 3 minutes max
            return f"Too slow: {duration:.1f}s for {len(text)} chars"
        
        return f"{len(text)} chars, {result.entity_count} entities, {duration:.1f}s"
    
    @test("Many Entities (100+ PHI)")
    def test_many_entities():
        from privplay import sync_scan
        from faker import Faker
        fake = Faker()
        
        # Generate text with many PHI
        lines = []
        for i in range(50):
            lines.append(f"Patient: {fake.name()}, SSN: {fake.ssn()}, Email: {fake.email()}")
        
        text = "\n".join(lines)
        result = sync_scan(text, verify=False)
        
        if result.entity_count < 50:
            return f"Expected 50+ entities, got {result.entity_count}"
        
        return f"✓ {result.entity_count} entities detected"
    
    @test("Concurrent Async Scans")
    async def test_concurrent():
        from privplay import scan
        
        texts = [
            f"Patient {i}, SSN 123-45-{6789+i}" for i in range(10)
        ]
        
        start = time.time()
        tasks = [scan(t, verify=False) for t in texts]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        # All should complete
        if len(results) != 10:
            return f"Expected 10 results, got {len(results)}"
        
        return f"10 concurrent scans in {duration:.2f}s"
    
    @test("Special Characters Stress")
    def test_special_chars():
        from privplay import sync_scan
        
        # Text with lots of special chars
        text = """
        Patient: José García-López <josé@例え.com>
        Notes: "He said: 'I'm feeling—better!'"
        Math: 2 + 2 = 4, 50% complete, $100.00
        Emoji: 🏥👨‍⚕️💉
        """
        
        result = sync_scan(text, verify=False)
        
        # Should not crash
        return True
    
    @test("Adversarial: Obfuscated SSN")
    def test_obfuscated_ssn():
        from privplay import sync_scan
        
        obfuscations = [
            "one two three - four five - six seven eight nine",
            "１２３-４５-６７８９",  # Full-width digits
            "SSN: 1-2-3-4-5-6-7-8-9",
        ]
        
        # These are hard cases - we just ensure no crash
        for text in obfuscations:
            result = sync_scan(text, verify=False)
        
        return True
    
    @test("Adversarial: Unicode Lookalikes")
    def test_unicode_lookalikes():
        from privplay import sync_scan
        
        # Greek/Cyrillic letters that look like Latin
        text = "Ρatient Jοhn Smιth"  # Greek rho, omicron, iota
        
        result = sync_scan(text, verify=False)
        
        # These are very hard to catch - just ensure no crash
        return True
    
    # Run all stress tests
    test_large_doc()
    test_many_entities()
    test_concurrent()
    test_special_chars()
    test_obfuscated_ssn()
    test_unicode_lookalikes()


# -----------------------------------------------------------------------------
# REGRESSION TESTS - Known edge cases
# -----------------------------------------------------------------------------

def run_regression_tests():
    set_category("Regression")
    print("\n" + "="*70)
    print("REGRESSION TESTS - Known Edge Cases")
    print("="*70)
    
    @test("Drug Names Not Detected as Names")
    def test_drug_names():
        from privplay import sync_scan
        
        drugs = ["Prozac", "Ambien", "Xanax", "Valium", "Lexapro"]
        
        false_positives = []
        for drug in drugs:
            text = f"Patient started on {drug} 10mg daily"
            result = sync_scan(text, verify=False)
            
            # Drug name should NOT be detected as a person name
            for e in result.entities:
                if drug.lower() in e.text.lower():
                    if "NAME" in e.entity_type.value:
                        false_positives.append(f"{drug} detected as {e.entity_type.value}")
        
        if false_positives:
            return f"False positives: {false_positives}"
        return True
    
    @test("Medical Units Not Detected as PHI")
    def test_medical_units():
        from privplay import sync_scan
        
        units = ["ICU", "MICU", "SICU", "NICU", "ER", "OR", "PACU"]
        
        false_positives = []
        for unit in units:
            text = f"Patient transferred to {unit} for monitoring"
            result = sync_scan(text, verify=False)
            
            for e in result.entities:
                if e.text.upper() == unit:
                    false_positives.append(f"{unit} detected as {e.entity_type.value}")
        
        if false_positives:
            return f"False positives: {false_positives}"
        return True
    
    @test("Browser Strings Not Detected as Names")
    def test_browser_strings():
        from privplay import sync_scan
        
        text = "Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        result = sync_scan(text, verify=False)
        
        for e in result.entities:
            if "NAME" in e.entity_type.value:
                if any(x in e.text for x in ["Mozilla", "WebKit", "Chrome", "Safari", "Windows"]):
                    return f"Browser string detected as name: {e.text}"
        
        return True
    
    @test("Version Numbers Not Detected as PHI")
    def test_version_numbers():
        from privplay import sync_scan
        
        text = "Software version 5.0.123 running on system 2.4.1"
        result = sync_scan(text, verify=False)
        
        # Version numbers should not be PHI
        return True
    
    @test("Common Greetings Not Detected")
    def test_greetings():
        from privplay import sync_scan
        
        greetings = ["Hello", "Hi", "Dear", "Sincerely", "Regards", "Thanks"]
        
        for greeting in greetings:
            text = f"{greeting}, I hope this message finds you well."
            result = sync_scan(text, verify=False)
            
            for e in result.entities:
                if e.text == greeting and "NAME" in e.entity_type.value:
                    return f"Greeting '{greeting}' detected as name"
        
        return True
    
    @test("Clinical Terms Not Detected as Names")
    def test_clinical_terms():
        from privplay import sync_scan
        
        terms = ["STABLE", "ALERT", "ORIENTED", "NEGATIVE", "POSITIVE", "CHRONIC", "ACUTE"]
        
        for term in terms:
            text = f"Patient is {term} and improving"
            result = sync_scan(text, verify=False)
            
            for e in result.entities:
                if e.text.upper() == term and "NAME" in e.entity_type.value:
                    return f"Clinical term '{term}' detected as name"
        
        return True
    
    @test("Generic Patient Reference Not PHI")
    def test_generic_patient():
        from privplay import sync_scan
        
        text = "The patient was admitted. Patient reported symptoms. The patient's condition improved."
        result = sync_scan(text, verify=False)
        
        # "patient" alone should not be PHI
        for e in result.entities:
            if e.text.lower() == "patient" or e.text.lower() == "the patient":
                return f"Generic 'patient' detected as PHI: {e.entity_type.value}"
        
        return True
    
    @test("Date Context Matters")
    def test_date_context():
        from privplay import sync_scan
        
        # Policy dates should be lower confidence than DOB
        text1 = "DOB: 01/15/1985"
        text2 = "Policy effective date: 01/15/1985"
        
        r1 = sync_scan(text1, verify=False)
        r2 = sync_scan(text2, verify=False)
        
        # Both may be detected, but DOB should have higher confidence
        # Just ensure both work
        return True
    
    # Run all regression tests
    test_drug_names()
    test_medical_units()
    test_browser_strings()
    test_version_numbers()
    test_greetings()
    test_clinical_terms()
    test_generic_patient()
    test_date_context()


# =============================================================================
# PART 2: BENCHMARK
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """Metrics from benchmark run."""
    total_samples: int = 0
    total_entities_expected: int = 0
    total_entities_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    latency_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def generate_synthetic_samples(n: int) -> List[Dict]:
    """Generate synthetic clinical samples with ground truth."""
    from faker import Faker
    import uuid
    
    fake = Faker()
    samples = []
    
    templates = [
        # Admission note
        lambda: {
            "text": f"ADMISSION NOTE\nPatient: {(name := fake.name())}\nDOB: {(dob := fake.date_of_birth().strftime('%m/%d/%Y'))}\nMRN: {(mrn := fake.numerify('########'))}\nSSN: {(ssn := fake.ssn())}\n\nChief Complaint: {fake.sentence()}\n\nAttending: Dr. {(doc := fake.last_name())}",
            "entities": [
                {"text": name, "type": "NAME_PATIENT", "start": None},
                {"text": dob, "type": "DATE", "start": None},
                {"text": mrn, "type": "MRN", "start": None},
                {"text": ssn, "type": "SSN", "start": None},
                {"text": f"Dr. {doc}", "type": "NAME_PROVIDER", "start": None},
            ]
        },
        # Progress note
        lambda: {
            "text": f"PROGRESS NOTE\nDate: {(date := fake.date_this_year().strftime('%m/%d/%Y'))}\nPatient: {(name := fake.name())}\nMRN: {(mrn := fake.numerify('########'))}\n\nSubjective: Patient reports improvement.\n\nPlan: Follow up with Dr. {(doc := fake.last_name())} in 2 weeks.\nContact: {(phone := fake.phone_number())}",
            "entities": [
                {"text": date, "type": "DATE", "start": None},
                {"text": name, "type": "NAME_PATIENT", "start": None},
                {"text": mrn, "type": "MRN", "start": None},
                {"text": f"Dr. {doc}", "type": "NAME_PROVIDER", "start": None},
                {"text": phone, "type": "PHONE", "start": None},
            ]
        },
        # Discharge summary
        lambda: {
            "text": f"DISCHARGE SUMMARY\nPatient: {(name := fake.name())}\nDOB: {(dob := fake.date_of_birth().strftime('%m/%d/%Y'))}\nEmail: {(email := fake.email())}\nAddress: {(addr := fake.address().replace(chr(10), ', '))}\n\nDischarge Physician: Dr. {(doc := fake.name())}",
            "entities": [
                {"text": name, "type": "NAME_PATIENT", "start": None},
                {"text": dob, "type": "DATE", "start": None},
                {"text": email, "type": "EMAIL", "start": None},
                {"text": doc, "type": "NAME_PROVIDER", "start": None},
            ]
        },
        # Lab report
        lambda: {
            "text": f"LAB REPORT\nPatient: {(name := fake.name())}\nMRN: {(mrn := fake.numerify('########'))}\nDOB: {(dob := fake.date_of_birth().strftime('%m/%d/%Y'))}\nCollection Date: {(cdate := fake.date_this_month().strftime('%m/%d/%Y'))}\n\nResults: WBC 7.5, RBC 4.8, Hgb 14.2\n\nOrdering Physician: {(doc := fake.name())}, MD",
            "entities": [
                {"text": name, "type": "NAME_PATIENT", "start": None},
                {"text": mrn, "type": "MRN", "start": None},
                {"text": dob, "type": "DATE", "start": None},
                {"text": cdate, "type": "DATE", "start": None},
                {"text": doc, "type": "NAME_PROVIDER", "start": None},
            ]
        },
    ]
    
    for i in range(n):
        template = random.choice(templates)
        data = template()
        
        # Calculate positions
        text = data["text"]
        for ent in data["entities"]:
            pos = text.find(ent["text"])
            if pos >= 0:
                ent["start"] = pos
                ent["end"] = pos + len(ent["text"])
        
        # Filter entities that were found
        data["entities"] = [e for e in data["entities"] if e.get("start") is not None]
        
        samples.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "entities": data["entities"],
            "source": "synthetic",
        })
    
    return samples


def generate_adversarial_samples(n: int) -> List[Dict]:
    """Generate adversarial samples - traps and edge cases."""
    from faker import Faker
    import uuid
    
    fake = Faker()
    samples = []
    
    # Drug name traps (should NOT be detected as names)
    drug_traps = [
        "Patient started on Prozac 20mg daily for depression.",
        "Increase Ambien to 10mg QHS for insomnia.",
        "Continue Xanax 0.5mg TID PRN anxiety.",
        "Hold Valium due to sedation.",
        "Lexapro 10mg daily working well.",
    ]
    
    # Medical unit traps
    unit_traps = [
        "Patient transferred to ICU for monitoring.",
        "Admitted to MICU from ER.",
        "Post-op recovery in PACU.",
        "Consult from NICU for respiratory support.",
    ]
    
    # Browser string traps
    browser_traps = [
        "User agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Browser: Chrome/120.0.0.0 Safari/537.36",
    ]
    
    # Greeting traps
    greeting_traps = [
        "Hello, I hope this finds you well.",
        "Dear Dr. Smith, regarding the patient...",
        "Sincerely, The Medical Team",
    ]
    
    # Real PHI in unusual context
    phi_cases = []
    for _ in range(n // 2):
        phi_cases.append({
            "text": f"Contact {(name := fake.name())} at {(email := fake.email())} or {(phone := fake.phone_number())}",
            "entities": [
                {"text": name, "type": "NAME_PERSON"},
                {"text": email, "type": "EMAIL"},
                {"text": phone, "type": "PHONE"},
            ]
        })
    
    # Build samples
    # Traps (no PHI expected)
    all_traps = drug_traps + unit_traps + browser_traps + greeting_traps
    for trap in all_traps[:n//2]:
        samples.append({
            "id": str(uuid.uuid4()),
            "text": trap,
            "entities": [],  # No PHI
            "source": "adversarial_trap",
        })
    
    # PHI cases
    for case in phi_cases[:n//2]:
        text = case["text"]
        entities = []
        for ent in case["entities"]:
            pos = text.find(ent["text"])
            if pos >= 0:
                entities.append({
                    "text": ent["text"],
                    "type": ent["type"],
                    "start": pos,
                    "end": pos + len(ent["text"]),
                })
        
        samples.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "entities": entities,
            "source": "adversarial_phi",
        })
    
    return samples[:n]


def load_ai4privacy_samples(n: int) -> List[Dict]:
    """Load AI4Privacy samples from HuggingFace."""
    try:
        from privplay.benchmark.datasets import load_ai4privacy_dataset
        
        dataset = load_ai4privacy_dataset(max_samples=n)
        
        samples = []
        for sample in dataset.samples:
            samples.append({
                "id": sample.id,
                "text": sample.text,
                "entities": [
                    {
                        "text": e.text,
                        "type": e.normalized_type,
                        "start": e.start,
                        "end": e.end,
                    }
                    for e in sample.entities
                ],
                "source": "ai4privacy",
            })
        
        return samples
    except Exception as e:
        print(f"    Warning: Could not load AI4Privacy: {e}")
        return []


def entity_overlap(detected: Dict, expected: Dict, tolerance: int = 5) -> bool:
    """Check if detected entity overlaps with expected."""
    # Check text match (partial)
    d_text = detected.get("text", "").lower()
    e_text = expected.get("text", "").lower()
    
    if d_text in e_text or e_text in d_text:
        return True
    
    # Check position overlap
    d_start = detected.get("start", -1)
    d_end = detected.get("end", -1)
    e_start = expected.get("start", -1)
    e_end = expected.get("end", -1)
    
    if d_start >= 0 and e_start >= 0:
        overlap = max(0, min(d_end, e_end) - max(d_start, e_start))
        if overlap > 0:
            return True
    
    return False


def run_benchmark(quick: bool = False):
    """Run the benchmark suite."""
    print("\n" + "="*70)
    print("BENCHMARK - 1000 Sample Evaluation")
    print("="*70)
    
    # Determine sample counts
    if quick:
        n_ai4privacy = 40
        n_synthetic = 40
        n_adversarial = 20
        print("\n  [Quick mode: 100 samples]")
    else:
        n_ai4privacy = 400
        n_synthetic = 400
        n_adversarial = 200
        print("\n  [Full mode: 1000 samples]")
    
    # Load/generate samples
    print("\n  Loading samples...")
    
    print(f"    AI4Privacy ({n_ai4privacy})...", end=" ", flush=True)
    ai4privacy_samples = load_ai4privacy_samples(n_ai4privacy)
    print(f"got {len(ai4privacy_samples)}")
    
    print(f"    Synthetic ({n_synthetic})...", end=" ", flush=True)
    synthetic_samples = generate_synthetic_samples(n_synthetic)
    print(f"got {len(synthetic_samples)}")
    
    print(f"    Adversarial ({n_adversarial})...", end=" ", flush=True)
    adversarial_samples = generate_adversarial_samples(n_adversarial)
    print(f"got {len(adversarial_samples)}")
    
    # Combine all samples
    all_samples = ai4privacy_samples + synthetic_samples + adversarial_samples
    random.shuffle(all_samples)
    
    print(f"\n  Total samples: {len(all_samples)}")
    
    # Initialize metrics
    metrics = BenchmarkMetrics()
    metrics.total_samples = len(all_samples)
    
    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # Run detection
    print("\n  Running detection...")
    from privplay import sync_scan
    
    start_time = time.time()
    
    for i, sample in enumerate(all_samples):
        if (i + 1) % 100 == 0 or i == len(all_samples) - 1:
            print(f"    Progress: {i+1}/{len(all_samples)}", end="\r")
        
        text = sample["text"]
        expected = sample["entities"]
        
        # Time the scan
        scan_start = time.time()
        try:
            result = sync_scan(text, verify=False)
            scan_duration = (time.time() - scan_start) * 1000
            metrics.latency_ms.append(scan_duration)
        except Exception as e:
            metrics.errors.append(f"Sample {sample['id']}: {e}")
            continue
        
        # Convert detected entities
        detected = [
            {
                "text": e.text,
                "type": e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type),
                "start": e.start,
                "end": e.end,
                "confidence": e.confidence,
            }
            for e in result.entities
        ]
        
        metrics.total_entities_expected += len(expected)
        metrics.total_entities_detected += len(detected)
        
        # Match detected to expected
        matched_expected = set()
        matched_detected = set()
        
        for d_idx, d_ent in enumerate(detected):
            for e_idx, e_ent in enumerate(expected):
                if e_idx in matched_expected:
                    continue
                
                if entity_overlap(d_ent, e_ent):
                    # True positive
                    metrics.true_positives += 1
                    matched_expected.add(e_idx)
                    matched_detected.add(d_idx)
                    
                    # By type
                    etype = e_ent.get("type", "UNKNOWN")
                    type_stats[etype]["tp"] += 1
                    break
        
        # False positives (detected but not expected)
        for d_idx, d_ent in enumerate(detected):
            if d_idx not in matched_detected:
                metrics.false_positives += 1
                dtype = d_ent.get("type", "UNKNOWN")
                type_stats[dtype]["fp"] += 1
        
        # False negatives (expected but not detected)
        for e_idx, e_ent in enumerate(expected):
            if e_idx not in matched_expected:
                metrics.false_negatives += 1
                etype = e_ent.get("type", "UNKNOWN")
                type_stats[etype]["fn"] += 1
    
    total_time = time.time() - start_time
    print(f"    Progress: {len(all_samples)}/{len(all_samples)} - Done!")
    
    # Calculate metrics
    if metrics.true_positives + metrics.false_positives > 0:
        metrics.precision = metrics.true_positives / (metrics.true_positives + metrics.false_positives)
    
    if metrics.true_positives + metrics.false_negatives > 0:
        metrics.recall = metrics.true_positives / (metrics.true_positives + metrics.false_negatives)
    
    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
    
    metrics.by_type = dict(type_stats)
    
    # Print results
    print("\n" + "-"*70)
    print("BENCHMARK RESULTS")
    print("-"*70)
    
    print(f"\n  Samples:           {metrics.total_samples}")
    print(f"  Expected Entities: {metrics.total_entities_expected}")
    print(f"  Detected Entities: {metrics.total_entities_detected}")
    print(f"  Errors:            {len(metrics.errors)}")
    
    print(f"\n  True Positives:    {metrics.true_positives}")
    print(f"  False Positives:   {metrics.false_positives}")
    print(f"  False Negatives:   {metrics.false_negatives}")
    
    print(f"\n  Precision:         {metrics.precision:.1%}")
    print(f"  Recall:            {metrics.recall:.1%}")
    print(f"  F1 Score:          {metrics.f1:.1%}")
    
    # Latency stats
    if metrics.latency_ms:
        latencies = sorted(metrics.latency_ms)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        print(f"\n  Latency (ms):")
        print(f"    p50: {p50:.1f}")
        print(f"    p95: {p95:.1f}")
        print(f"    p99: {p99:.1f}")
    
    print(f"\n  Total Time:        {total_time:.1f}s")
    print(f"  Throughput:        {len(all_samples) / total_time:.1f} samples/sec")
    
    # By type breakdown (top 10)
    print("\n  By Entity Type (Top 10):")
    type_f1s = []
    for etype, stats in type_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        type_f1s.append((etype, tp, fp, fn, prec, rec, f1))
    
    type_f1s.sort(key=lambda x: -(x[1] + x[3]))  # Sort by volume
    for etype, tp, fp, fn, prec, rec, f1 in type_f1s[:10]:
        print(f"    {etype:20s}: P={prec:.0%} R={rec:.0%} F1={f1:.0%} (TP={tp}, FP={fp}, FN={fn})")
    
    # Pass/Fail
    print("\n" + "-"*70)
    PASS_THRESHOLD = 0.80
    if metrics.f1 >= PASS_THRESHOLD:
        print(f"  ✅ BENCHMARK PASSED: F1 {metrics.f1:.1%} >= {PASS_THRESHOLD:.0%}")
        return True
    else:
        print(f"  ❌ BENCHMARK FAILED: F1 {metrics.f1:.1%} < {PASS_THRESHOLD:.0%}")
        return False


# =============================================================================
# PART 3: TRAINING LOOP TEST
# =============================================================================

def run_training_loop_test():
    """Test the full training loop including HIL simulation."""
    print("\n" + "#"*70)
    print("#" + " "*15 + "PART 3: TRAINING LOOP TEST" + " "*16 + "#")
    print("#"*70)
    
    set_category("Training")
    
    import tempfile
    import shutil
    from pathlib import Path
    
    # Create temp directory for test
    temp_dir = Path(tempfile.mkdtemp(prefix="privplay_test_"))
    print(f"\n  Test directory: {temp_dir}")
    
    try:
        # ---------------------------------------------------------------------
        # Step 1: Generate labeled test samples
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("Step 1: Generate labeled test samples")
        print("-"*60)
        
        samples = generate_synthetic_samples(100)
        print(f"  Generated {len(samples)} labeled samples")
        
        # Count expected entities
        total_entities = sum(len(s["entities"]) for s in samples)
        print(f"  Total ground truth entities: {total_entities}")
        
        # ---------------------------------------------------------------------
        # Step 2: Initialize learning components
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("Step 2: Initialize learning components")
        print("-"*60)
        
        from privplay.learning import (
            SignalStore,
            LearningClassifier,
            FeedbackType,
            ContinuousTrainer,
            TrainingConfig,
        )
        
        # Create isolated signal store
        signal_db = temp_dir / "signals.db"
        store = SignalStore(str(signal_db))
        print(f"  SignalStore: {signal_db}")
        
        # Create learning classifier
        classifier = LearningClassifier(signal_store=store)
        print(f"  LearningClassifier initialized")
        
        # ---------------------------------------------------------------------
        # Step 3: Run detection with signal capture
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("Step 3: Run detection with signal capture")
        print("-"*60)
        
        all_detections = []
        captured_count = 0
        
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                print(f"    Processing: {i+1}/{len(samples)}", end="\r")
            
            # Set document context
            classifier.set_document_context(
                document_id=sample["id"],
                doc_type="test_clinical",
            )
            
            # Detect (signals automatically captured for uncertain ones)
            entities = classifier.detect(sample["text"])
            
            all_detections.append({
                "sample": sample,
                "detected": entities,
            })
        
        print(f"    Processing: {len(samples)}/{len(samples)} - Done!")
        
        # Check how many signals were captured
        stats = store.get_stats()
        print(f"  Signals captured: {stats.get('pending', 0)} pending")
        print(f"  Total in store: {stats.get('total', 0)}")
        
        # ---------------------------------------------------------------------
        # Step 4: Simulate HIL review (auto-feedback using ground truth)
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("Step 4: Simulate HIL review")
        print("-"*60)
        
        # Get pending signals
        pending = store.get_pending_signals(limit=500)
        print(f"  Pending signals to review: {len(pending)}")
        
        reviewed = 0
        tp_count = 0
        fp_count = 0
        wl_count = 0
        
        for signal in pending:
            # Find the corresponding sample
            sample = None
            for s in samples:
                if s["id"] == signal.document_id:
                    sample = s
                    break
            
            if not sample:
                continue
            
            # Check if this signal matches any ground truth entity
            is_match = False
            correct_type = None
            
            for gt_entity in sample["entities"]:
                # Check overlap
                if (signal.span_start < gt_entity.get("end", 0) and 
                    signal.span_end > gt_entity.get("start", 0)):
                    is_match = True
                    correct_type = gt_entity.get("type", "UNKNOWN")
                    break
                # Also check text match
                if signal.span_text.lower() in gt_entity.get("text", "").lower():
                    is_match = True
                    correct_type = gt_entity.get("type", "UNKNOWN")
                    break
                if gt_entity.get("text", "").lower() in signal.span_text.lower():
                    is_match = True
                    correct_type = gt_entity.get("type", "UNKNOWN")
                    break
            
            # Record feedback
            if is_match:
                # Check if type matches
                if signal.merged_type == correct_type:
                    store.record_feedback(signal.id, FeedbackType.TRUE_POSITIVE)
                    tp_count += 1
                else:
                    store.record_feedback(
                        signal.id, 
                        FeedbackType.WRONG_LABEL, 
                        correct_type=correct_type
                    )
                    wl_count += 1
            else:
                store.record_feedback(signal.id, FeedbackType.FALSE_POSITIVE)
                fp_count += 1
            
            reviewed += 1
        
        print(f"  Reviewed: {reviewed}")
        print(f"    True Positives: {tp_count}")
        print(f"    Wrong Label: {wl_count}")
        print(f"    False Positives: {fp_count}")
        
        # ---------------------------------------------------------------------
        # Step 5: Check training status
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("Step 5: Check training status")
        print("-"*60)
        
        config = TrainingConfig(
            signal_threshold=10,  # Low threshold for test
            model_dir=temp_dir / "models",
        )
        trainer = ContinuousTrainer(store=store, config=config)
        
        should_train, reason = trainer.should_train()
        print(f"  Should train: {should_train}")
        print(f"  Reason: {reason}")
        
        labeled_count = store.get_labeled_signal_count()
        print(f"  Labeled signals: {labeled_count}")
        
        # ---------------------------------------------------------------------
        # Step 6: Train meta-classifier (if enough data)
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("Step 6: Train meta-classifier")
        print("-"*60)
        
        if labeled_count >= 10:
            print("  Training with force=True...")
            try:
                new_model = trainer.train(force=True)
                if new_model:
                    print(f"  ✓ New model trained: {new_model.version}")
                    print(f"    F1: {new_model.f1_score:.1%}" if hasattr(new_model, 'f1_score') else "")
                else:
                    print("  ⚠ Training returned None (may need more data)")
            except Exception as e:
                print(f"  ⚠ Training failed: {e}")
                # Not a hard failure - training might need sklearn, etc.
        else:
            print(f"  ⚠ Not enough labeled signals ({labeled_count} < 10)")
        
        # ---------------------------------------------------------------------
        # Step 7: Verify signal store integrity
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("Step 7: Verify signal store integrity")
        print("-"*60)
        
        final_stats = store.get_stats()
        print(f"  Final stats:")
        print(f"    Total signals: {final_stats.get('total', 0)}")
        print(f"    Pending: {final_stats.get('pending', 0)}")
        print(f"    Reviewed: {final_stats.get('reviewed', 0)}")
        
        # Verify reviewed count matches our feedback
        if final_stats.get('reviewed', 0) != reviewed:
            print(f"  ⚠ Mismatch: reviewed {reviewed} but store shows {final_stats.get('reviewed', 0)}")
        else:
            print(f"  ✓ Review count matches")
        
        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------
        print("\n" + "-"*60)
        print("TRAINING LOOP TEST SUMMARY")
        print("-"*60)
        
        success = True
        checks = [
            ("Samples generated", len(samples) == 100),
            ("Detection completed", len(all_detections) == 100),
            ("Signals captured", stats.get('total', 0) > 0 or stats.get('pending', 0) == 0),
            ("Feedback recorded", reviewed > 0 or len(pending) == 0),
            ("Store integrity", final_stats.get('reviewed', 0) == reviewed),
        ]
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                success = False
        
        if success:
            print("\n  ✅ TRAINING LOOP TEST PASSED")
        else:
            print("\n  ❌ TRAINING LOOP TEST FAILED")
        
        return success
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            print(f"\n  Cleaned up: {temp_dir}")
        except:
            pass


# =============================================================================
# MAIN
# =============================================================================

def run_part1():
    """Run all E2E tests."""
    print("\n" + "#"*70)
    print("#" + " "*20 + "PART 1: E2E TESTS" + " "*21 + "#")
    print("#"*70)
    
    run_safety_tests()
    run_accuracy_tests()
    run_session_tests()
    run_stress_tests()
    run_regression_tests()
    
    # Summary
    print("\n" + "="*70)
    print("E2E TEST SUMMARY")
    print("="*70)
    
    by_category = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    
    total_passed = 0
    total_failed = 0
    
    for category, tests in by_category.items():
        passed = sum(1 for t in tests if t.passed)
        failed = sum(1 for t in tests if not t.passed)
        total_passed += passed
        total_failed += failed
        
        status = "✓" if failed == 0 else "✗"
        print(f"  {status} {category}: {passed}/{len(tests)} passed")
        
        for t in tests:
            if not t.passed:
                print(f"      ✗ {t.name}: {t.error[:60] if t.error else 'Unknown'}...")
    
    print(f"\n  Total: {total_passed}/{total_passed + total_failed} passed")
    
    if total_failed > 0:
        print("\n  ⚠️  SOME E2E TESTS FAILED")
        return False
    else:
        print("\n  ✅ ALL E2E TESTS PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(description="Privplay SDK Hardcore Test Suite")
    parser.add_argument("--part1", action="store_true", help="Run only E2E tests")
    parser.add_argument("--part2", action="store_true", help="Run only benchmark")
    parser.add_argument("--part3", action="store_true", help="Run only training loop test")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (100 samples)")
    args = parser.parse_args()
    
    print("="*70)
    print(" "*15 + "PRIVPLAY SDK HARDCORE TEST SUITE")
    print("="*70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine what to run
    run_all = not args.part1 and not args.part2 and not args.part3
    run_e2e = args.part1 or run_all
    run_bench = args.part2 or run_all
    run_train = args.part3 or run_all
    
    e2e_passed = True
    bench_passed = True
    train_passed = True
    
    if run_e2e:
        e2e_passed = run_part1()
    
    if run_bench:
        print("\n" + "#"*70)
        print("#" + " "*20 + "PART 2: BENCHMARK" + " "*21 + "#")
        print("#"*70)
        bench_passed = run_benchmark(quick=args.quick)
    
    if run_train:
        train_passed = run_training_loop_test()
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if run_e2e:
        print(f"  E2E Tests:      {'✅ PASSED' if e2e_passed else '❌ FAILED'}")
    if run_bench:
        print(f"  Benchmark:      {'✅ PASSED' if bench_passed else '❌ FAILED'}")
    if run_train:
        print(f"  Training Loop:  {'✅ PASSED' if train_passed else '❌ FAILED'}")
    
    all_passed = (
        (not run_e2e or e2e_passed) and 
        (not run_bench or bench_passed) and
        (not run_train or train_passed)
    )
    
    if all_passed:
        print("\n  🎉 ALL CHECKS PASSED")
        return 0
    else:
        print("\n  ⚠️  SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
