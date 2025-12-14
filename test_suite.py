#!/usr/bin/env python3
"""
HARDCORE Test Suite for Privplay PHI/PII Detection
No "acceptable" false positives. No glossing over issues.
"""

import sys
import time
import json
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    critical: bool = False


@dataclass
class SectionResult:
    name: str
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def passed(self): return sum(1 for t in self.tests if t.passed)
    @property
    def failed(self): return sum(1 for t in self.tests if not t.passed)
    @property
    def total(self): return len(self.tests)
    @property
    def has_critical_failure(self): return any(t.critical and not t.passed for t in self.tests)


@dataclass
class SuiteResult:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sections: List[SectionResult] = field(default_factory=list)
    baselines: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_passed(self): return sum(s.passed for s in self.sections)
    @property
    def total_failed(self): return sum(s.failed for s in self.sections)
    @property
    def total_tests(self): return sum(s.total for s in self.sections)
    
    def to_dict(self):
        return {"timestamp": self.timestamp.isoformat(), "passed": self.total_passed, 
                "failed": self.total_failed, "baselines": self.baselines}


class TestRunner:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.current_section = None
        self.suite_result = SuiteResult()
    
    def section(self, name):
        if self.current_section:
            self.suite_result.sections.append(self.current_section)
        self.current_section = SectionResult(name=name)
        print(f"\n{'='*70}\n {name}\n{'='*70}")
    
    def test(self, name, func, critical=False):
        start = time.time()
        try:
            passed, msg, details = func()
        except Exception as e:
            passed, msg, details = False, f"EXCEPTION: {e}", {"error": str(e)}
        
        result = TestResult(name, passed, msg, time.time()-start, details, critical)
        if self.current_section:
            self.current_section.tests.append(result)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        crit = " [CRITICAL]" if critical and not passed else ""
        print(f"  {color}{status}\033[0m {name}{crit}")
        if self.verbose or not passed:
            print(f"         {msg}")
    
    def finalize(self):
        if self.current_section:
            self.suite_result.sections.append(self.current_section)
        return self.suite_result


# SECTION 1: COMPONENT HEALTH
def run_section_1(runner):
    runner.section("SECTION 1: Component Health [CRITICAL]")
    
    def test_imports():
        from privplay.types import EntityType
        required = ["SSN", "EMAIL", "PHONE", "NAME_PERSON", "DRUG", "DIAGNOSIS", "FACILITY"]
        missing = [t for t in required if not hasattr(EntityType, t)]
        if missing: return False, f"Missing: {missing}", {}
        return True, f"Core types OK ({len(EntityType)} types)", {}
    runner.test("Core imports", test_imports, critical=True)
    
    def test_dicts():
        from privplay.dictionaries import get_dictionary_status
        s = get_dictionary_status()
        if not s.get("all_available"): return False, "Missing dictionaries", s
        if s.get("total_terms", 0) < 100000: return False, f"Too few terms: {s.get('total_terms')}", s
        return True, f"Dictionaries OK ({s['total_terms']:,} terms)", s
    runner.test("Dictionaries", test_dicts, critical=True)
    
    def test_rules():
        from privplay.engine.rules.engine import RuleEngine
        e = RuleEngine()
        n = len(e.rules) if hasattr(e, 'rules') else 0
        if n < 10: return False, f"Too few rules: {n}", {}
        return True, f"Rules OK ({n} rules)", {}
    runner.test("Rules engine", test_rules, critical=True)
    
    def test_presidio():
        from privplay.engine.models.presidio_detector import get_presidio_detector
        d = get_presidio_detector(score_threshold=0.7)
        if not d.is_available(): return False, f"Not available: {d.load_error}", {}
        return True, "Presidio OK", {}
    runner.test("Presidio", test_presidio, critical=True)
    
    def test_transformer():
        from privplay.engine.models.transformer import get_model
        m = get_model(use_mock=False)
        m.detect("test")
        return True, f"Transformer OK: {m.name}", {}
    runner.test("Transformer", test_transformer, critical=True)
    
    def test_engine():
        from privplay.engine.classifier import ClassificationEngine
        e = ClassificationEngine()
        return True, "Engine OK", e.get_stack_status()
    runner.test("Classification engine", test_engine, critical=True)


# SECTION 2: TYPE INTEGRITY
def run_section_2(runner):
    runner.section("SECTION 2: Entity Type Integrity")
    
    def test_hospital_mapping():
        from privplay.types import Entity, EntityType, SourceType
        e = Entity(text="X", start=0, end=1, entity_type="HOSPITAL", confidence=0.9, source=SourceType.RULE)
        if e.entity_type != EntityType.FACILITY:
            return False, f"HOSPITAL->FACILITY failed, got {e.entity_type}", {}
        return True, "HOSPITAL->FACILITY OK", {}
    runner.test("HOSPITAL->FACILITY mapping", test_hospital_mapping)
    
    def test_compat_groups():
        from privplay.benchmark.runner import types_are_compatible
        must_match = [("FACILITY","HOSPITAL"),("NAME_PERSON","NAME_PATIENT"),("DATE","DATE_DOB"),("PHONE","FAX")]
        must_not = [("SSN","EMAIL"),("DRUG","NAME_PERSON")]
        fails = [f"{a}/{b}" for a,b in must_match if not types_are_compatible(a,b)]
        wrongs = [f"{a}/{b}" for a,b in must_not if types_are_compatible(a,b)]
        if fails: return False, f"Missing compat: {fails}", {}
        if wrongs: return False, f"Wrong compat: {wrongs}", {}
        return True, "Type compatibility OK", {}
    runner.test("Type compatibility", test_compat_groups, critical=True)


# SECTION 3: DETECTION CORRECTNESS
def run_section_3(runner):
    runner.section("SECTION 3: Detection Correctness")
    
    def test_ssn():
        from privplay.engine.rules.engine import RuleEngine
        from privplay.types import EntityType
        e = RuleEngine()
        found = [x for x in e.detect("SSN: 078-05-1120") if x.entity_type == EntityType.SSN]
        if not found: return False, "SSN not detected", {}
        return True, "SSN detection OK", {}
    runner.test("Rules: SSN", test_ssn)
    
    def test_email():
        from privplay.engine.rules.engine import RuleEngine
        from privplay.types import EntityType
        e = RuleEngine()
        found = [x for x in e.detect("Email: john@test.com") if x.entity_type == EntityType.EMAIL]
        if not found: return False, "Email not detected", {}
        return True, "Email detection OK", {}
    runner.test("Rules: Email", test_email)
    
    def test_names():
        from privplay.engine.models.presidio_detector import get_presidio_detector
        from privplay.types import EntityType
        d = get_presidio_detector(score_threshold=0.5)
        names = [e for e in d.detect("Dr. John Smith examined patient Mary Jones.") 
                 if e.entity_type in [EntityType.NAME_PERSON, EntityType.NAME_PATIENT, EntityType.NAME_PROVIDER]]
        if len(names) < 2: return False, f"Expected 2+ names, got {len(names)}", {}
        return True, f"Name detection OK ({len(names)} found)", {}
    runner.test("Presidio: Names", test_names)
    
    def test_dict_terms():
        from privplay.dictionaries.loader import load_drugs
        drugs = load_drugs()
        expected = ["lisinopril", "metformin", "atorvastatin"]
        missing = [d for d in expected if d not in drugs]
        if missing: return False, f"Missing drugs: {missing}", {}
        return True, f"Dictionary terms OK ({len(drugs)} drugs)", {}
    runner.test("Dictionary: Terms exist", test_dict_terms)
    
    def test_no_garbage():
        from privplay.dictionaries.loader import load_drugs, GLOBAL_BLOCKLIST
        drugs = load_drugs()
        short = [d for d in drugs if len(d) < 4]
        blocked = [d for d in drugs if d in GLOBAL_BLOCKLIST]
        if short: return False, f"Short terms: {short[:5]}", {}
        if blocked: return False, f"Blocked terms: {blocked[:5]}", {}
        return True, "No garbage terms", {}
    runner.test("Dictionary: No garbage", test_no_garbage)


# SECTION 4: INTEGRATION
def run_section_4(runner):
    runner.section("SECTION 4: Integration")
    
    def test_pipeline():
        from privplay.engine.classifier import ClassificationEngine
        from privplay.types import EntityType
        e = ClassificationEngine()
        text = "Patient: John Smith\nSSN: 078-05-1120\nEmail: john@test.com"
        ents = e.detect(text, verify=False)
        types = {x.entity_type for x in ents}
        need = {EntityType.SSN, EntityType.EMAIL}
        has_name = any(x.entity_type in [EntityType.NAME_PERSON, EntityType.NAME_PATIENT] for x in ents)
        if not need.issubset(types) or not has_name:
            return False, f"Missing entities. Found: {[x.entity_type.value for x in ents]}", {}
        return True, f"Pipeline OK ({len(ents)} entities)", {}
    runner.test("Full pipeline", test_pipeline)
    
    def test_empty():
        from privplay.engine.classifier import ClassificationEngine
        e = ClassificationEngine()
        for inp in ["", "   ", "\n"]:
            if e.detect(inp, verify=False) != []:
                return False, f"Empty input '{repr(inp)}' should return []", {}
        return True, "Empty handling OK", {}
    runner.test("Empty input", test_empty)
    
    def test_spans():
        from privplay.engine.classifier import ClassificationEngine
        e = ClassificationEngine()
        text = "Patient John Smith has SSN 078-05-1120."
        for ent in e.detect(text, verify=False):
            if ent.start < 0 or ent.end > len(text): return False, f"Bad bounds: {ent}", {}
            if text[ent.start:ent.end] != ent.text: return False, f"Text mismatch: {ent}", {}
        return True, "Spans OK", {}
    runner.test("Span validation", test_spans)


# SECTION 5: FALSE POSITIVES
def run_section_5(runner):
    runner.section("SECTION 5: False Positive Baseline [STRICT]")
    
    def test_clean_news():
        from privplay.engine.classifier import ClassificationEngine
        e = ClassificationEngine()
        text = """The stock market rallied today as investors reacted to positive economic data.
        The Federal Reserve announced its decision to maintain current interest rates.
        Technology companies led the gains with the sector up three percent overall."""
        ents = e.detect(text, verify=False)
        if ents:
            return False, f"FPs on clean news: {[(x.text, x.entity_type.value) for x in ents]}", {}
        return True, "ZERO FPs on news", {}
    runner.test("Clean news: ZERO FPs", test_clean_news)
    
    def test_tech_docs():
        from privplay.engine.classifier import ClassificationEngine
        from privplay.types import EntityType
        e = ClassificationEngine()
        text = """To install run pip install mypackage. Config options include timeout and retries.
        The API accepts POST requests with JSON payloads."""
        non_url = [x for x in e.detect(text, verify=False) if x.entity_type != EntityType.URL]
        if non_url:
            return False, f"FPs in tech docs: {[(x.text, x.entity_type.value) for x in non_url]}", {}
        return True, "ZERO FPs in tech docs", {}
    runner.test("Tech docs: ZERO FPs", test_tech_docs)
    
    def test_tricky():
        from privplay.engine.classifier import ClassificationEngine
        e = ClassificationEngine()
        cases = [("Call 911 for emergencies","911"),("Version 2.0 released","2.0"),
                 ("I love Dr. Pepper","Pepper"),("Route 66 is famous","66")]
        issues = []
        for text, bad in cases:
            for ent in e.detect(text, verify=False):
                if bad in ent.text: issues.append(f"'{ent.text}' in '{text}'")
        if issues: return False, f"Tricky FPs: {issues}", {}
        return True, f"Tricky text OK ({len(cases)} cases)", {}
    runner.test("Tricky text: No FPs", test_tricky)


# SECTION 6: BENCHMARK
def run_section_6(runner, quick=False):
    runner.section("SECTION 6: Benchmark Readiness")
    
    def test_synthetic():
        from privplay.benchmark import get_dataset, BenchmarkRunner
        from privplay.engine.classifier import ClassificationEngine
        e = ClassificationEngine()
        r = BenchmarkRunner(e, storage=None)
        d = get_dataset("synthetic_phi", max_samples=30)
        res = r.run(d, verify=False, show_progress=False)
        if res.recall < 0.60: return False, f"Recall too low: {res.recall:.1%}", {}
        if res.precision < 0.50: return False, f"Precision too low: {res.precision:.1%}", {}
        runner.suite_result.baselines["synthetic"] = {"f1": f"{res.f1:.1%}", "p": f"{res.precision:.1%}", "r": f"{res.recall:.1%}"}
        return True, f"Synthetic: F1={res.f1:.1%} P={res.precision:.1%} R={res.recall:.1%}", {}
    runner.test("Synthetic baseline", test_synthetic)
    
    if not quick:
        def test_ai4p():
            from privplay.benchmark import get_dataset, BenchmarkRunner
            from privplay.engine.classifier import ClassificationEngine
            e = ClassificationEngine()
            r = BenchmarkRunner(e, storage=None)
            try:
                d = get_dataset("ai4privacy", max_samples=50)
            except Exception as ex:
                return False, f"Load failed: {ex}", {}
            res = r.run(d, verify=False, show_progress=False)
            runner.suite_result.baselines["ai4privacy"] = {"f1": f"{res.f1:.1%}"}
            return True, f"AI4Privacy: F1={res.f1:.1%}", {}
        runner.test("AI4Privacy baseline", test_ai4p)
    
    def test_capture():
        from privplay.benchmark import get_dataset, BenchmarkRunner, capture_benchmark_errors
        from privplay.engine.classifier import ClassificationEngine
        from privplay.db import Database
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            p = f.name
        try:
            db = Database(p)
            e = ClassificationEngine()
            r = BenchmarkRunner(e, storage=None)
            d = get_dataset("synthetic_phi", max_samples=10)
            res = r.run(d, verify=False, show_progress=False)
            cap = capture_benchmark_errors(res, d, db)
            if cap.get("tps_captured", 0) == 0: return False, "No TPs captured", cap
            return True, f"Captured {cap['tps_captured']} TPs", cap
        finally:
            os.unlink(p)
    runner.test("Training capture", test_capture)


# SECTION 7: SIGNAL QUALITY
def run_section_7(runner):
    runner.section("SECTION 7: Training Signal Quality")
    
    def test_context():
        from privplay.engine.classifier import ClassificationEngine
        e = ClassificationEngine()
        text = "Patient John Smith was admitted on 01/15/2024."
        ents = e.detect(text, verify=False)
        name = next((x for x in ents if "Smith" in x.text or "John" in x.text), None)
        if not name: return False, "No name for context test", {}
        b, a = e.get_context(text, name.start, name.end)
        if len(b) < 3 or len(a) < 3: return False, f"Context too short: '{b}|{a}'", {}
        return True, f"Context OK: '{b}[{name.text}]{a}'", {}
    runner.test("Context extraction", test_context)
    
    def test_distribution():
        from privplay.benchmark import get_dataset
        d = get_dataset("synthetic_phi", max_samples=50)
        counts = defaultdict(int)
        for s in d.samples:
            for e in s.entities:
                counts[e.normalized_type] += 1
        total = sum(counts.values())
        other_pct = counts.get("OTHER", 0) / total * 100 if total else 0
        if other_pct > 20: return False, f"Too many OTHER: {other_pct:.1f}%", {}
        if len(counts) < 5: return False, f"Too few types: {len(counts)}", {}
        return True, f"{len(counts)} types, {other_pct:.1f}% OTHER", {}
    runner.test("Entity distribution", test_distribution)


def run_all_tests(verbose=False, quick=False, section=None, save_baseline=False):
    runner = TestRunner(verbose)
    print(f"\n{'='*70}\n PRIVPLAY HARDCORE TEST SUITE\n {datetime.now()}\n{'='*70}")
    
    sections = [(1,run_section_1),(2,run_section_2),(3,run_section_3),(4,run_section_4),
                (5,run_section_5),(6,lambda r:run_section_6(r,quick)),(7,run_section_7)]
    
    for num, func in sections:
        if section and num != section: continue
        try:
            func(runner)
        except Exception as e:
            print(f"\n  SECTION {num} ERROR: {e}")
        if num == 1 and runner.current_section and runner.current_section.has_critical_failure:
            print(f"\n{'='*70}\n CRITICAL FAILURE\n{'='*70}")
            break
    
    result = runner.finalize()
    
    print(f"\n{'='*70}\n SUMMARY\n{'='*70}")
    c_pass, c_fail, reset = "\033[92m", "\033[91m", "\033[0m"
    print(f"\n  Total:  {result.total_tests}")
    print(f"  Passed: {c_pass}{result.total_passed}{reset}")
    print(f"  Failed: {c_fail}{result.total_failed}{reset}")
    
    if result.total_failed: print(f"\n  {c_fail}TESTS FAILED{reset}")
    else: print(f"\n  {c_pass}ALL PASSED{reset}")
    
    if result.baselines:
        print("\n  Baselines:")
        for k, v in result.baselines.items():
            print(f"    {k}: {v}")
    
    if save_baseline:
        p = Path.home() / ".privplay" / "test_baseline.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\n  Saved: {p}")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--section", "-s", type=int)
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()
    
    result = run_all_tests(args.verbose, args.quick, args.section, args.save_baseline)
    sys.exit(0 if result.total_failed == 0 else 1)


if __name__ == "__main__":
    main()
