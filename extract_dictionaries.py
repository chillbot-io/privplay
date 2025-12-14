#!/usr/bin/env python3
"""
Extract curated dictionaries from raw source files.

Usage:
    python extract_dictionaries.py /mnt/d/privplay/downloads /mnt/d/privplay/privplay/dictionaries/data

This will create:
    - drugs.txt (from OpenFDA NDC)
    - diagnoses.txt (from ICD-10-CM)
    - lab_tests.txt (from LOINC)
    - facilities.txt (from CMS Hospital + POS files)
"""

import json
import csv
import sys
import re
from pathlib import Path
from typing import Set


# =============================================================================
# DRUGS - OpenFDA NDC
# =============================================================================

def is_valid_drug_name(name: str) -> bool:
    """Filter out garbage drug entries."""
    if len(name) < 3 or len(name) > 100:
        return False
    
    # Must start with a letter
    if not re.match(r'^[a-zA-Z]', name):
        return False
    
    # Skip dosage patterns
    if re.search(r'\d+\s*(mg|mcg|ml|g|%|units?|iu)\b', name, re.IGNORECASE):
        return False
    
    # Skip ratio patterns (10/325)
    if re.search(r'\d+/\d+', name):
        return False
    
    # Skip percentage-only or number-only
    if re.match(r'^[\d\.\%\s]+$', name):
        return False
    
    # Skip homeopathic dilutions (30X, 100X)
    if re.search(r'\d+X\b', name, re.IGNORECASE):
        return False
    
    # Skip chemical prefixes
    if re.match(r'^\([A-Za-z0-9]+\)-', name):
        return False
    
    return True


def extract_drugs(downloads_dir: Path) -> Set[str]:
    """Extract drug names from OpenFDA NDC JSON."""
    drug_names = set()
    
    json_file = downloads_dir / "drug-ndc-0001-of-0001.json" / "drug-ndc-0001-of-0001.json"
    
    if not json_file.exists():
        print(f"  WARNING: Drug file not found: {json_file}")
        return drug_names
    
    print(f"  Reading: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for result in data.get('results', []):
        # Brand name
        brand = result.get('brand_name', '').strip()
        if brand and is_valid_drug_name(brand):
            drug_names.add(brand)
        
        # Generic name
        generic = result.get('generic_name', '')
        if generic:
            for g in generic.split(','):
                g = g.strip()
                if g and is_valid_drug_name(g):
                    drug_names.add(g)
        
        # Active ingredients
        for ing in result.get('active_ingredients', []):
            ing_name = ing.get('name', '').strip()
            if ing_name and is_valid_drug_name(ing_name):
                drug_names.add(ing_name)
    
    return drug_names


# =============================================================================
# DIAGNOSES - ICD-10-CM
# =============================================================================

def extract_diagnoses(downloads_dir: Path) -> Set[str]:
    """Extract diagnosis names from ICD-10-CM order file."""
    diagnoses = set()
    
    # Try both possible locations
    possible_paths = [
        downloads_dir / "icd10cm-Code Descriptions-2026" / "icd10cm-order-2026.txt",
        downloads_dir / "icd10orderfiles" / "icd10cm_order_2026.txt",
    ]
    
    order_file = None
    for p in possible_paths:
        if p.exists():
            order_file = p
            break
    
    if not order_file:
        print(f"  WARNING: ICD-10-CM file not found")
        return diagnoses
    
    print(f"  Reading: {order_file}")
    
    # Format: SEQNUM CODE HEADER SHORT_DESC LONG_DESC (fixed width)
    # Example: 00001 A00     0 Cholera                                                      Cholera
    
    with open(order_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip()
            if len(line) < 80:
                continue
            
            # Parse fixed-width format
            # Positions: 0-4 seq, 6-13 code, 14 header flag, 16-76 short desc, 77+ long desc
            try:
                # The long description starts around position 77
                # Short description is roughly positions 16-76
                short_desc = line[16:77].strip()
                long_desc = line[77:].strip() if len(line) > 77 else ""
                
                # Use long description if available, otherwise short
                desc = long_desc if long_desc else short_desc
                
                if desc and len(desc) >= 5:
                    diagnoses.add(desc)
            except:
                continue
    
    return diagnoses


# =============================================================================
# LAB TESTS - LOINC
# =============================================================================

def extract_lab_tests(downloads_dir: Path) -> Set[str]:
    """Extract lab test names from LOINC."""
    lab_tests = set()
    
    loinc_file = downloads_dir / "Loinc_2.81" / "LoincTable" / "Loinc.csv"
    
    if not loinc_file.exists():
        print(f"  WARNING: LOINC file not found: {loinc_file}")
        return lab_tests
    
    print(f"  Reading: {loinc_file}")
    
    with open(loinc_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # LONG_COMMON_NAME is the human-readable test name
            long_name = row.get('LONG_COMMON_NAME', '').strip()
            if long_name and len(long_name) >= 5:
                lab_tests.add(long_name)
            
            # COMPONENT is the analyte being measured
            component = row.get('COMPONENT', '').strip()
            if component and len(component) >= 3:
                lab_tests.add(component)
    
    return lab_tests


# =============================================================================
# FACILITIES - CMS Hospital + POS
# =============================================================================

def extract_facilities(downloads_dir: Path) -> Set[str]:
    """Extract facility names from CMS data."""
    facilities = set()
    
    # Hospital General Information
    hospital_file = downloads_dir / "Hospital_General_Information.csv"
    if hospital_file.exists():
        print(f"  Reading: {hospital_file}")
        with open(hospital_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('Facility Name', '').strip()
                if name and len(name) >= 3:
                    # Title case if all caps
                    if name.isupper():
                        name = name.title()
                    facilities.add(name)
    
    # Provider of Services file
    pos_file = (downloads_dir / 
                "Provider of Services File - Quality Improvement and Evaluation System" /
                "Provider of Services File - Quality Improvement and Evaluation System" /
                "2025-Q3" /
                "Hospital_and_other.DATA.Q3_2025.csv")
    
    if pos_file.exists():
        print(f"  Reading: {pos_file}")
        with open(pos_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('FAC_NAME', '').strip()
                if name and len(name) >= 3:
                    # Title case if all caps
                    if name.isupper():
                        name = name.title()
                    facilities.add(name)
    
    return facilities


# =============================================================================
# MAIN
# =============================================================================

def write_dictionary(terms: Set[str], output_path: Path, name: str, source: str):
    """Write terms to dictionary file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# {name}\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Count: {len(terms)}\n")
        f.write(f"# Generated by extract_dictionaries.py\n")
        f.write("#\n\n")
        
        for term in sorted(terms, key=str.lower):
            f.write(f"{term}\n")
    
    print(f"  Wrote {len(terms):,} terms to {output_path.name}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_dictionaries.py <downloads_dir> <output_dir>")
        print()
        print("Example:")
        print("  python extract_dictionaries.py /mnt/d/privplay/downloads /mnt/d/privplay/privplay/dictionaries/data")
        sys.exit(1)
    
    downloads_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    if not downloads_dir.exists():
        print(f"Error: Downloads directory not found: {downloads_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PRIVPLAY DICTIONARY EXTRACTION")
    print("=" * 60)
    print()
    
    # Drugs
    print("[1/4] Extracting DRUGS from OpenFDA NDC...")
    drugs = extract_drugs(downloads_dir)
    write_dictionary(drugs, output_dir / "drugs.txt", 
                     "Drug Names", "OpenFDA NDC Directory")
    print()
    
    # Diagnoses
    print("[2/4] Extracting DIAGNOSES from ICD-10-CM...")
    diagnoses = extract_diagnoses(downloads_dir)
    write_dictionary(diagnoses, output_dir / "diagnoses.txt",
                     "Diagnosis Names", "ICD-10-CM 2026")
    print()
    
    # Lab Tests
    print("[3/4] Extracting LAB TESTS from LOINC...")
    lab_tests = extract_lab_tests(downloads_dir)
    write_dictionary(lab_tests, output_dir / "lab_tests.txt",
                     "Lab Test Names", "LOINC 2.81")
    print()
    
    # Facilities
    print("[4/4] Extracting FACILITIES from CMS...")
    facilities = extract_facilities(downloads_dir)
    write_dictionary(facilities, output_dir / "facilities.txt",
                     "Healthcare Facility Names", "CMS Hospital General Information + Provider of Services")
    print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Drugs:      {len(drugs):>8,}")
    print(f"  Diagnoses:  {len(diagnoses):>8,}")
    print(f"  Lab Tests:  {len(lab_tests):>8,}")
    print(f"  Facilities: {len(facilities):>8,}")
    print(f"  {'─' * 20}")
    print(f"  Total:      {len(drugs) + len(diagnoses) + len(lab_tests) + len(facilities):>8,}")
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Review the generated .txt files")
    print("  2. Commit to your repo")
    print("  3. Update loader.py to use bundled files")


if __name__ == '__main__':
    main()
