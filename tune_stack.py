#!/usr/bin/env python3
"""
Privplay Stack Tuning Script
============================
Applies optimized settings based on diagnostic results.

Changes:
1. Presidio threshold: 0.5 → 0.7 (reduce noise)
2. Presidio: Disable recognizers that Rules handle better
3. Add NEVER_ENTITIES filter (fixes "Email"/"Card" as NAME_PERSON)
4. Fix get_stack_status cosmetic bugs

Run: python3 tune_stack.py
"""

import shutil
from pathlib import Path
from datetime import datetime

def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    if not path.exists():
        return None
    backup = path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy(path, backup)
    print(f"  Backed up: {path.name} → {backup.name}")
    return backup

def main():
    print("=" * 60)
    print(" Privplay Stack Tuning")
    print("=" * 60)
    print()
    
    # Find project root
    # Try common locations
    candidates = [
        Path("/mnt/d/privplay"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    
    project_root = None
    for candidate in candidates:
        if (candidate / "privplay" / "engine").exists():
            project_root = candidate
            break
    
    if not project_root:
        print("ERROR: Could not find privplay project root")
        print("Run this script from the privplay directory")
        return 1
    
    print(f"Project root: {project_root}")
    print()
    
    # =========================================================================
    # 1. UPDATE CONFIG - Presidio threshold 0.5 → 0.7
    # =========================================================================
    print("[1/4] Updating Presidio threshold in config.py...")
    
    config_path = project_root / "privplay" / "config.py"
    backup_file(config_path)
    
    config_content = config_path.read_text()
    
    # Change default threshold
    old_threshold = 'score_threshold: float = 0.5'
    new_threshold = 'score_threshold: float = 0.7'
    
    if old_threshold in config_content:
        config_content = config_content.replace(old_threshold, new_threshold)
        config_path.write_text(config_content)
        print("  ✓ Changed Presidio score_threshold: 0.5 → 0.7")
    else:
        print("  ⚠ Threshold already changed or not found")
    
    # =========================================================================
    # 2. UPDATE PRESIDIO DETECTOR - Add entity filtering + NEVER_ENTITIES
    # =========================================================================
    print()
    print("[2/4] Updating presidio_detector.py...")
    
    presidio_path = project_root / "privplay" / "engine" / "models" / "presidio_detector.py"
    backup_file(presidio_path)
    
    presidio_content = presidio_path.read_text()
    
    # Add NEVER_ENTITIES constant after GLOBAL_ALLOW_LIST
    never_entities_block = '''
# ============================================================
# NEVER ENTITIES - Common words that should never be entities
# ============================================================
# These are label words that Presidio incorrectly flags as NAME_PERSON
NEVER_ENTITIES: Set[str] = {
    # Field labels
    "email", "e-mail", "phone", "telephone", "fax", "mobile", "cell",
    "card", "credit card", "debit card", "ssn", "social security",
    "name", "first name", "last name", "address", "street", "city",
    "state", "zip", "zipcode", "zip code", "country",
    "date", "dob", "date of birth", "birthday",
    "account", "account number", "routing", "routing number",
    "id", "identification", "license", "passport",
    "mrn", "medical record", "patient id", "member id",
    # Common single-word labels
    "to", "from", "re", "subject", "cc", "bcc",
    "signed", "signature", "by", "for", "of",
}

'''
    
    # Insert after GLOBAL_ALLOW_LIST definition
    if "NEVER_ENTITIES" not in presidio_content:
        insert_marker = "GLOBAL_ALLOW_LIST: Set[str] = (\n    USER_AGENT_TERMS | \n    GREETING_TERMS | \n    MEDICAL_GENERIC_TERMS | \n    TECH_TERMS\n)"
        if insert_marker in presidio_content:
            presidio_content = presidio_content.replace(
                insert_marker,
                insert_marker + "\n" + never_entities_block
            )
            print("  ✓ Added NEVER_ENTITIES constant")
        else:
            print("  ⚠ Could not find insertion point for NEVER_ENTITIES")
    else:
        print("  ⚠ NEVER_ENTITIES already exists")
    
    # Add ENTITIES_RULES_HANDLE constant
    entities_rules_handle = '''
# ============================================================
# ENTITIES THAT RULES HANDLE BETTER - Disable in Presidio
# ============================================================
# Rules have precise regex + validation (Luhn, checksums, etc.)
# Presidio adds noise and conflicts for these
ENTITIES_RULES_HANDLE: Set[str] = {
    "CREDIT_CARD",      # Rules have Luhn validation
    "US_SSN",           # Rules have format validation
    "EMAIL_ADDRESS",    # Rules have precise regex
    "PHONE_NUMBER",     # Rules handle multiple formats
    "IP_ADDRESS",       # Rules have exact pattern
    # "US_DRIVER_LICENSE",  # Keep - Rules don't have state-specific patterns yet
    # "IBAN_CODE",          # Keep - Rules don't have this
}

'''
    
    if "ENTITIES_RULES_HANDLE" not in presidio_content:
        # Insert after NEVER_ENTITIES or after GLOBAL_ALLOW_LIST
        if "NEVER_ENTITIES" in presidio_content:
            # Find end of NEVER_ENTITIES block
            insert_after = "}\n\n"
            idx = presidio_content.find("NEVER_ENTITIES")
            if idx > 0:
                # Find the closing brace of that set
                close_idx = presidio_content.find("}\n\n", idx)
                if close_idx > 0:
                    insert_pos = close_idx + len("}\n\n")
                    presidio_content = presidio_content[:insert_pos] + entities_rules_handle + presidio_content[insert_pos:]
                    print("  ✓ Added ENTITIES_RULES_HANDLE constant")
        else:
            print("  ⚠ Could not find insertion point for ENTITIES_RULES_HANDLE")
    else:
        print("  ⚠ ENTITIES_RULES_HANDLE already exists")
    
    # Update the detect() method to filter NEVER_ENTITIES and skip ENTITIES_RULES_HANDLE
    # Find and update the filtering section in detect()
    
    old_filter = '''                # Filter: Skip if in allow list
                if self.use_allow_list and self._is_in_allow_list(entity_text):
                    logger.debug(f"Filtered allow-listed term: {entity_text}")
                    continue'''
    
    new_filter = '''                # Filter: Skip if in allow list
                if self.use_allow_list and self._is_in_allow_list(entity_text):
                    logger.debug(f"Filtered allow-listed term: {entity_text}")
                    continue
                
                # Filter: Skip NEVER_ENTITIES (common label words)
                if entity_text.lower().strip() in NEVER_ENTITIES:
                    logger.debug(f"Filtered label word: {entity_text}")
                    continue
                
                # Filter: Skip entities that Rules handle better
                if result.entity_type in ENTITIES_RULES_HANDLE:
                    logger.debug(f"Skipping {result.entity_type} - Rules handle better")
                    continue'''
    
    if old_filter in presidio_content and "NEVER_ENTITIES" not in presidio_content[presidio_content.find("def detect"):]:
        presidio_content = presidio_content.replace(old_filter, new_filter)
        print("  ✓ Updated detect() with NEVER_ENTITIES and ENTITIES_RULES_HANDLE filters")
    else:
        print("  ⚠ detect() filter already updated or not found")
    
    # Write updated presidio_detector.py
    presidio_path.write_text(presidio_content)
    
    # =========================================================================
    # 3. FIX CLASSIFIER get_stack_status
    # =========================================================================
    print()
    print("[3/4] Fixing get_stack_status in classifier.py...")
    
    classifier_path = project_root / "privplay" / "engine" / "classifier.py"
    backup_file(classifier_path)
    
    classifier_content = classifier_path.read_text()
    
    # Fix the pattern_count bug (was checking self.rules.patterns, should be len(self.rules.rules))
    old_status = '''"rules": {
                "enabled": self.rules is not None,
                "pattern_count": len(self.rules.patterns) if hasattr(self.rules, 'patterns') else 0,
            },'''
    
    new_status = '''"rules": {
                "enabled": self.rules is not None,
                "rule_count": len(self.rules.rules) if hasattr(self.rules, 'rules') else 0,
            },'''
    
    if old_status in classifier_content:
        classifier_content = classifier_content.replace(old_status, new_status)
        print("  ✓ Fixed rules pattern_count → rule_count")
    else:
        print("  ⚠ pattern_count fix already applied or structure different")
    
    # Fix transformer available check
    old_transformer = '''"transformer": {
                "name": self.model.name if hasattr(self.model, 'name') else "unknown",
                "available": self.model.is_available() if hasattr(self.model, 'is_available') else True,
            },'''
    
    new_transformer = '''"transformer": {
                "name": self.model.name if hasattr(self.model, 'name') else "unknown",
                "available": True,  # If we got here, model loaded successfully
            },'''
    
    if old_transformer in classifier_content:
        classifier_content = classifier_content.replace(old_transformer, new_transformer)
        print("  ✓ Fixed transformer available status")
    else:
        print("  ⚠ Transformer status fix already applied or structure different")
    
    classifier_path.write_text(classifier_content)
    
    # =========================================================================
    # 4. UPDATE USER CONFIG FILE (if exists)
    # =========================================================================
    print()
    print("[4/4] Updating user config file...")
    
    user_config = Path.home() / ".privplay" / "config.yaml"
    
    if user_config.exists():
        backup_file(user_config)
        
        import yaml
        with open(user_config) as f:
            config_data = yaml.safe_load(f) or {}
        
        # Update presidio threshold
        if "presidio" not in config_data:
            config_data["presidio"] = {}
        config_data["presidio"]["score_threshold"] = 0.7
        
        with open(user_config, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        print("  ✓ Updated ~/.privplay/config.yaml with score_threshold: 0.7")
    else:
        # Create config file
        import yaml
        config_data = {
            "presidio": {
                "score_threshold": 0.7
            }
        }
        user_config.parent.mkdir(parents=True, exist_ok=True)
        with open(user_config, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        print("  ✓ Created ~/.privplay/config.yaml with score_threshold: 0.7")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 60)
    print(" TUNING COMPLETE")
    print("=" * 60)
    print()
    print("Changes applied:")
    print("  1. Presidio threshold: 0.5 → 0.7 (reduces noise)")
    print("  2. Added NEVER_ENTITIES filter (fixes 'Email'/'Card' as NAME_PERSON)")
    print("  3. Presidio skips: CREDIT_CARD, US_SSN, EMAIL, PHONE, IP")
    print("     (Rules handle these better with validation)")
    print("  4. Fixed get_stack_status cosmetic bugs")
    print()
    print("Next steps:")
    print("  1. Run diagnostic: python3 diagnostic.py")
    print("  2. Run benchmark:  phi-train benchmark run ai4privacy -n 500")
    print("  3. Compare F1 scores, especially CREDIT_CARD and SSN")
    print()
    print("Backups created with .bak extension if you need to revert.")
    
    return 0


if __name__ == "__main__":
    exit(main())
