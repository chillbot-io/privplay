"""
Allowlist for PHI/PII detection - terms that should NOT be flagged.

Enhanced version with context-based filtering for precision improvement.

Usage:
    from privplay.allowlist import (
        is_allowed, 
        is_date_context_excluded,
        is_phone_context_excluded,
        is_account_context_excluded,
        apply_context_filtering,
    )
"""

import re
from typing import Set, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Entity

# =============================================================================
# RELATIVE DATE TERMS
# =============================================================================

RELATIVE_DATES = {
    "today",
    "yesterday",
    "tomorrow",
    "now",
    "currently",
    "recently",
    "soon",
    "later",
    "earlier",
    "daily",
    "weekly",
    "monthly",
    "yearly",
    "annually",
}

# =============================================================================
# BRAND NAME FALSE POSITIVES
# =============================================================================

BRAND_NAME_FPS = {
    "dr. pepper",
    "dr pepper",
    "dr. scholl",
    "dr. scholl's",
    "dr scholl",
    "dr scholls",
    "mr. clean",
    "mr clean",
    "mrs. butterworth",
    "mrs butterworth",
}

# =============================================================================
# TECHNICAL TERMS
# =============================================================================

TECH_TERMS = {
    "api",
    "json",
    "xml",
    "html",
    "css",
    "http",
    "https",
    "url",
    "post",
    "get",
    "put",
    "delete",
    "sql",
    "sdk",
    "cli",
    "uuid",
    "guid",
    "hash",
    "sha256",
    "md5",
    "base64",
    "utf-8",
    "ascii",
}

# =============================================================================
# US GEOGRAPHIC REGIONS
# =============================================================================

GEOGRAPHIC_REGIONS = {
    "northeast", "midwest", "south", "west",
    "new england", "middle atlantic", "east north central",
    "west north central", "south atlantic", "east south central",
    "west south central", "mountain", "pacific",
    "southwest", "southeast", "northwest", "northeast",
    "north", "south", "east", "west", "central",
    "midatlantic", "mid-atlantic", "mid atlantic",
    "midwestern", "southwestern", "southeastern",
    "northwestern", "northeastern", "northern",
    "southern", "eastern", "western",
    "sun belt", "sunbelt", "rust belt", "rustbelt",
    "bible belt", "corn belt", "cotton belt",
    "great plains", "great lakes", "gulf coast",
    "east coast", "west coast", "left coast",
    "pacific northwest", "four corners",
    "tristate", "tri-state", "tri state",
    "upstate", "downstate", "inland empire",
    "silicon valley", "bay area",
}

# =============================================================================
# CLINICAL TERMS THAT ARE NOT NAMES
# =============================================================================

CLINICAL_NON_PHI = {
    # Status terms
    "stable", "normal", "alert", "oriented",
    "negative", "positive", "chronic", "acute",
    "mild", "moderate", "severe", "benign", "malignant",
    "primary", "secondary", "bilateral", "unilateral",
    "anterior", "posterior", "proximal", "distal",
    "lateral", "medial", "superior", "inferior",
    
    # Medical abbreviations
    "prn", "bid", "tid", "qid", "qhs", "stat", "asap",
    "npo", "dnr", "dni", "wbc", "rbc", "hgb", "hct",
    "plt", "bmp", "cbc", "lfts", "ekg", "ecg", "mri",
    "ct", "cxr", "ua", "abg",
    
    # Generic roles
    "patient", "doctor", "nurse", "physician", "surgeon",
    "resident", "attending", "intern", "fellow",
    "therapist", "technician", "pharmacist",
    
    # Unit names
    "icu", "micu", "sicu", "picu", "nicu", "ccu",
    "pacu", "or", "er", "ed",
}

# =============================================================================
# DRUG NAMES THAT LOOK LIKE PERSON NAMES
# =============================================================================

DRUG_NAME_FPS = {
    # Brand names that look like names
    "allegra", "ambien", "celebrex", "crestor", "cymbalta",
    "enbrel", "humira", "januvia", "lexapro", "lyrica",
    "plavix", "prozac", "remicade", "tamiflu", "valium",
    "xanax", "zoloft", "botox", "viagra", "cialis",
    "lipitor", "nexium", "prilosec", "zyrtec", "claritin",
    "advil", "tylenol", "motrin", "aleve", "flomax",
    
    # Generic medications
    "lisinopril", "metformin", "atorvastatin", "omeprazole",
    "metoprolol", "amlodipine", "losartan", "gabapentin",
    "sertraline", "fluoxetine", "escitalopram", "duloxetine",
    "tramadol", "hydrocodone", "oxycodone", "morphine",
    "fentanyl", "prednisone", "methylprednisolone",
    "amoxicillin", "azithromycin", "ciprofloxacin",
    "levothyroxine", "insulin", "warfarin", "heparin", "aspirin",
}

# =============================================================================
# GENERIC SYSTEM/PLACEHOLDER TERMS
# =============================================================================

SYSTEM_TERMS = {
    "null", "undefined", "none", "nil", "void", "empty",
    "default", "anonymous", "unknown", "n/a", "na",
    "tbd", "tba", "admin", "root", "user", "guest",
    "test", "demo", "example", "sample", "system",
    "service", "localhost", "placeholder",
}

# =============================================================================
# NON-PHI DATE CONTEXTS (patterns before a date that indicate it's not PHI)
# =============================================================================

NON_PHI_DATE_PATTERNS = [
    re.compile(r'published\s+(?:on\s+)?(?:in\s+)?', re.IGNORECASE),
    re.compile(r'study\s+(?:from|in|dated|conducted)', re.IGNORECASE),
    re.compile(r'protocol\s+(?:version|dated|v\d)', re.IGNORECASE),
    re.compile(r'FDA\s+approved?\s*(?:on|in)?', re.IGNORECASE),
    re.compile(r'guidelines?\s+(?:updated|from|dated)', re.IGNORECASE),
    re.compile(r'as\s+of\s+', re.IGNORECASE),
    re.compile(r'effective\s+(?:date|from)', re.IGNORECASE),
    re.compile(r'version\s+\d', re.IGNORECASE),
    re.compile(r'rev(?:ised|ision)?\s+', re.IGNORECASE),
    re.compile(r'copyright\s+', re.IGNORECASE),
    re.compile(r'©\s*', re.IGNORECASE),
    re.compile(r'last\s+(?:updated|modified|revised)', re.IGNORECASE),
    re.compile(r'released?\s+(?:on|in)?', re.IGNORECASE),
    re.compile(r'data\s+(?:from|through|between)', re.IGNORECASE),
    re.compile(r'created\s+(?:on|in)?', re.IGNORECASE),
    re.compile(r'valid\s+(?:from|until|through)', re.IGNORECASE),
    re.compile(r'expires?\s+(?:on)?', re.IGNORECASE),
]

# =============================================================================
# NON-PHI PHONE CONTEXTS
# =============================================================================

NON_PHI_PHONE_PATTERNS = [
    re.compile(r'room\s*#?\s*$', re.IGNORECASE),
    re.compile(r'ext(?:ension)?\.?\s*[:=#]?\s*$', re.IGNORECASE),
    re.compile(r'unit\s+$', re.IGNORECASE),
    re.compile(r'lab\s*(?:id|#|code)\s*[:=#]?\s*$', re.IGNORECASE),
    re.compile(r'code\s+$', re.IGNORECASE),
    re.compile(r'reference\s*(?:#|number|code)?\s*[:=#]?\s*$', re.IGNORECASE),
    re.compile(r'order\s*(?:#|number)?\s*[:=#]?\s*$', re.IGNORECASE),
    re.compile(r'case\s*(?:#|number|id)?\s*[:=#]?\s*$', re.IGNORECASE),
    re.compile(r'protocol\s+$', re.IGNORECASE),
    re.compile(r'tracking\s*(?:#|number)?\s*[:=#]?\s*$', re.IGNORECASE),
    re.compile(r'pin\s*[:=#]?\s*$', re.IGNORECASE),
    re.compile(r'version\s+$', re.IGNORECASE),
]

# =============================================================================
# NON-PHI ACCOUNT NUMBER CONTEXTS
# =============================================================================

NON_PHI_ACCOUNT_PATTERNS = [
    re.compile(r'reference\s*(?:#|number|code|id)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'order\s*(?:#|number|id)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'case\s*(?:#|number|id)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'protocol\s*(?:#|number|id)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'tracking\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'item\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'part\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'invoice\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'receipt\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'confirmation\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'icd[-\s]?\d+', re.IGNORECASE),
    re.compile(r'cpt\s+code', re.IGNORECASE),
    re.compile(r'ndc\s+', re.IGNORECASE),
    re.compile(r'lot\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'batch\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'serial\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
    re.compile(r'model\s*(?:#|number)?[:=]?\s*$', re.IGNORECASE),
]

# =============================================================================
# COMBINED ALLOWLIST
# =============================================================================

ALLOWLIST: Set[str] = set()
ALLOWLIST.update(RELATIVE_DATES)
ALLOWLIST.update(BRAND_NAME_FPS)
ALLOWLIST.update(TECH_TERMS)
ALLOWLIST.update(GEOGRAPHIC_REGIONS)
ALLOWLIST.update(CLINICAL_NON_PHI)
ALLOWLIST.update(DRUG_NAME_FPS)
ALLOWLIST.update(SYSTEM_TERMS)


def is_allowed(text: str) -> bool:
    """Check if text is in the allowlist."""
    return text.lower().strip() in ALLOWLIST


def is_date_context_excluded(entity_text: str, context_before: str) -> bool:
    """
    Check if a date detection should be excluded based on context.
    
    Args:
        entity_text: The detected date text
        context_before: Text immediately before the entity (50-100 chars)
        
    Returns:
        True if this appears to be a non-PHI date
    """
    context_lower = context_before.lower()
    
    for pattern in NON_PHI_DATE_PATTERNS:
        if pattern.search(context_lower):
            return True
    
    return False


def is_phone_context_excluded(entity_text: str, context_before: str) -> bool:
    """
    Check if a phone-like detection should be excluded based on context.
    
    Args:
        entity_text: The detected phone-like text
        context_before: Text immediately before (30-50 chars)
        
    Returns:
        True if this appears to be a room number, extension, etc.
    """
    context_lower = context_before.lower()
    
    for pattern in NON_PHI_PHONE_PATTERNS:
        if pattern.search(context_lower):
            return True
    
    return False


def is_account_context_excluded(entity_text: str, context_before: str) -> bool:
    """
    Check if an account number detection should be excluded based on context.
    
    Args:
        entity_text: The detected account-like text
        context_before: Text immediately before
        
    Returns:
        True if this appears to be a reference/order number, etc.
    """
    context_lower = context_before.lower()
    
    for pattern in NON_PHI_ACCOUNT_PATTERNS:
        if pattern.search(context_lower):
            return True
    
    return False


def apply_context_filtering(entities: List, text: str) -> List:
    """
    Apply context-based filtering to remove false positives.
    
    This is the main entry point for context-aware FP reduction.
    Call this after basic allowlist filtering.
    
    Args:
        entities: List of Entity objects
        text: Full document text
        
    Returns:
        Filtered list of entities
    """
    # Import here to avoid circular imports
    from .types import EntityType
    
    filtered = []
    
    for entity in entities:
        entity_type = entity.entity_type
        if hasattr(entity_type, 'value'):
            entity_type_str = entity_type.value
        else:
            entity_type_str = str(entity_type)
        
        # Get context before entity
        context_start = max(0, entity.start - 60)
        context_before = text[context_start:entity.start]
        
        should_exclude = False
        
        # Date context filtering
        if entity_type_str in ('DATE', 'DATE_DOB', 'DATE_ADMISSION', 'DATE_DISCHARGE'):
            if is_date_context_excluded(entity.text, context_before):
                should_exclude = True
        
        # Phone context filtering
        elif entity_type_str in ('PHONE', 'FAX'):
            if is_phone_context_excluded(entity.text, context_before):
                should_exclude = True
        
        # Account number context filtering
        elif entity_type_str in ('ACCOUNT_NUMBER', 'MRN'):
            if is_account_context_excluded(entity.text, context_before):
                should_exclude = True
        
        if not should_exclude:
            filtered.append(entity)
    
    return filtered


def get_allowlist() -> Set[str]:
    """Get a copy of the current allowlist."""
    return ALLOWLIST.copy()


def get_allowlist_stats() -> dict:
    """Get statistics about the allowlist."""
    return {
        "total": len(ALLOWLIST),
        "relative_dates": len(RELATIVE_DATES),
        "brand_fps": len(BRAND_NAME_FPS),
        "tech_terms": len(TECH_TERMS),
        "geographic_regions": len(GEOGRAPHIC_REGIONS),
        "clinical_non_phi": len(CLINICAL_NON_PHI),
        "drug_name_fps": len(DRUG_NAME_FPS),
        "system_terms": len(SYSTEM_TERMS),
    }
