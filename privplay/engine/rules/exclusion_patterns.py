"""Exclusion patterns for known false positives.

This module provides patterns and functions to filter out known FPs
before or after primary detection. Import into your rules engine.

Usage:
    from .exclusion_patterns import should_exclude, EXCLUSION_PATTERNS
    
    # In your detection code:
    if should_exclude(detected_text, detected_type, context):
        continue  # Skip this detection
"""

import re
from typing import Optional, Set, List, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class ExclusionRule:
    """A rule for excluding false positives."""
    name: str
    pattern: re.Pattern
    description: str
    applies_to: Optional[Set[str]] = None  # Entity types this applies to, None = all


# ============================================================
# EXCLUSION PATTERNS
# ============================================================

EXCLUSION_PATTERNS: List[ExclusionRule] = [
    # ---------------------------------------------------------
    # User-Agent / Browser Strings
    # ---------------------------------------------------------
    ExclusionRule(
        name="user_agent_mozilla",
        pattern=re.compile(r"Mozilla(?:/[\d.]+)?", re.IGNORECASE),
        description="Mozilla browser identifier",
        applies_to={"NAME_PERSON", "OTHER", "USERNAME"},
    ),
    ExclusionRule(
        name="user_agent_webkit",
        pattern=re.compile(r"(?:Apple)?WebKit(?:/[\d.]+)?", re.IGNORECASE),
        description="WebKit rendering engine",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_gecko",
        pattern=re.compile(r"Gecko(?:/[\d.]+)?", re.IGNORECASE),
        description="Gecko rendering engine",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_khtml",
        pattern=re.compile(r"KHTML(?:,?\s*like\s+Gecko)?", re.IGNORECASE),
        description="KHTML engine reference",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_chrome",
        pattern=re.compile(r"Chrome(?:/[\d.]+)?", re.IGNORECASE),
        description="Chrome browser",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_safari",
        pattern=re.compile(r"Safari(?:/[\d.]+)?", re.IGNORECASE),
        description="Safari browser",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_firefox",
        pattern=re.compile(r"Firefox(?:/[\d.]+)?", re.IGNORECASE),
        description="Firefox browser",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_edge",
        pattern=re.compile(r"Edg(?:e|A|iOS)?(?:/[\d.]+)?", re.IGNORECASE),
        description="Edge browser",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_trident",
        pattern=re.compile(r"Trident(?:/[\d.]+)?", re.IGNORECASE),
        description="Trident engine (IE)",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_presto",
        pattern=re.compile(r"Presto(?:/[\d.]+)?", re.IGNORECASE),
        description="Presto engine (Opera)",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="user_agent_platform",
        pattern=re.compile(r"\b(Windows\s*NT|Macintosh|Linux|Android|iPhone|iPad|X11)\b", re.IGNORECASE),
        description="OS platform identifiers",
        applies_to={"NAME_PERSON", "OTHER", "ADDRESS"},
    ),
    ExclusionRule(
        name="user_agent_version_string",
        pattern=re.compile(r"^\d+\.\d+(?:\.\d+)*$"),
        description="Version number strings like 5.0, 537.36",
        applies_to={"OTHER", "ACCOUNT_NUMBER", "MRN"},
    ),
    
    # ---------------------------------------------------------
    # Greetings / Salutations
    # ---------------------------------------------------------
    ExclusionRule(
        name="greeting_hello",
        pattern=re.compile(r"^(Hello|Hi|Hey|Greetings|Welcome)$", re.IGNORECASE),
        description="Common greetings",
        applies_to={"NAME_PERSON"},
    ),
    ExclusionRule(
        name="greeting_dear",
        pattern=re.compile(r"^Dear(\s+(Sir|Madam|Doctor|Dr|Mr|Mrs|Ms))?$", re.IGNORECASE),
        description="Letter salutations",
        applies_to={"NAME_PERSON"},
    ),
    ExclusionRule(
        name="greeting_closing",
        pattern=re.compile(r"^(Sincerely|Regards|Best|Cheers|Thanks|Thank\s*you)$", re.IGNORECASE),
        description="Letter closings",
        applies_to={"NAME_PERSON"},
    ),
    
    # ---------------------------------------------------------
    # Medical Generic Terms
    # ---------------------------------------------------------
    ExclusionRule(
        name="medical_unit",
        pattern=re.compile(r"^(ICU|MICU|SICU|PICU|NICU|CCU|PACU|OR|ER|ED|L&D|LDRP)$", re.IGNORECASE),
        description="Medical unit abbreviations",
        applies_to={"NAME_PERSON", "OTHER", "DEVICE_ID"},
    ),
    ExclusionRule(
        name="medical_role_generic",
        pattern=re.compile(r"^(the\s+)?(patient|doctor|nurse|physician|surgeon|resident|attending|intern|fellow|np|pa|rn|lpn|cna|md|do)s?$", re.IGNORECASE),
        description="Generic medical roles",
        applies_to={"NAME_PERSON"},
    ),
    ExclusionRule(
        name="medical_dept",
        pattern=re.compile(r"^(Radiology|Pathology|Cardiology|Neurology|Oncology|Pediatrics|Geriatrics|Psychiatry|Surgery|Medicine|Pharmacy|Laboratory|Lab)$", re.IGNORECASE),
        description="Hospital department names",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="medical_family_ref",
        pattern=re.compile(r"^(the\s+)?(mother|father|mom|dad|son|daughter|spouse|partner|husband|wife|sibling|brother|sister|parent|child|family)s?$", re.IGNORECASE),
        description="Generic family references",
        applies_to={"NAME_PERSON"},
    ),
    
    # ---------------------------------------------------------
    # Tech / Programming Terms
    # ---------------------------------------------------------
    ExclusionRule(
        name="tech_placeholder",
        pattern=re.compile(r"^(null|undefined|none|nil|void|empty|default|anonymous|unknown)$", re.IGNORECASE),
        description="Programming placeholders",
        applies_to={"NAME_PERSON", "OTHER", "USERNAME"},
    ),
    ExclusionRule(
        name="tech_user_generic",
        pattern=re.compile(r"^(admin|root|user|guest|test|demo|example|sample|system|service)$", re.IGNORECASE),
        description="Generic system users",
        applies_to={"NAME_PERSON", "USERNAME"},
    ),
    ExclusionRule(
        name="tech_protocol",
        pattern=re.compile(r"^(HTTP|HTTPS|FTP|SSH|SMTP|DNS|TCP|UDP|IP|SSL|TLS)$", re.IGNORECASE),
        description="Network protocol names",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="tech_localhost",
        pattern=re.compile(r"^localhost$", re.IGNORECASE),
        description="Localhost reference",
        applies_to={"URL", "ADDRESS", "IP_ADDRESS"},
    ),
    
    # ---------------------------------------------------------
    # Common False Positive Patterns
    # ---------------------------------------------------------
    ExclusionRule(
        name="single_letter",
        pattern=re.compile(r"^[A-Za-z]$"),
        description="Single letters",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    ExclusionRule(
        name="single_digit",
        pattern=re.compile(r"^\d$"),
        description="Single digits",
        applies_to={"AGE", "OTHER", "ACCOUNT_NUMBER"},
    ),
    ExclusionRule(
        name="common_words",
        pattern=re.compile(r"^(the|a|an|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|must|shall|can|need|dare|ought|used|to|of|in|for|on|with|at|by|from|as|into|through|during|before|after|above|below|between|under|again|further|then|once|here|there|when|where|why|how|all|each|every|both|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|just|also|now|and|but|or|if|because|until|while)$", re.IGNORECASE),
        description="Common English words",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
    
    # ---------------------------------------------------------
    # Bot / Crawler Identifiers
    # ---------------------------------------------------------
    ExclusionRule(
        name="bot_identifier",
        pattern=re.compile(r"(bot|crawler|spider|scraper|archiver|fetcher)$", re.IGNORECASE),
        description="Web bot identifiers",
        applies_to={"NAME_PERSON", "OTHER", "USERNAME"},
    ),
    ExclusionRule(
        name="bot_specific",
        pattern=re.compile(r"^(Googlebot|Bingbot|Slurp|DuckDuckBot|Baiduspider|YandexBot|Sogou|Exabot|facebot|ia_archiver)$", re.IGNORECASE),
        description="Specific search engine bots",
        applies_to={"NAME_PERSON", "OTHER"},
    ),
]


# Pre-compiled set of exact matches for fast lookup
EXACT_EXCLUSIONS: Set[str] = {
    # User-Agent components
    "mozilla", "webkit", "applewebkit", "gecko", "khtml", "chrome", "safari",
    "firefox", "edge", "trident", "presto", "opera", "brave", "chromium",
    "windows", "macintosh", "linux", "android", "iphone", "ipad", "x11",
    # Greetings
    "hello", "hi", "hey", "dear", "greetings", "welcome", "thanks", "regards",
    "sincerely", "best", "cheers",
    # Medical
    "icu", "micu", "sicu", "picu", "nicu", "ccu", "er", "or", "ed", "pacu",
    "patient", "doctor", "nurse", "physician", "attending", "resident",
    # Tech
    "null", "undefined", "none", "admin", "root", "user", "test", "localhost",
    "http", "https", "ftp", "ssh", "tcp", "udp", "ip",
}


def should_exclude(
    text: str,
    entity_type: str,
    context: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if detected entity should be excluded as a known false positive.
    
    Args:
        text: The detected entity text
        entity_type: The detected entity type (e.g., "NAME_PERSON")
        context: Optional surrounding context
        
    Returns:
        Tuple of (should_exclude: bool, reason: str or None)
    """
    text_lower = text.lower().strip()
    entity_type_upper = entity_type.upper()
    
    # Fast path: exact match in exclusion set
    if text_lower in EXACT_EXCLUSIONS:
        return True, f"Exact match in exclusion list: {text_lower}"
    
    # Check each exclusion pattern
    for rule in EXCLUSION_PATTERNS:
        # Skip if rule doesn't apply to this entity type
        if rule.applies_to and entity_type_upper not in rule.applies_to:
            continue
            
        # Check pattern match
        if rule.pattern.match(text):
            return True, f"Matched exclusion rule: {rule.name}"
    
    # Context-based exclusions
    if context:
        context_lower = context.lower()
        
        # User-Agent string detection (full context check)
        if "mozilla/" in context_lower or "user-agent" in context_lower:
            # If we're inside a User-Agent string, exclude common terms
            ua_terms = {"compatible", "msie", "rv:", "wow64", "win64", "x64", "arm"}
            if text_lower in ua_terms:
                return True, "Inside User-Agent string"
    
    return False, None


def filter_entities(entities: list, context: str) -> list:
    """
    Filter a list of entities, removing known false positives.
    
    Args:
        entities: List of Entity objects (must have .text and .entity_type attributes)
        context: The full text context
        
    Returns:
        Filtered list of entities
    """
    filtered = []
    for entity in entities:
        entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        exclude, reason = should_exclude(entity.text, entity_type, context)
        if not exclude:
            filtered.append(entity)
    return filtered


# ============================================================
# UTILITY: Get statistics on exclusion patterns
# ============================================================

def get_exclusion_stats() -> dict:
    """Get statistics about exclusion patterns."""
    return {
        "total_patterns": len(EXCLUSION_PATTERNS),
        "exact_exclusions": len(EXACT_EXCLUSIONS),
        "by_category": {
            "user_agent": len([r for r in EXCLUSION_PATTERNS if "user_agent" in r.name]),
            "greeting": len([r for r in EXCLUSION_PATTERNS if "greeting" in r.name]),
            "medical": len([r for r in EXCLUSION_PATTERNS if "medical" in r.name]),
            "tech": len([r for r in EXCLUSION_PATTERNS if "tech" in r.name]),
            "common": len([r for r in EXCLUSION_PATTERNS if "common" in r.name or "single" in r.name]),
            "bot": len([r for r in EXCLUSION_PATTERNS if "bot" in r.name]),
        }
    }
