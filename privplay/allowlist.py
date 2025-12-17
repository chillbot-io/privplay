"""
Allowlist for PHI/PII detection - terms that should NOT be flagged.

This is a minimal, defensible baseline. Each category is documented
with rationale. Add terms sparingly and with justification.

Usage:
    from privplay.allowlist import ALLOWLIST, is_allowed
    
    if is_allowed(entity.text):
        # Skip this entity
        pass
"""

from typing import Set

# =============================================================================
# RELATIVE DATE TERMS
# =============================================================================
# These are temporal references, not specific dates. A specific date like
# "01/15/1980" is PHI, but "today" or "yesterday" cannot identify anyone.

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
}

# =============================================================================
# BRAND NAME FALSE POSITIVES
# =============================================================================
# These trigger the "Dr. [Name]" or "Mr. [Name]" pattern but are brands/products.
# Only add well-known brands that consistently cause FPs.

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
# Common tech/programming terms that transformer models sometimes flag as
# "something" (OTHER type). These are never PHI in any context.

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
    "delete",  # HTTP methods
    "sql",
    "sdk",
    "cli",
}

# =============================================================================
# US GEOGRAPHIC REGIONS (Census Bureau + Common Directional)
# =============================================================================
# These are regional/directional terms that are NOT specific enough to be PHI.
# Per HIPAA, geographic data smaller than a state IS PHI, but broad regions
# like "Southwest" or "New England" are not identifying.
#
# Source: US Census Bureau regions/divisions + common directional terms
# https://www.census.gov/programs-surveys/popest/about/glossary/geo-terms.html

GEOGRAPHIC_REGIONS = {
    # US Census Bureau 4 Regions
    "northeast",
    "midwest",
    "south",
    "west",
    
    # US Census Bureau 9 Divisions
    "new england",
    "middle atlantic",
    "east north central",
    "west north central",
    "south atlantic",
    "east south central",
    "west south central",
    "mountain",
    "pacific",
    
    # Common directional regions (not specific locations)
    "southwest",
    "southeast",
    "northwest",
    "northeast",
    "north",
    "south",
    "east",
    "west",
    "central",
    "midwest",
    "midatlantic",
    "mid-atlantic",
    "mid atlantic",
    "midwestern",
    "southwestern",
    "southeastern",
    "northwestern",
    "northeastern",
    "northern",
    "southern",
    "eastern",
    "western",
    
    # Regional nicknames (commonly used, not specific)
    "sun belt",
    "sunbelt",
    "rust belt",
    "rustbelt",
    "bible belt",
    "corn belt",
    "cotton belt",
    "great plains",
    "great lakes",
    "gulf coast",
    "east coast",
    "west coast",
    "left coast",
    "pacific northwest",
    "four corners",
    "tristate",
    "tri-state",
    "tri state",
    "upstate",
    "downstate",
    "inland empire",
    "silicon valley",  # Commonly used generic term
    "bay area",        # Generic regional term
}

# =============================================================================
# COMBINED ALLOWLIST
# =============================================================================

ALLOWLIST: Set[str] = set()
ALLOWLIST.update(RELATIVE_DATES)
ALLOWLIST.update(BRAND_NAME_FPS)
ALLOWLIST.update(TECH_TERMS)
ALLOWLIST.update(GEOGRAPHIC_REGIONS)


def is_allowed(text: str) -> bool:
    """
    Check if text is in the allowlist.
    
    Args:
        text: Entity text to check
        
    Returns:
        True if text should NOT be flagged as PHI/PII
    """
    return text.lower().strip() in ALLOWLIST


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
    }
