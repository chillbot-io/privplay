"""Dictionary loader and Presidio integration with strict validation."""

import logging
import re
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# Bundled data location (relative to this module)
BUNDLED_DATA_DIR = Path(__file__).parent / "data"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum term length by dictionary type
# Shorter terms have higher FP risk
MIN_TERM_LENGTH = {
    "payers": 4,      # Insurance names are typically longer
    "drugs": 4,       # Short drug names like "Advil" (5) are fine, but avoid "ACE" etc.
    "diagnoses": 4,   # Medical conditions
    "lab_tests": 4,   # Lab test names
    "facilities": 5,  # Facility names should be substantial
}

DEFAULT_MIN_LENGTH = 4

# Maximum terms to register with Presidio (performance limit)
MAX_PRESIDIO_TERMS = 10000

# Terms that should NEVER match regardless of dictionary
# These are common words that appear in some medical dictionaries
GLOBAL_BLOCKLIST = {
    # Common English words that might appear in drug/diagnosis lists
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "is", "it", "be", "as", "was", "were", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "who", "what", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now", "here", "there", "then", "once",
    
    # Common clinical terms that aren't PHI
    "patient", "doctor", "nurse", "hospital", "clinic", "medical", "health",
    "treatment", "diagnosis", "test", "result", "normal", "abnormal",
    "positive", "negative", "left", "right", "upper", "lower",
    "pain", "acute", "chronic", "mild", "moderate", "severe",
    "daily", "weekly", "monthly", "oral", "topical",
    
    # Numbers and measurements (sometimes in drug names)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "mg", "ml", "mcg", "kg", "lb", "oz", "cc",
    
    # Generic terms
    "type", "form", "group", "class", "category", "level", "grade", "stage",
    "unit", "dose", "tablet", "capsule", "solution", "injection",
}

# Patterns that indicate a term is likely garbage (not a real entity name)
GARBAGE_PATTERNS = [
    r'^\d+$',                    # Pure numbers
    r'^\d+\.\d+$',               # Decimal numbers
    r'^\d+%$',                   # Percentages
    r'^\d+mg$',                  # Dosages
    r'^\d+ml$',                  # Volumes
    r'^[A-Z]-\d+$',              # Codes like "A-10"
    r'^\([A-Z]\)-',              # Stereochemistry prefixes like "(S)-"
    r'^[\W_]+$',                 # Only punctuation/symbols
    r'^\s*$',                    # Empty/whitespace
]

GARBAGE_REGEX = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]


def is_valid_term(term: str, min_length: int = DEFAULT_MIN_LENGTH) -> Tuple[bool, str]:
    """
    Validate a dictionary term.
    
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Strip whitespace
    term = term.strip()
    
    # Check empty
    if not term:
        return False, "empty"
    
    # Check minimum length
    if len(term) < min_length:
        return False, "too_short"
    
    # Check global blocklist
    if term.lower() in GLOBAL_BLOCKLIST:
        return False, "blocklisted"
    
    # Check garbage patterns
    for pattern in GARBAGE_REGEX:
        if pattern.match(term):
            return False, "garbage_pattern"
    
    # Must start with a letter (filters out numeric codes)
    if not term[0].isalpha():
        return False, "starts_with_non_letter"
    
    return True, ""


def load_dictionary(path: Path, min_length: int = DEFAULT_MIN_LENGTH) -> Set[str]:
    """
    Load a dictionary file into a set with strict validation.
    
    Args:
        path: Path to dictionary file (one term per line)
        min_length: Minimum term length to include
        
    Returns:
        Set of validated terms (lowercase, stripped)
    """
    terms = set()
    rejected = {"too_short": 0, "blocklisted": 0, "garbage_pattern": 0, "starts_with_non_letter": 0, "empty": 0}
    
    if not path.exists():
        logger.warning(f"Dictionary not found: {path}")
        return terms
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Validate term
                is_valid, reason = is_valid_term(line, min_length)
                
                if is_valid:
                    terms.add(line.lower())
                else:
                    rejected[reason] = rejected.get(reason, 0) + 1
        
        # Log statistics
        total_rejected = sum(rejected.values())
        if total_rejected > 0:
            logger.info(
                f"Loaded {len(terms)} terms from {path.name} "
                f"(rejected {total_rejected}: {rejected})"
            )
        else:
            logger.info(f"Loaded {len(terms)} terms from {path.name}")
        
        return terms
        
    except Exception as e:
        logger.error(f"Failed to load dictionary {path}: {e}")
        return terms


def get_bundled_path(name: str) -> Path:
    """Get path to a bundled dictionary."""
    return BUNDLED_DATA_DIR / f"{name}.txt"


# =============================================================================
# DICTIONARY LOADERS - All bundled, no downloads required
# =============================================================================

def load_payers() -> Set[str]:
    """Load health insurance payer names."""
    return load_dictionary(
        get_bundled_path("payers"),
        min_length=MIN_TERM_LENGTH.get("payers", DEFAULT_MIN_LENGTH)
    )


def load_drugs() -> Set[str]:
    """Load drug names (brand, generic, ingredients)."""
    return load_dictionary(
        get_bundled_path("drugs"),
        min_length=MIN_TERM_LENGTH.get("drugs", DEFAULT_MIN_LENGTH)
    )


def load_diagnoses() -> Set[str]:
    """Load diagnosis/condition names from ICD-10-CM."""
    return load_dictionary(
        get_bundled_path("diagnoses"),
        min_length=MIN_TERM_LENGTH.get("diagnoses", DEFAULT_MIN_LENGTH)
    )


def load_lab_tests() -> Set[str]:
    """Load laboratory test names from LOINC."""
    return load_dictionary(
        get_bundled_path("lab_tests"),
        min_length=MIN_TERM_LENGTH.get("lab_tests", DEFAULT_MIN_LENGTH)
    )


def load_facilities() -> Set[str]:
    """Load healthcare facility names (hospitals, clinics, etc.)."""
    return load_dictionary(
        get_bundled_path("facilities"),
        min_length=MIN_TERM_LENGTH.get("facilities", DEFAULT_MIN_LENGTH)
    )


# Legacy alias for backwards compatibility
def load_hospitals() -> Set[str]:
    """Load healthcare facility names. Alias for load_facilities()."""
    return load_facilities()


def load_all_dictionaries() -> Dict[str, Set[str]]:
    """
    Load all available dictionaries.
    
    Returns:
        Dict mapping entity type to set of terms
    """
    return {
        "HEALTH_PLAN": load_payers(),
        "DRUG": load_drugs(),
        "DIAGNOSIS": load_diagnoses(),
        "LAB_TEST": load_lab_tests(),
        "FACILITY": load_facilities(),
    }


def register_with_presidio(analyzer):
    """
    Register dictionary-based recognizers with Presidio analyzer.
    
    Args:
        analyzer: Presidio AnalyzerEngine instance
    """
    try:
        from presidio_analyzer import PatternRecognizer
    except ImportError:
        logger.warning("Presidio not installed, skipping dictionary registration")
        return
    
    dictionaries = load_all_dictionaries()
    
    for entity_type, terms in dictionaries.items():
        if not terms:
            logger.debug(f"No terms for {entity_type}, skipping")
            continue
        
        original_count = len(terms)
        
        # Limit dictionary size for performance
        if len(terms) > MAX_PRESIDIO_TERMS:
            logger.warning(
                f"{entity_type} has {len(terms)} terms, limiting to {MAX_PRESIDIO_TERMS} "
                f"(prioritizing longer, more specific terms)"
            )
            # Prioritize longer terms (more specific, fewer FPs)
            sorted_terms = sorted(terms, key=lambda t: (-len(t), t))
            terms = set(sorted_terms[:MAX_PRESIDIO_TERMS])
        
        # Create deny_list recognizer
        recognizer = PatternRecognizer(
            supported_entity=entity_type,
            deny_list=list(terms),
            name=f"{entity_type.lower()}_dictionary",
        )
        
        try:
            analyzer.registry.add_recognizer(recognizer)
            logger.info(
                f"Registered {entity_type} recognizer "
                f"({len(terms)}/{original_count} terms)"
            )
        except Exception as e:
            logger.error(f"Failed to register {entity_type}: {e}")


class DictionaryDetector:
    """
    Standalone dictionary-based detector.
    
    Use this if Presidio is not available or for direct dictionary lookups.
    """
    
    def __init__(self):
        self.dictionaries: Dict[str, Set[str]] = {}
        self._loaded = False
    
    def _load(self):
        """Lazy load dictionaries."""
        if self._loaded:
            return
        
        self.dictionaries = load_all_dictionaries()
        self._loaded = True
    
    def detect(self, text: str) -> List[dict]:
        """
        Detect dictionary terms in text.
        
        Returns list of matches with entity_type, text, start, end.
        """
        self._load()
        
        # Handle empty/whitespace input
        if not text or not text.strip():
            return []
        
        text_lower = text.lower()
        matches = []
        
        for entity_type, terms in self.dictionaries.items():
            for term in terms:
                # Skip short terms (already filtered at load, but double-check)
                if len(term) < 4:
                    continue
                
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(term, start)
                    if pos == -1:
                        break
                    
                    # Check word boundaries
                    before_ok = pos == 0 or not text_lower[pos-1].isalnum()
                    after_pos = pos + len(term)
                    after_ok = after_pos >= len(text) or not text_lower[after_pos].isalnum()
                    
                    if before_ok and after_ok:
                        matches.append({
                            "entity_type": entity_type,
                            "text": text[pos:pos + len(term)],
                            "start": pos,
                            "end": pos + len(term),
                            "confidence": 0.85,
                        })
                    
                    start = pos + 1
        
        return matches
    
    def contains(self, text: str, entity_type: str) -> bool:
        """Check if text contains any term of the given entity type."""
        self._load()
        
        terms = self.dictionaries.get(entity_type, set())
        
        if not text or not text.strip():
            return False
        
        text_lower = text.lower()
        
        for term in terms:
            if len(term) >= 4 and term in text_lower:
                # Verify word boundary
                pos = text_lower.find(term)
                if pos != -1:
                    before_ok = pos == 0 or not text_lower[pos-1].isalnum()
                    after_pos = pos + len(term)
                    after_ok = after_pos >= len(text_lower) or not text_lower[after_pos].isalnum()
                    if before_ok and after_ok:
                        return True
        
        return False
    
    def lookup(self, term: str) -> Optional[str]:
        """
        Look up a term and return its entity type if found.
        
        Args:
            term: Term to look up
            
        Returns:
            Entity type string if found, None otherwise
        """
        self._load()
        
        term_lower = term.lower().strip()
        
        for entity_type, terms in self.dictionaries.items():
            if term_lower in terms:
                return entity_type
        
        return None


def get_dictionary_status() -> dict:
    """Get detailed status of all bundled dictionaries."""
    status = {}
    counts = {}
    issues = []
    
    for name in ["payers", "drugs", "diagnoses", "lab_tests", "facilities"]:
        path = get_bundled_path(name)
        exists = path.exists()
        status[name] = exists
        
        if exists:
            # Load with validation to get accurate count
            min_len = MIN_TERM_LENGTH.get(name, DEFAULT_MIN_LENGTH)
            terms = load_dictionary(path, min_len)
            counts[name] = len(terms)
            
            # Check for potential issues
            if len(terms) == 0:
                issues.append(f"{name}: empty after validation")
            elif len(terms) < 10:
                issues.append(f"{name}: suspiciously few terms ({len(terms)})")
        else:
            counts[name] = 0
            issues.append(f"{name}: file not found")
    
    return {
        "available": status,
        "counts": counts,
        "all_available": all(status.values()),
        "total_terms": sum(counts.values()),
        "issues": issues if issues else None,
    }


def validate_dictionaries() -> Dict[str, Any]:
    """
    Perform comprehensive validation of all dictionaries.
    
    Returns detailed report of any issues found.
    """
    report = {
        "valid": True,
        "dictionaries": {},
        "issues": [],
    }
    
    for name in ["payers", "drugs", "diagnoses", "lab_tests", "facilities"]:
        path = get_bundled_path(name)
        dict_report = {
            "path": str(path),
            "exists": path.exists(),
            "total_lines": 0,
            "valid_terms": 0,
            "rejected": {},
            "sample_terms": [],
            "sample_rejected": [],
        }
        
        if not path.exists():
            report["valid"] = False
            report["issues"].append(f"{name}: file not found at {path}")
            report["dictionaries"][name] = dict_report
            continue
        
        min_len = MIN_TERM_LENGTH.get(name, DEFAULT_MIN_LENGTH)
        valid_terms = []
        rejected_samples = []
        rejected_counts = {}
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                dict_report["total_lines"] += 1
                
                is_valid, reason = is_valid_term(line, min_len)
                if is_valid:
                    valid_terms.append(line)
                else:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                    if len(rejected_samples) < 5:
                        rejected_samples.append(f"{line!r} ({reason})")
        
        dict_report["valid_terms"] = len(valid_terms)
        dict_report["rejected"] = rejected_counts
        dict_report["sample_terms"] = valid_terms[:5]
        dict_report["sample_rejected"] = rejected_samples
        
        # Check for issues
        if len(valid_terms) == 0:
            report["valid"] = False
            report["issues"].append(f"{name}: no valid terms after filtering")
        
        rejection_rate = sum(rejected_counts.values()) / max(dict_report["total_lines"], 1)
        if rejection_rate > 0.5:
            report["issues"].append(
                f"{name}: high rejection rate ({rejection_rate:.1%})"
            )
        
        report["dictionaries"][name] = dict_report
    
    return report
