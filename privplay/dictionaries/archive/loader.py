"""Dictionary loader and Presidio integration."""

import logging
from pathlib import Path
from typing import List, Set, Optional

logger = logging.getLogger(__name__)

# Bundled data location (relative to this module)
BUNDLED_DATA_DIR = Path(__file__).parent / "data"


def load_dictionary(path: Path) -> Set[str]:
    """
    Load a dictionary file into a set.
    
    Args:
        path: Path to dictionary file (one term per line)
        
    Returns:
        Set of terms (lowercase, stripped)
    """
    terms = set()
    
    if not path.exists():
        logger.warning(f"Dictionary not found: {path}")
        return terms
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                terms.add(line.lower())
        
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
    return load_dictionary(get_bundled_path("payers"))


def load_drugs() -> Set[str]:
    """Load drug names (brand, generic, ingredients)."""
    return load_dictionary(get_bundled_path("drugs"))


def load_diagnoses() -> Set[str]:
    """Load diagnosis/condition names from ICD-10-CM."""
    return load_dictionary(get_bundled_path("diagnoses"))


def load_lab_tests() -> Set[str]:
    """Load laboratory test names from LOINC."""
    return load_dictionary(get_bundled_path("lab_tests"))


def load_facilities() -> Set[str]:
    """Load healthcare facility names (hospitals, clinics, etc.)."""
    return load_dictionary(get_bundled_path("facilities"))


# Legacy alias for backwards compatibility
def load_hospitals() -> Set[str]:
    """Load healthcare facility names. Alias for load_facilities()."""
    return load_facilities()


def load_all_dictionaries() -> dict:
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
        
        # Create deny_list recognizer
        recognizer = PatternRecognizer(
            supported_entity=entity_type,
            deny_list=list(terms),
            name=f"{entity_type.lower()}_dictionary",
        )
        
        try:
            analyzer.registry.add_recognizer(recognizer)
            logger.info(f"Registered {entity_type} recognizer ({len(terms)} terms)")
        except Exception as e:
            logger.error(f"Failed to register {entity_type}: {e}")


class DictionaryDetector:
    """
    Standalone dictionary-based detector.
    
    Use this if Presidio is not available or for direct dictionary lookups.
    """
    
    def __init__(self):
        self.dictionaries = {}
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
        
        text_lower = text.lower()
        matches = []
        
        for entity_type, terms in self.dictionaries.items():
            for term in terms:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(term, start)
                    if pos == -1:
                        break
                    
                    # Check word boundaries (simple check)
                    before_ok = pos == 0 or not text_lower[pos-1].isalnum()
                    after_ok = pos + len(term) >= len(text) or not text_lower[pos + len(term)].isalnum()
                    
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
        text_lower = text.lower()
        
        for term in terms:
            if term in text_lower:
                return True
        
        return False


def get_dictionary_status() -> dict:
    """Get status of all bundled dictionaries."""
    status = {
        "payers": get_bundled_path("payers").exists(),
        "drugs": get_bundled_path("drugs").exists(),
        "diagnoses": get_bundled_path("diagnoses").exists(),
        "lab_tests": get_bundled_path("lab_tests").exists(),
        "facilities": get_bundled_path("facilities").exists(),
    }
    
    # Get term counts for loaded dictionaries
    counts = {}
    for name, exists in status.items():
        if exists:
            terms = load_dictionary(get_bundled_path(name))
            counts[name] = len(terms)
        else:
            counts[name] = 0
    
    return {
        "available": status,
        "counts": counts,
        "all_available": all(status.values()),
        "total_terms": sum(counts.values()),
    }
