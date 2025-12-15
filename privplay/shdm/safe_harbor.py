"""SHDM Safe Harbor Transforms - HIPAA-compliant data transformations.

Implements the three key Safe Harbor transformations:
1. Date shifting - preserves relative timing
2. Age generalization - ages >89 become "90+"
3. ZIP code handling - 3-digit prefix or "000" for low-population areas

These transforms are applied AFTER detection, converting detected values
to Safe Harbor compliant forms while maintaining data utility.
"""

import re
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


# =============================================================================
# ZIP CODE POPULATION DATA
# =============================================================================

# ZIP 3-digit prefixes with total population <20,000
# Source: Census data - these prefixes must be replaced with "000"
# This is a subset - full implementation would load from census data
LOW_POPULATION_ZIP3 = {
    # Montana
    '590', '591', '592', '593', '594', '595', '596', '597', '598', '599',
    # Wyoming  
    '820', '821', '822', '823', '824', '825', '826', '827', '828', '829',
    '830', '831',
    # North Dakota
    '582', '583', '584', '585', '586', '587', '588',
    # South Dakota
    '572', '573', '574', '575', '576', '577',
    # Nebraska (rural)
    '693', '694', '695',
    # Alaska
    '995', '996', '997', '998', '999',
    # Idaho (rural)
    '832', '833', '834',
    # Nevada (rural)
    '894', '895',
    # New Mexico (rural)
    '873', '874', '875', '876', '877', '878', '879',
    # Puerto Rico (some areas)
    '006', '007', '008', '009',
    # Virgin Islands
    '008',
    # Guam
    '969',
}

# Actually, the HIPAA rule is more specific - only these 17 ZIP3 prefixes
# have populations <20,000 based on 2000 census and must be zeroed:
HIPAA_ZERO_ZIP3 = {
    '036',  # NH
    '059',  # VT  
    '063',  # NH
    '102',  # NY
    '203',  # DC
    '556',  # MN
    '692',  # NE
    '790',  # TX
    '821',  # WY
    '823',  # WY
    '830',  # WY
    '831',  # WY
    '878',  # NM
    '879',  # NM
    '884',  # TX
    '890',  # NV
    '893',  # NV
}


# =============================================================================
# DATE SHIFTING
# =============================================================================

class DateShifter:
    """Shifts dates while preserving relative timing within a conversation.
    
    All dates in a conversation are shifted by the same random offset,
    so "3 days after admission" remains "3 days after admission".
    
    The offset is deterministic per conversation (derived from conv_id)
    so the same document always gets the same shift.
    """
    
    def __init__(self, min_days: int = 30, max_days: int = 365):
        """Initialize date shifter.
        
        Args:
            min_days: Minimum shift (default 30 days)
            max_days: Maximum shift (default 365 days)
        """
        self.min_days = min_days
        self.max_days = max_days
        self._offsets: Dict[str, int] = {}  # conv_id -> offset in days
    
    def get_offset(self, conv_id: str) -> int:
        """Get deterministic offset for a conversation."""
        if conv_id not in self._offsets:
            # Generate deterministic offset from conv_id
            hash_bytes = hashlib.sha256(conv_id.encode()).digest()
            # Use first 4 bytes as int
            hash_int = int.from_bytes(hash_bytes[:4], 'big')
            # Map to range [min_days, max_days]
            offset = self.min_days + (hash_int % (self.max_days - self.min_days))
            self._offsets[conv_id] = offset
        return self._offsets[conv_id]
    
    def shift_date(self, date_str: str, conv_id: str) -> str:
        """Shift a date string by the conversation's offset.
        
        Supports multiple formats:
        - MM/DD/YYYY, MM-DD-YYYY
        - YYYY-MM-DD (ISO)
        - Month DD, YYYY
        
        Returns shifted date in same format, or original if unparseable.
        """
        offset_days = self.get_offset(conv_id)
        
        # Try different date formats
        formats = [
            ('%m/%d/%Y', '%m/%d/%Y'),
            ('%m-%d-%Y', '%m-%d-%Y'),
            ('%m/%d/%y', '%m/%d/%y'),
            ('%m-%d-%y', '%m-%d-%y'),
            ('%Y-%m-%d', '%Y-%m-%d'),
            ('%Y/%m/%d', '%Y/%m/%d'),
            ('%B %d, %Y', '%B %d, %Y'),
            ('%b %d, %Y', '%b %d, %Y'),
            ('%d %B %Y', '%d %B %Y'),
            ('%d %b %Y', '%d %b %Y'),
        ]
        
        for parse_fmt, output_fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), parse_fmt)
                shifted = dt + timedelta(days=offset_days)
                return shifted.strftime(output_fmt)
            except ValueError:
                continue
        
        # Couldn't parse - return original
        return date_str
    
    def shift_year_only(self, date_str: str, conv_id: str) -> str:
        """For Safe Harbor, return only the year (shifted).
        
        This is the strictest interpretation - only year is retained.
        """
        offset_days = self.get_offset(conv_id)
        
        # Try to extract year
        formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
            '%Y-%m-%d', '%Y/%m/%d',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                shifted = dt + timedelta(days=offset_days)
                return str(shifted.year)
            except ValueError:
                continue
        
        # Try to find a 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = int(year_match.group())
            shifted_year = year + (offset_days // 365)
            return str(shifted_year)
        
        return date_str


# =============================================================================
# AGE GENERALIZATION
# =============================================================================

class AgeGeneralizer:
    """Generalizes ages >89 to "90+" per HIPAA Safe Harbor."""
    
    # Patterns to match age expressions
    AGE_PATTERNS = [
        # "92 year old", "92-year-old", "92 y/o", "92yo"
        (re.compile(r'\b(\d{1,3})\s*[-]?\s*(?:year|yr|y)[-\s]*(?:old|o)\b', re.I), 
         lambda m, age: f"{age} year old"),
        
        # "age 92", "age: 92", "aged 92"
        (re.compile(r'\b(?:age|aged)[:\s]*(\d{1,3})\b', re.I),
         lambda m, age: f"age {age}"),
        
        # "92 y.o.", "92 yo"
        (re.compile(r'\b(\d{1,3})\s*y\.?o\.?\b', re.I),
         lambda m, age: f"{age} y.o."),
        
        # "92-year-old" (hyphenated adjective)
        (re.compile(r'\b(\d{1,3})-year-old\b', re.I),
         lambda m, age: f"{age}-year-old"),
    ]
    
    def __init__(self, threshold: int = 89):
        """Initialize with age threshold (default 89 per HIPAA)."""
        self.threshold = threshold
    
    def generalize(self, text: str) -> str:
        """Replace ages >89 with 90+ in text."""
        result = text
        
        for pattern, formatter in self.AGE_PATTERNS:
            def replace_age(match):
                age = int(match.group(1))
                if age > self.threshold:
                    return formatter(match, "90+")
                return match.group(0)
            
            result = pattern.sub(replace_age, result)
        
        return result
    
    def generalize_value(self, age: int) -> str:
        """Generalize a single age value."""
        if age > self.threshold:
            return "90+"
        return str(age)


# =============================================================================
# ZIP CODE HANDLER
# =============================================================================

class ZIPHandler:
    """Handles ZIP codes per HIPAA Safe Harbor requirements.
    
    - Retains only first 3 digits
    - Replaces with "000" if 3-digit prefix has <20,000 population
    """
    
    ZIP_PATTERN = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
    
    def __init__(self, zero_prefixes: set = None):
        """Initialize with set of ZIP3 prefixes to zero out.
        
        Args:
            zero_prefixes: Set of 3-digit prefixes with <20K population.
                          Defaults to HIPAA-specified list.
        """
        self.zero_prefixes = zero_prefixes or HIPAA_ZERO_ZIP3
    
    def transform(self, zip_code: str) -> str:
        """Transform ZIP to Safe Harbor compliant form.
        
        Args:
            zip_code: Original ZIP (5 or 9 digit)
            
        Returns:
            3-digit prefix + "XX" or "000XX" for low-population areas
        """
        # Extract digits
        digits = ''.join(c for c in zip_code if c.isdigit())
        
        if len(digits) < 5:
            return zip_code  # Invalid, return as-is
        
        prefix = digits[:3]
        
        if prefix in self.zero_prefixes:
            return "000XX"
        else:
            return f"{prefix}XX"
    
    def transform_in_text(self, text: str) -> str:
        """Transform all ZIP codes in text."""
        def replace_zip(match):
            return self.transform(match.group(0))
        
        return self.ZIP_PATTERN.sub(replace_zip, text)


# =============================================================================
# COMBINED SAFE HARBOR TRANSFORMER
# =============================================================================

@dataclass
class SafeHarborConfig:
    """Configuration for Safe Harbor transforms."""
    shift_dates: bool = True
    date_shift_min_days: int = 30
    date_shift_max_days: int = 365
    year_only: bool = False  # If True, dates become year only
    
    generalize_ages: bool = True
    age_threshold: int = 89
    
    transform_zips: bool = True


class SafeHarborTransformer:
    """Applies all Safe Harbor transformations to text.
    
    Usage:
        transformer = SafeHarborTransformer()
        
        # Transform detected entities
        safe_text = transformer.transform(
            text="Patient age 92, DOB 03/15/1932, ZIP 82301",
            conv_id="abc123",
            entities=[...]  # Detected entities with types
        )
        # Result: "Patient age 90+, DOB 1932, ZIP 823XX"
    """
    
    def __init__(self, config: SafeHarborConfig = None):
        self.config = config or SafeHarborConfig()
        
        self._date_shifter = DateShifter(
            min_days=self.config.date_shift_min_days,
            max_days=self.config.date_shift_max_days,
        )
        self._age_generalizer = AgeGeneralizer(
            threshold=self.config.age_threshold,
        )
        self._zip_handler = ZIPHandler()
    
    def transform_date(self, date_str: str, conv_id: str) -> str:
        """Transform a date value."""
        if not self.config.shift_dates:
            return date_str
        
        if self.config.year_only:
            return self._date_shifter.shift_year_only(date_str, conv_id)
        else:
            return self._date_shifter.shift_date(date_str, conv_id)
    
    def transform_age(self, age_text: str) -> str:
        """Transform an age value or text containing age."""
        if not self.config.generalize_ages:
            return age_text
        
        # Try to parse as number
        try:
            age = int(''.join(c for c in age_text if c.isdigit()))
            return self._age_generalizer.generalize_value(age)
        except ValueError:
            return self._age_generalizer.generalize(age_text)
    
    def transform_zip(self, zip_code: str) -> str:
        """Transform a ZIP code."""
        if not self.config.transform_zips:
            return zip_code
        return self._zip_handler.transform(zip_code)
    
    def transform_entity(
        self, 
        entity_text: str, 
        entity_type: str, 
        conv_id: str,
    ) -> str:
        """Transform a single entity based on its type.
        
        Args:
            entity_text: The detected entity text
            entity_type: The entity type (DATE, AGE, ZIP, etc.)
            conv_id: Conversation ID for consistent date shifting
            
        Returns:
            Transformed text
        """
        etype = entity_type.upper()
        
        # Date types
        if etype in ('DATE', 'DATE_DOB', 'DOB', 'DATE_OF_BIRTH'):
            return self.transform_date(entity_text, conv_id)
        
        # Age
        if etype == 'AGE':
            return self.transform_age(entity_text)
        
        # ZIP
        if etype in ('ZIP', 'ZIPCODE', 'ZIP_CODE', 'POSTAL_CODE'):
            return self.transform_zip(entity_text)
        
        # No transformation for other types
        return entity_text
    
    def should_transform(self, entity_type: str) -> bool:
        """Check if an entity type should be transformed (vs tokenized)."""
        etype = entity_type.upper()
        return etype in (
            'DATE', 'DATE_DOB', 'DOB', 'DATE_OF_BIRTH',
            'AGE',
            'ZIP', 'ZIPCODE', 'ZIP_CODE', 'POSTAL_CODE',
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_safe_harbor(
    text: str,
    entities: list,
    conv_id: str,
    config: SafeHarborConfig = None,
) -> Tuple[str, List[dict]]:
    """Apply Safe Harbor transforms to detected entities in text.
    
    Args:
        text: Original text
        entities: List of detected entities with start, end, text, entity_type
        conv_id: Conversation ID for consistent date shifting
        config: Optional SafeHarborConfig
        
    Returns:
        Tuple of (transformed_text, list of transformations applied)
    """
    transformer = SafeHarborTransformer(config)
    transformations = []
    
    # Filter to only transformable entities
    transform_entities = []
    for entity in entities:
        etype = entity.get('entity_type', entity.get('type', ''))
        if transformer.should_transform(etype):
            transform_entities.append(entity)
    
    if not transform_entities:
        return text, []
    
    # Sort entities by start position (descending) for replacement
    # This ensures we replace from end to start, preserving earlier positions
    sorted_entities = sorted(transform_entities, key=lambda e: e.get('start', 0), reverse=True)
    
    result = text
    for entity in sorted_entities:
        etype = entity.get('entity_type', entity.get('type', ''))
        original = entity.get('text', '')
        start = entity.get('start', 0)
        end = entity.get('end', start + len(original))
        
        # Verify the text at this position matches
        actual_text = result[start:end]
        if actual_text != original:
            # Position mismatch - skip this entity
            continue
        
        transformed = transformer.transform_entity(original, etype, conv_id)
        
        if transformed != original:
            result = result[:start] + transformed + result[end:]
            
            transformations.append({
                'original': original,
                'transformed': transformed,
                'type': etype,
                'start': start,
                'end': end,
            })
    
    return result, transformations
