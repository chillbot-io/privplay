"""Rule-based PHI/PII detection - comprehensive patterns with validation."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Callable
import logging

from ...types import Entity, EntityType, SourceType

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATORS
# =============================================================================

def luhn_checksum(number: str) -> bool:
    """
    Validate number using Luhn algorithm.
    Used for credit cards, NPI, and other checksummed identifiers.
    """
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 10:
        return False
    
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0


def validate_ssn(ssn: str) -> bool:
    """
    Validate SSN format and area number.
    Invalid: 000, 666, 900-999 in area (first 3 digits)
    Invalid: 00 in group (middle 2 digits)
    Invalid: 0000 in serial (last 4 digits)
    """
    digits = ''.join(d for d in ssn if d.isdigit())
    if len(digits) != 9:
        return False
    
    area = int(digits[:3])
    group = int(digits[3:5])
    serial = int(digits[5:])
    
    # Invalid area numbers
    if area == 0 or area == 666 or area >= 900:
        return False
    # Invalid group
    if group == 0:
        return False
    # Invalid serial
    if serial == 0:
        return False
    
    return True


def validate_dea(dea: str) -> bool:
    """
    Validate DEA number checksum.
    Format: 2 letters + 6 digits + 1 check digit
    Check digit = (sum of digits 1,3,5 + 2*sum of digits 2,4,6) mod 10
    """
    # Extract just the alphanumeric part
    cleaned = ''.join(c for c in dea.upper() if c.isalnum())
    
    # Find the DEA pattern: 2 letters + 7 digits
    match = re.search(r'[A-Z]{2}(\d{7})', cleaned)
    if not match:
        return False
    
    digits = match.group(1)
    
    # Calculate checksum
    odd_sum = int(digits[0]) + int(digits[2]) + int(digits[4])
    even_sum = int(digits[1]) + int(digits[3]) + int(digits[5])
    total = odd_sum + (2 * even_sum)
    
    check_digit = total % 10
    return check_digit == int(digits[6])


def validate_npi(npi: str) -> bool:
    """
    Validate NPI using Luhn algorithm with prefix 80840.
    NPI is 10 digits, validated with 80840 prefix making it 15 digits.
    """
    digits = ''.join(d for d in npi if d.isdigit())
    if len(digits) != 10:
        return False
    
    # NPI must start with 1 or 2
    if digits[0] not in '12':
        return False
    
    # Luhn check with 80840 prefix
    full_number = '80840' + digits
    return luhn_checksum(full_number)


def validate_sin(sin: str) -> bool:
    """Validate Canadian SIN using Luhn algorithm (9 digits)."""
    digits = [int(d) for d in sin if d.isdigit()]
    if len(digits) != 9:
        return False
    
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0


def validate_iban(iban: str) -> bool:
    """
    Validate IBAN using mod-97 checksum.
    """
    # Remove spaces and convert to uppercase
    cleaned = iban.replace(' ', '').upper()
    
    if len(cleaned) < 15 or len(cleaned) > 34:
        return False
    
    # Move first 4 chars to end
    rearranged = cleaned[4:] + cleaned[:4]
    
    # Convert letters to numbers (A=10, B=11, etc.)
    numeric = ''
    for char in rearranged:
        if char.isdigit():
            numeric += char
        else:
            numeric += str(ord(char) - 55)
    
    # Check mod 97
    return int(numeric) % 97 == 1


# =============================================================================
# RULE CLASS
# =============================================================================

@dataclass
class Rule:
    """A detection rule with optional validation."""
    name: str
    pattern: Pattern
    entity_type: EntityType
    confidence: float = 0.90
    validator: Optional[Callable[[str], bool]] = None
    
    def find_all(self, text: str) -> List[Entity]:
        """Find all matches in text, applying validator if present."""
        entities = []
        for match in self.pattern.finditer(text):
            matched_text = match.group()
            
            # Apply validator if present
            if self.validator is not None:
                if not self.validator(matched_text):
                    continue
            
            entities.append(Entity(
                text=matched_text,
                start=match.start(),
                end=match.end(),
                entity_type=self.entity_type,
                confidence=self.confidence,
                source=SourceType.RULE,
            ))
        return entities


# =============================================================================
# RULE ENGINE
# =============================================================================

class RuleEngine:
    """Rule-based detection engine with comprehensive PII/PHI patterns."""
    
    def __init__(self):
        self.rules: List[Rule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load comprehensive detection rules."""
        
        # =================================================================
        # GOVERNMENT IDs
        # =================================================================
        
        # US SSN - with validation
        self.add_rule(Rule(
            name="ssn_dashed",
            pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            entity_type=EntityType.SSN,
            confidence=0.99,
            validator=validate_ssn,
        ))
        self.add_rule(Rule(
            name="ssn_spaced",
            pattern=re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),
            entity_type=EntityType.SSN,
            confidence=0.98,
            validator=validate_ssn,
        ))
        self.add_rule(Rule(
            name="ssn_labeled",
            pattern=re.compile(r'(?:SSN|Social\s*Security)[:\s#]*(\d{3}[-\s]?\d{2}[-\s]?\d{4})', re.I),
            entity_type=EntityType.SSN,
            confidence=0.99,
            validator=validate_ssn,
        ))
        
        # US EIN (Employer Identification Number)
        self.add_rule(Rule(
            name="ein",
            pattern=re.compile(r'\b\d{2}-\d{7}\b'),
            entity_type=EntityType.ACCOUNT_NUMBER,
            confidence=0.85,
        ))
        
        # US ITIN (Individual Taxpayer ID) - starts with 9, 4th digit 7-8
        self.add_rule(Rule(
            name="itin",
            pattern=re.compile(r'\b9\d{2}-[78]\d-\d{4}\b'),
            entity_type=EntityType.SSN,
            confidence=0.95,
        ))
        
        # UK National Insurance Number
        self.add_rule(Rule(
            name="uk_nino",
            pattern=re.compile(r'\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b', re.I),
            entity_type=EntityType.SSN,
            confidence=0.95,
        ))
        
        # Canadian SIN (Social Insurance Number) - with Luhn validation
        self.add_rule(Rule(
            name="canada_sin",
            pattern=re.compile(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b'),
            entity_type=EntityType.SSN,
            confidence=0.92,
            validator=validate_sin,
        ))
        
        # Passport (generic patterns)
        self.add_rule(Rule(
            name="passport_labeled",
            pattern=re.compile(r'(?:passport)[:\s#]*[A-Z]{1,2}\d{6,9}', re.I),
            entity_type=EntityType.PASSPORT,
            confidence=0.95,
        ))
        self.add_rule(Rule(
            name="passport_us",
            pattern=re.compile(r'\b[A-Z]\d{8}\b'),  # US format
            entity_type=EntityType.PASSPORT,
            confidence=0.75,
        ))
        
        # Driver's License
        self.add_rule(Rule(
            name="drivers_license_labeled",
            pattern=re.compile(r"(?:DL|Driver'?s?\s*(?:License|Lic\.?))[:\s#]*[A-Z0-9]{5,15}", re.I),
            entity_type=EntityType.DRIVER_LICENSE,
            confidence=0.92,
        ))
        
        # =================================================================
        # FINANCIAL - Credit Cards with Luhn validation
        # =================================================================
        
        # Credit Cards - Visa (16 digits, starts with 4)
        self.add_rule(Rule(
            name="cc_visa",
            pattern=re.compile(r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            entity_type=EntityType.CREDIT_CARD,
            confidence=0.99,
            validator=luhn_checksum,
        ))
        
        # Credit Cards - Mastercard (16 digits, starts with 51-55 or 2221-2720)
        self.add_rule(Rule(
            name="cc_mastercard",
            pattern=re.compile(r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            entity_type=EntityType.CREDIT_CARD,
            confidence=0.99,
            validator=luhn_checksum,
        ))
        
        # Credit Cards - Amex (15 digits, starts with 34 or 37)
        self.add_rule(Rule(
            name="cc_amex",
            pattern=re.compile(r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'),
            entity_type=EntityType.CREDIT_CARD,
            confidence=0.99,
            validator=luhn_checksum,
        ))
        
        # Credit Cards - Discover (16 digits, starts with 6011, 65, 644-649)
        self.add_rule(Rule(
            name="cc_discover",
            pattern=re.compile(r'\b6(?:011|5\d{2}|4[4-9]\d)[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            entity_type=EntityType.CREDIT_CARD,
            confidence=0.99,
            validator=luhn_checksum,
        ))
        
        # Credit Cards - Generic 16 digit with Luhn (fallback)
        self.add_rule(Rule(
            name="cc_generic",
            pattern=re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            entity_type=EntityType.CREDIT_CARD,
            confidence=0.95,
            validator=luhn_checksum,
        ))
        
        # IBAN (International Bank Account Number) with validation
        self.add_rule(Rule(
            name="iban",
            pattern=re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b'),
            entity_type=EntityType.BANK_ACCOUNT,
            confidence=0.95,
            validator=validate_iban,
        ))
        
        # SWIFT/BIC Code
        self.add_rule(Rule(
            name="swift_bic",
            pattern=re.compile(r'\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b'),
            entity_type=EntityType.BANK_ACCOUNT,
            confidence=0.88,
        ))
        
        # US Bank Routing Number (9 digits, specific checksum pattern)
        self.add_rule(Rule(
            name="routing_number_labeled",
            pattern=re.compile(r'(?:routing|ABA|RTN)[:\s#]*\d{9}\b', re.I),
            entity_type=EntityType.BANK_ACCOUNT,
            confidence=0.92,
        ))
        
        # Account numbers (labeled)
        self.add_rule(Rule(
            name="account_labeled",
            pattern=re.compile(r'(?:Acct?\.?|Account|Bank\s*Account)[:\s#]*\d{6,17}', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            confidence=0.90,
        ))
        
        # Crypto - Bitcoin address
        self.add_rule(Rule(
            name="bitcoin_address",
            pattern=re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'),
            entity_type=EntityType.OTHER,
            confidence=0.90,
        ))
        
        # Crypto - Bitcoin Bech32 (bc1)
        self.add_rule(Rule(
            name="bitcoin_bech32",
            pattern=re.compile(r'\bbc1[a-zA-HJ-NP-Z0-9]{39,59}\b'),
            entity_type=EntityType.OTHER,
            confidence=0.92,
        ))
        
        # Crypto - Ethereum address
        self.add_rule(Rule(
            name="ethereum_address",
            pattern=re.compile(r'\b0x[a-fA-F0-9]{40}\b'),
            entity_type=EntityType.OTHER,
            confidence=0.95,
        ))
        
        # =================================================================
        # CONTACT INFO
        # =================================================================
        
        # Phone - US format
        self.add_rule(Rule(
            name="phone_us",
            pattern=re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            entity_type=EntityType.PHONE,
            confidence=0.90,
        ))
        
        # Phone - International
        self.add_rule(Rule(
            name="phone_intl",
            pattern=re.compile(r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
            entity_type=EntityType.PHONE,
            confidence=0.88,
        ))
        
        # Phone - labeled
        self.add_rule(Rule(
            name="phone_labeled",
            pattern=re.compile(r'(?:phone|tel|cell|mobile|fax)[:\s#]*[\d\-\.\s\(\)]{7,20}', re.I),
            entity_type=EntityType.PHONE,
            confidence=0.92,
        ))
        
        # Fax - labeled
        self.add_rule(Rule(
            name="fax_labeled",
            pattern=re.compile(r'(?:fax)[:\s#]*[\d\-\.\s\(\)]{7,20}', re.I),
            entity_type=EntityType.FAX,
            confidence=0.95,
        ))
        
        # Email
        self.add_rule(Rule(
            name="email",
            pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            entity_type=EntityType.EMAIL,
            confidence=0.99,
        ))
        
        # =================================================================
        # DIGITAL IDENTIFIERS
        # =================================================================
        
        # IP Address v4
        self.add_rule(Rule(
            name="ipv4",
            pattern=re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
            entity_type=EntityType.IP_ADDRESS,
            confidence=0.95,
        ))
        
        # IP Address v6
        self.add_rule(Rule(
            name="ipv6",
            pattern=re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
            entity_type=EntityType.IP_ADDRESS,
            confidence=0.95,
        ))
        
        # IP Address v6 abbreviated
        self.add_rule(Rule(
            name="ipv6_abbrev",
            pattern=re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){2,7}:[0-9a-fA-F]{1,4}\b'),
            entity_type=EntityType.IP_ADDRESS,
            confidence=0.90,
        ))
        
        # MAC Address
        self.add_rule(Rule(
            name="mac_address_colon",
            pattern=re.compile(r'\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b'),
            entity_type=EntityType.MAC_ADDRESS,
            confidence=0.95,
        ))
        self.add_rule(Rule(
            name="mac_address_dash",
            pattern=re.compile(r'\b(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}\b'),
            entity_type=EntityType.MAC_ADDRESS,
            confidence=0.95,
        ))
        
        # URL
        self.add_rule(Rule(
            name="url",
            pattern=re.compile(r'https?://[^\s<>"\']+'),
            entity_type=EntityType.URL,
            confidence=0.95,
        ))
        
        # Username/handle patterns
        self.add_rule(Rule(
            name="username_labeled",
            pattern=re.compile(r'(?:user(?:name)?|login|handle|screen\s*name)[:\s#]*[A-Za-z0-9_\-\.]{3,30}', re.I),
            entity_type=EntityType.USERNAME,
            confidence=0.88,
        ))
        self.add_rule(Rule(
            name="social_handle",
            pattern=re.compile(r'@[A-Za-z][A-Za-z0-9_]{2,29}\b'),
            entity_type=EntityType.USERNAME,
            confidence=0.85,
        ))
        
        # =================================================================
        # DEVICE IDENTIFIERS
        # =================================================================
        
        # VIN (Vehicle Identification Number) - 17 chars, no I, O, Q
        self.add_rule(Rule(
            name="vin",
            pattern=re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
            entity_type=EntityType.DEVICE_ID,
            confidence=0.88,
        ))
        
        # VIN labeled
        self.add_rule(Rule(
            name="vin_labeled",
            pattern=re.compile(r'(?:VIN|Vehicle\s*ID)[:\s#]*[A-HJ-NPR-Z0-9]{17}', re.I),
            entity_type=EntityType.DEVICE_ID,
            confidence=0.95,
        ))
        
        # IMEI (15 digits)
        self.add_rule(Rule(
            name="imei",
            pattern=re.compile(r'\b\d{15}\b'),
            entity_type=EntityType.DEVICE_ID,
            confidence=0.70,  # Low - many 15-digit numbers aren't IMEI
        ))
        
        # IMEI labeled
        self.add_rule(Rule(
            name="imei_labeled",
            pattern=re.compile(r'(?:IMEI)[:\s#]*\d{15}', re.I),
            entity_type=EntityType.DEVICE_ID,
            confidence=0.95,
        ))
        
        # Serial Number (labeled)
        self.add_rule(Rule(
            name="serial_number",
            pattern=re.compile(r'(?:serial|S/N|SN)[:\s#]*[A-Z0-9]{6,20}', re.I),
            entity_type=EntityType.DEVICE_ID,
            confidence=0.88,
        ))
        
        # License Plate (US - varies by state but common patterns)
        self.add_rule(Rule(
            name="license_plate_us",
            pattern=re.compile(r'\b[A-Z]{1,3}[-\s]?\d{1,4}[-\s]?[A-Z]{0,3}\b'),
            entity_type=EntityType.DEVICE_ID,
            confidence=0.70,  # Low - many FPs
        ))
        
        # License Plate labeled
        self.add_rule(Rule(
            name="license_plate_labeled",
            pattern=re.compile(r'(?:license\s*plate|plate\s*(?:no|number|#)|tag)[:\s#]*[A-Z0-9]{2,8}', re.I),
            entity_type=EntityType.DEVICE_ID,
            confidence=0.90,
        ))
        
        # =================================================================
        # MEDICAL / PHI
        # =================================================================
        
        # MRN (labeled only - avoids FPs)
        self.add_rule(Rule(
            name="mrn_labeled",
            pattern=re.compile(r'(?:MRN|MR#|Medical\s*Record|Patient\s*ID|Chart)[:\s#]*\d{5,12}', re.I),
            entity_type=EntityType.MRN,
            confidence=0.95,
        ))
        
        # DEA Number (prescriber ID) - with validation
        self.add_rule(Rule(
            name="dea_number",
            pattern=re.compile(r'\b(?:DEA)[:\s#]*[A-Z]{2}\d{7}\b', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            confidence=0.99,
            validator=validate_dea,
        ))
        
        # NPI (National Provider Identifier) - 10 digits starting with 1 or 2, with Luhn
        self.add_rule(Rule(
            name="npi_labeled",
            pattern=re.compile(r'(?:NPI)[:\s#]*[12]\d{9}\b', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            confidence=0.99,
            validator=validate_npi,
        ))
        
        # NPI unlabeled (10 digits starting with 1 or 2) - lower confidence
        self.add_rule(Rule(
            name="npi_unlabeled",
            pattern=re.compile(r'\b[12]\d{9}\b'),
            entity_type=EntityType.ACCOUNT_NUMBER,
            confidence=0.85,
            validator=validate_npi,
        ))
        
        # Health Plan ID (labeled)
        self.add_rule(Rule(
            name="health_plan_id",
            pattern=re.compile(r'(?:member|policy|group|subscriber|plan)\s*(?:ID|#|number)[:\s#]*[A-Z0-9]{6,20}', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            confidence=0.90,
        ))
        
        # =================================================================
        # DATES & AGE
        # =================================================================
        
        # Date MM/DD/YYYY or MM-DD-YYYY
        self.add_rule(Rule(
            name="date_mdy_slash",
            pattern=re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
            entity_type=EntityType.DATE,
            confidence=0.85,
        ))
        self.add_rule(Rule(
            name="date_mdy_dash",
            pattern=re.compile(r'\b\d{1,2}-\d{1,2}-\d{2,4}\b'),
            entity_type=EntityType.DATE,
            confidence=0.85,
        ))
        
        # Date YYYY-MM-DD (ISO format)
        self.add_rule(Rule(
            name="date_iso",
            pattern=re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
            entity_type=EntityType.DATE,
            confidence=0.90,
        ))
        
        # Date written out
        self.add_rule(Rule(
            name="date_written",
            pattern=re.compile(
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', 
                re.I
            ),
            entity_type=EntityType.DATE,
            confidence=0.92,
        ))
        
        # Date European format (DD.MM.YYYY)
        self.add_rule(Rule(
            name="date_european",
            pattern=re.compile(r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b'),
            entity_type=EntityType.DATE,
            confidence=0.82,
        ))
        
        # DOB (labeled)
        self.add_rule(Rule(
            name="dob_labeled",
            pattern=re.compile(
                r'(?:DOB|D\.O\.B\.?|Date\s*of\s*Birth|Birth\s*Date|Born)[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
                re.I
            ),
            entity_type=EntityType.DATE_DOB,
            confidence=0.96,
        ))
        
        # Age patterns
        self.add_rule(Rule(
            name="age_years_old",
            pattern=re.compile(r'\b(\d{1,3})\s*(?:y\.?o\.?|years?\s*old|yr\.?\s*old)\b', re.I),
            entity_type=EntityType.AGE,
            confidence=0.90,
        ))
        self.add_rule(Rule(
            name="age_labeled",
            pattern=re.compile(r'\b(?:age|aged)[:\s]*(\d{1,3})\b', re.I),
            entity_type=EntityType.AGE,
            confidence=0.92,
        ))
        self.add_rule(Rule(
            name="age_month_old",
            pattern=re.compile(r'\b(\d{1,2})\s*(?:month|mo)s?\s*old\b', re.I),
            entity_type=EntityType.AGE,
            confidence=0.90,
        ))
        
        # =================================================================
        # LOCATION
        # =================================================================
        
        # ZIP code (US 5 or 5+4)
        self.add_rule(Rule(
            name="zip_code_plus4",
            pattern=re.compile(r'\b\d{5}-\d{4}\b'),
            entity_type=EntityType.ZIP,
            confidence=0.95,  # High - plus-4 is almost always a ZIP
        ))
        self.add_rule(Rule(
            name="zip_code_5",
            pattern=re.compile(r'\b\d{5}\b'),
            entity_type=EntityType.ZIP,
            confidence=0.65,  # Lower - many false positives
        ))
        
        # UK Postcode
        self.add_rule(Rule(
            name="uk_postcode",
            pattern=re.compile(r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b', re.I),
            entity_type=EntityType.ZIP,
            confidence=0.90,
        ))
        
        # Canadian Postal Code
        self.add_rule(Rule(
            name="canada_postal",
            pattern=re.compile(r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b', re.I),
            entity_type=EntityType.ZIP,
            confidence=0.92,
        ))
        
        # GPS Coordinates
        self.add_rule(Rule(
            name="gps_coordinates",
            pattern=re.compile(r'\b-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}\b'),
            entity_type=EntityType.ADDRESS,
            confidence=0.90,
        ))
    
    def add_rule(self, rule: Rule):
        """Add a detection rule."""
        self.rules.append(rule)
    
    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False
    
    def get_rule(self, name: str) -> Optional[Rule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None
    
    def detect(self, text: str) -> List[Entity]:
        """Run all rules against text."""
        entities = []
        
        for rule in self.rules:
            try:
                matches = rule.find_all(text)
                entities.extend(matches)
            except Exception as e:
                logger.error(f"Rule {rule.name} failed: {e}")
        
        return entities
    
    def add_pattern(
        self, 
        name: str, 
        pattern: str, 
        entity_type: EntityType,
        confidence: float = 0.90,
        validator: Optional[Callable[[str], bool]] = None,
    ):
        """Add a new pattern rule."""
        self.add_rule(Rule(
            name=name,
            pattern=re.compile(pattern),
            entity_type=entity_type,
            confidence=confidence,
            validator=validator,
        ))
    
    def list_rules(self) -> List[dict]:
        """List all rules with their details."""
        return [
            {
                "name": r.name,
                "entity_type": r.entity_type.value,
                "confidence": r.confidence,
                "pattern": r.pattern.pattern,
                "has_validator": r.validator is not None,
            }
            for r in self.rules
        ]
