"""
Checksum-validated rules only.

These rules have ALGORITHMIC CERTAINTY - if they match, it's real.
No fuzzy patterns, no regex-only detection.

Rules:
- Credit Card (Luhn checksum)
- SSN (format + area validation)  
- NPI (Luhn with 80840 prefix)
- DEA (DEA checksum formula)
- IBAN (Mod-97 checksum)
- Canadian SIN (Luhn)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Callable
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


def validate_credit_card(text: str) -> bool:
    """Validate credit card with Luhn and length check."""
    digits = ''.join(d for d in text if d.isdigit())
    
    # Valid CC lengths: 13-19 digits (most are 15-16)
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    # Must pass Luhn
    return luhn_checksum(digits)


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


def validate_dea(dea: str) -> bool:
    """
    Validate DEA number checksum.
    Format: 2 letters + 6 digits + 1 check digit
    Check digit = (sum of digits 1,3,5 + 2*sum of digits 2,4,6) mod 10
    """
    cleaned = ''.join(c for c in dea.upper() if c.isalnum())
    
    match = re.search(r'[A-Z]{2}(\d{7})', cleaned)
    if not match:
        return False
    
    digits = match.group(1)
    
    odd_sum = int(digits[0]) + int(digits[2]) + int(digits[4])
    even_sum = int(digits[1]) + int(digits[3]) + int(digits[5])
    total = odd_sum + (2 * even_sum)
    
    check_digit = total % 10
    return check_digit == int(digits[6])


def validate_iban(iban: str) -> bool:
    """Validate IBAN using mod-97 checksum."""
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
    
    return int(numeric) % 97 == 1


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


# =============================================================================
# RULE DEFINITIONS
# =============================================================================

@dataclass
class ChecksumRule:
    """A rule with algorithmic validation."""
    name: str
    pattern: re.Pattern
    entity_type: EntityType
    validator: Callable[[str], bool]
    confidence: float = 0.99  # Checksum-validated = near-certain
    
    def find_all(self, text: str) -> List[Entity]:
        """Find all matches in text."""
        entities = []
        
        for match in self.pattern.finditer(text):
            matched_text = match.group(0)
            
            # Must pass validation
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


class ChecksumRuleEngine:
    """
    Minimal rule engine - only checksum-validated patterns.
    
    These bypass the meta-classifier because they're algorithmically certain.
    """
    
    def __init__(self):
        self.rules: List[ChecksumRule] = []
        self._build_rules()
    
    def _build_rules(self):
        """Build checksum-validated rules only."""
        
        # =================================================================
        # CREDIT CARD (Luhn validated)
        # =================================================================
        
        # Standard format: 16 digits with optional separators
        self.rules.append(ChecksumRule(
            name="credit_card_standard",
            pattern=re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            entity_type=EntityType.CREDIT_CARD,
            validator=validate_credit_card,
        ))
        
        # Amex format: 15 digits (4-6-5)
        self.rules.append(ChecksumRule(
            name="credit_card_amex",
            pattern=re.compile(r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'),
            entity_type=EntityType.CREDIT_CARD,
            validator=validate_credit_card,
        ))
        
        # Continuous digits 13-19
        self.rules.append(ChecksumRule(
            name="credit_card_continuous",
            pattern=re.compile(r'\b[3-6]\d{12,18}\b'),
            entity_type=EntityType.CREDIT_CARD,
            validator=validate_credit_card,
        ))
        
        # Labeled credit card (context helps)
        self.rules.append(ChecksumRule(
            name="credit_card_labeled",
            pattern=re.compile(r'(?:card|credit|debit|visa|mastercard|amex)[#:\s]*(\d[\d\s-]{12,22}\d)', re.I),
            entity_type=EntityType.CREDIT_CARD,
            validator=validate_credit_card,
        ))
        
        # =================================================================
        # SSN (Format + area validation)
        # =================================================================
        
        # Standard format: XXX-XX-XXXX
        self.rules.append(ChecksumRule(
            name="ssn_dashed",
            pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            entity_type=EntityType.SSN,
            validator=validate_ssn,
        ))
        
        # Spaced format: XXX XX XXXX
        self.rules.append(ChecksumRule(
            name="ssn_spaced",
            pattern=re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),
            entity_type=EntityType.SSN,
            validator=validate_ssn,
        ))
        
        # Labeled SSN (allows continuous digits)
        self.rules.append(ChecksumRule(
            name="ssn_labeled",
            pattern=re.compile(r'(?:SSN|social\s*security)[#:\s]*(\d{9}|\d{3}[-\s]\d{2}[-\s]\d{4})', re.I),
            entity_type=EntityType.SSN,
            validator=validate_ssn,
        ))
        
        # =================================================================
        # NPI (National Provider Identifier)
        # =================================================================
        
        # Labeled NPI
        self.rules.append(ChecksumRule(
            name="npi_labeled",
            pattern=re.compile(r'(?:NPI)[#:\s]*([12]\d{9})', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            validator=validate_npi,
        ))
        
        # Unlabeled but valid NPI (10 digits starting with 1 or 2)
        self.rules.append(ChecksumRule(
            name="npi_unlabeled",
            pattern=re.compile(r'\b[12]\d{9}\b'),
            entity_type=EntityType.ACCOUNT_NUMBER,
            validator=validate_npi,
            confidence=0.95,  # Slightly lower without label
        ))
        
        # =================================================================
        # DEA Number
        # =================================================================
        
        # Labeled DEA
        self.rules.append(ChecksumRule(
            name="dea_labeled",
            pattern=re.compile(r'(?:DEA)[#:\s]*([A-Z]{2}\d{7})', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            validator=validate_dea,
        ))
        
        # Unlabeled DEA pattern
        self.rules.append(ChecksumRule(
            name="dea_unlabeled",
            pattern=re.compile(r'\b[A-Z]{2}\d{7}\b'),
            entity_type=EntityType.ACCOUNT_NUMBER,
            validator=validate_dea,
            confidence=0.95,
        ))
        
        # =================================================================
        # IBAN
        # =================================================================
        
        # Standard IBAN (2 letters + 2 digits + up to 30 alphanumeric)
        self.rules.append(ChecksumRule(
            name="iban",
            pattern=re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9\s]{11,30}\b'),
            entity_type=EntityType.ACCOUNT_NUMBER,
            validator=validate_iban,
        ))
        
        # Labeled IBAN
        self.rules.append(ChecksumRule(
            name="iban_labeled",
            pattern=re.compile(r'(?:IBAN)[#:\s]*([A-Z]{2}\d{2}[A-Z0-9\s]{11,30})', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            validator=validate_iban,
        ))
        
        # =================================================================
        # Canadian SIN
        # =================================================================
        
        # Labeled SIN
        self.rules.append(ChecksumRule(
            name="sin_labeled",
            pattern=re.compile(r'(?:SIN|social\s*insurance)[#:\s]*(\d{3}[-\s]?\d{3}[-\s]?\d{3})', re.I),
            entity_type=EntityType.SSN,  # Map to SSN type
            validator=validate_sin,
        ))
    
    def detect(self, text: str) -> List[Entity]:
        """Run all checksum rules against text."""
        entities = []
        
        for rule in self.rules:
            try:
                matches = rule.find_all(text)
                for entity in matches:
                    entity._detector = "checksum_rule"
                    entity._rule_name = rule.name
                entities.extend(matches)
            except Exception as e:
                logger.error(f"Rule {rule.name} failed: {e}")
        
        # Deduplicate overlapping matches (keep highest confidence)
        entities = self._deduplicate(entities)
        
        return entities
    
    def _deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate/overlapping entities, keep best."""
        if not entities:
            return []
        
        # Sort by start position, then by confidence descending
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        result = []
        last_end = -1
        
        for entity in entities:
            if entity.start >= last_end:
                result.append(entity)
                last_end = entity.end
            elif entity.confidence > result[-1].confidence:
                # Higher confidence overlapping entity - replace
                result[-1] = entity
                last_end = entity.end
        
        return result
    
    def list_rules(self) -> List[dict]:
        """List all rules."""
        return [
            {
                "name": r.name,
                "entity_type": r.entity_type.value,
                "confidence": r.confidence,
            }
            for r in self.rules
        ]


# Convenience alias
RuleEngine = ChecksumRuleEngine
