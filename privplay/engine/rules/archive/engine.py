"""Rule-based PHI/PII detection."""

import re
from dataclasses import dataclass
from typing import List, Optional, Pattern
import logging

from ...types import Entity, EntityType, SourceType

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A detection rule."""
    name: str
    pattern: Pattern
    entity_type: EntityType
    confidence: float = 0.90
    
    def find_all(self, text: str) -> List[Entity]:
        """Find all matches in text."""
        entities = []
        for match in self.pattern.finditer(text):
            entities.append(Entity(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                entity_type=self.entity_type,
                confidence=self.confidence,
                source=SourceType.RULE,
            ))
        return entities


class RuleEngine:
    """Rule-based detection engine."""
    
    def __init__(self):
        self.rules: List[Rule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default detection rules."""
        
        # SSN patterns
        self.add_rule(Rule(
            name="ssn_dashed",
            pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            entity_type=EntityType.SSN,
            confidence=0.99,
        ))
        self.add_rule(Rule(
            name="ssn_spaced",
            pattern=re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),
            entity_type=EntityType.SSN,
            confidence=0.98,
        ))
        
        # Phone patterns
        self.add_rule(Rule(
            name="phone_us",
            pattern=re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            entity_type=EntityType.PHONE,
            confidence=0.92,
        ))
        
        # Fax (usually labeled)
        self.add_rule(Rule(
            name="fax_labeled",
            pattern=re.compile(r'(?:fax|facsimile)[:\s]*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', re.I),
            entity_type=EntityType.FAX,
            confidence=0.95,
        ))
        
        # Email
        self.add_rule(Rule(
            name="email",
            pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            entity_type=EntityType.EMAIL,
            confidence=0.98,
        ))
        
        # Date patterns
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
        self.add_rule(Rule(
            name="date_ymd",
            pattern=re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
            entity_type=EntityType.DATE,
            confidence=0.88,
        ))
        self.add_rule(Rule(
            name="date_written",
            pattern=re.compile(
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b', 
                re.I
            ),
            entity_type=EntityType.DATE,
            confidence=0.90,
        ))
        
        # DOB (labeled dates)
        self.add_rule(Rule(
            name="dob_labeled",
            pattern=re.compile(
                r'(?:DOB|D\.O\.B\.|Date of Birth|Birth\s*Date)[:\s]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                re.I
            ),
            entity_type=EntityType.DATE_DOB,
            confidence=0.95,
        ))
        
        # Age >89 (HIPAA PHI)
        self.add_rule(Rule(
            name="age_over_89",
            pattern=re.compile(r'\b(?:age[d]?[:\s]*)?(?:9\d|1\d{2})\s*(?:y\.?o\.?|years?\s*old)\b', re.I),
            entity_type=EntityType.AGE,
            confidence=0.92,
        ))
        
        # MRN patterns (various formats)
        self.add_rule(Rule(
            name="mrn_labeled",
            pattern=re.compile(r'(?:MRN|MR#|Medical Record)[:\s#]*\d{6,10}', re.I),
            entity_type=EntityType.MRN,
            confidence=0.95,
        ))
        
        # Credit card (basic pattern)
        self.add_rule(Rule(
            name="credit_card",
            pattern=re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            entity_type=EntityType.CREDIT_CARD,
            confidence=0.88,
        ))
        
        # IP Address
        self.add_rule(Rule(
            name="ip_address",
            pattern=re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            entity_type=EntityType.IP_ADDRESS,
            confidence=0.90,
        ))
        
        # URL
        self.add_rule(Rule(
            name="url",
            pattern=re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            entity_type=EntityType.URL,
            confidence=0.95,
        ))
        
        # ZIP code (5 or 5+4)
        self.add_rule(Rule(
            name="zip_code",
            pattern=re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            entity_type=EntityType.ZIP,
            confidence=0.70,  # Lower confidence, many false positives
        ))
        
        # Driver's license (varies by state, this is generic)
        self.add_rule(Rule(
            name="drivers_license_labeled",
            pattern=re.compile(r"(?:DL|Driver'?s?\s*(?:License|Lic\.?))[:\s#]*[A-Z]?\d{5,12}", re.I),
            entity_type=EntityType.DRIVER_LICENSE,
            confidence=0.90,
        ))
        
        # Account numbers (labeled)
        self.add_rule(Rule(
            name="account_labeled",
            pattern=re.compile(r'(?:Acct?\.?|Account)[:\s#]*\d{6,12}', re.I),
            entity_type=EntityType.ACCOUNT_NUMBER,
            confidence=0.88,
        ))
        
        # === HIPAA #12: Vehicle Identification Number (VIN) ===
        # 17 characters: letters (except I, O, Q) and digits
        self.add_rule(Rule(
            name="vin",
            pattern=re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),
            entity_type=EntityType.VIN,
            confidence=0.85,
        ))
        self.add_rule(Rule(
            name="vin_labeled",
            pattern=re.compile(r'(?:VIN|Vehicle\s*(?:ID|Identification))[:\s#]*[A-HJ-NPR-Z0-9]{17}', re.I),
            entity_type=EntityType.VIN,
            confidence=0.95,
        ))
        
        # === HIPAA #13: Unique Device Identifier (UDI) ===
        # GS1 format: (01) followed by 14-digit GTIN
        self.add_rule(Rule(
            name="udi_gs1",
            pattern=re.compile(r'\(01\)\d{14}(?:\([\d]+\)[A-Za-z0-9]+)*'),
            entity_type=EntityType.UDI,
            confidence=0.92,
        ))
        # HIBCC format: + followed by alphanumeric
        self.add_rule(Rule(
            name="udi_hibcc",
            pattern=re.compile(r'\+[A-Z]\d{3}[A-Z0-9]{5,15}'),
            entity_type=EntityType.UDI,
            confidence=0.90,
        ))
        # Labeled device identifiers
        self.add_rule(Rule(
            name="udi_labeled",
            pattern=re.compile(r'(?:UDI|Device\s*ID|Serial\s*(?:Number|No\.?|#))[:\s#]*[A-Z0-9\-]{8,30}', re.I),
            entity_type=EntityType.UDI,
            confidence=0.88,
        ))
        
        # === HIPAA #9: Health Plan Beneficiary Number ===
        # Medicare Beneficiary Identifier (MBI): 11 chars, specific format
        self.add_rule(Rule(
            name="medicare_mbi",
            pattern=re.compile(r'\b[1-9][AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d[AC-HJKMNP-RT-Y]{2}\d{2}\b'),
            entity_type=EntityType.HEALTH_PLAN_ID,
            confidence=0.90,
        ))
        # Labeled health plan IDs
        self.add_rule(Rule(
            name="health_plan_id_labeled",
            pattern=re.compile(
                r'(?:Member\s*(?:ID|#|Number)|Subscriber\s*(?:ID|#)|Policy\s*(?:#|Number)|'
                r'Group\s*(?:#|Number)|Beneficiary\s*(?:ID|#)|Insurance\s*(?:ID|#)|'
                r'Health\s*Plan\s*(?:ID|#))[:\s]*[A-Z0-9\-]{6,20}',
                re.I
            ),
            entity_type=EntityType.HEALTH_PLAN_ID,
            confidence=0.92,
        ))
        
        # === HIPAA #11: DEA Number ===
        # Format: 2 letters + 7 digits (with checksum)
        self.add_rule(Rule(
            name="dea_number",
            pattern=re.compile(r'\b[ABCDEFGHJKLMPRSTUX][A-Z]\d{7}\b'),
            entity_type=EntityType.DEA_NUMBER,
            confidence=0.92,
        ))
        self.add_rule(Rule(
            name="dea_labeled",
            pattern=re.compile(r'(?:DEA|DEA\s*#|DEA\s*Number)[:\s#]*[ABCDEFGHJKLMPRSTUX][A-Z]\d{7}', re.I),
            entity_type=EntityType.DEA_NUMBER,
            confidence=0.98,
        ))
        
        # === HIPAA #11: State Medical License ===
        self.add_rule(Rule(
            name="medical_license_labeled",
            pattern=re.compile(
                r'(?:Medical\s*License|License\s*(?:#|Number|No\.?)|'
                r'(?:MD|DO|NP|PA|RN|LPN|APRN)\s*License|'
                r'State\s*License)[:\s#]*[A-Z]{0,3}\d{4,12}',
                re.I
            ),
            entity_type=EntityType.MEDICAL_LICENSE,
            confidence=0.90,
        ))
        # State-prefixed license (e.g., CA12345, NY-MD-12345)
        self.add_rule(Rule(
            name="medical_license_state_prefix",
            pattern=re.compile(r'\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)[-\s]?(?:MD|DO|NP|PA|RN)?[-\s]?\d{4,10}\b'),
            entity_type=EntityType.MEDICAL_LICENSE,
            confidence=0.85,
        ))
        
        # === NPI (National Provider Identifier) ===
        # 10-digit number starting with 1 or 2, with Luhn checksum
        self.add_rule(Rule(
            name="npi_labeled",
            pattern=re.compile(r'(?:NPI|National\s*Provider)[:\s#]*[12]\d{9}\b', re.I),
            entity_type=EntityType.NPI,
            confidence=0.95,
        ))
        # Bare NPI (lower confidence due to false positive risk)
        self.add_rule(Rule(
            name="npi_bare",
            pattern=re.compile(r'\b[12]\d{9}\b'),
            entity_type=EntityType.NPI,
            confidence=0.60,  # Low confidence - needs context
        ))
    
    @property
    def patterns(self) -> List[Rule]:
        """Alias for rules list (used by classifier stack status)."""
        return self.rules
    
    def add_rule(self, rule: Rule):
        """Add a detection rule."""
        self.rules.append(rule)
    
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
        confidence: float = 0.90
    ):
        """Add a new pattern rule."""
        self.add_rule(Rule(
            name=name,
            pattern=re.compile(pattern),
            entity_type=entity_type,
            confidence=confidence,
        ))
