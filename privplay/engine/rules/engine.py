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


# Valid US ZIP code 3-digit prefixes
# Source: USPS, covers all assigned prefixes including territories
# Unassigned ranges are excluded (e.g., 000-004, 213, 269, etc.)
VALID_ZIP_PREFIXES = {
    # 0xx - Northeast (CT, MA, ME, NH, NJ, PR, RI, VT, VI)
    '005', '006', '007', '008', '009',  # PR, VI
    '010', '011', '012', '013', '014', '015', '016', '017', '018', '019',  # MA
    '020', '021', '022', '023', '024', '025', '026', '027',  # MA
    '028', '029',  # RI
    '030', '031', '032', '033', '034', '035', '036', '037', '038',  # NH
    '039',  # ME
    '040', '041', '042', '043', '044', '045', '046', '047', '048', '049',  # ME
    '050', '051', '052', '053', '054', '056', '057', '058', '059',  # VT
    '060', '061', '062', '063', '064', '065', '066', '067', '068', '069',  # CT
    '070', '071', '072', '073', '074', '075', '076', '077', '078', '079',  # NJ
    '080', '081', '082', '083', '084', '085', '086', '087', '088', '089',  # NJ
    '090', '091', '092', '093', '094', '095', '096', '097', '098', '099',  # Military APO/FPO
    
    # 1xx - NY, PA, DE
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',  # NY
    '110', '111', '112', '113', '114', '115', '116', '117', '118', '119',  # NY
    '120', '121', '122', '123', '124', '125', '126', '127', '128', '129',  # NY
    '130', '131', '132', '133', '134', '135', '136', '137', '138', '139',  # NY
    '140', '141', '142', '143', '144', '145', '146', '147', '148', '149',  # NY
    '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',  # PA
    '160', '161', '162', '163', '164', '165', '166', '167', '168', '169',  # PA
    '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',  # PA
    '180', '181', '182', '183', '184', '185', '186', '187', '188', '189',  # PA
    '190', '191', '192', '193', '194', '195', '196', '197', '198', '199',  # PA, DE
    
    # 2xx - DC, MD, NC, SC, VA, WV
    '200', '201', '202', '203', '204', '205', '206', '207', '208', '209',  # DC, MD, VA
    '210', '211', '212', '214', '215', '216', '217', '218', '219',  # MD
    '220', '221', '222', '223', '224', '225', '226', '227', '228', '229',  # VA
    '230', '231', '232', '233', '234', '235', '236', '237', '238', '239',  # VA
    '240', '241', '242', '243', '244', '245', '246', '247', '248', '249',  # VA, WV
    '250', '251', '252', '253', '254', '255', '256', '257', '258', '259',  # WV
    '260', '261', '262', '263', '264', '265', '266', '267', '268',  # WV
    '270', '271', '272', '273', '274', '275', '276', '277', '278', '279',  # NC
    '280', '281', '282', '283', '284', '285', '286', '287', '288', '289',  # NC
    '290', '291', '292', '293', '294', '295', '296', '297', '298', '299',  # SC
    
    # 3xx - AL, FL, GA, MS, TN
    '300', '301', '302', '303', '304', '305', '306', '307', '308', '309',  # GA
    '310', '311', '312', '313', '314', '315', '316', '317', '318', '319',  # GA
    '320', '321', '322', '323', '324', '325', '326', '327', '328', '329',  # FL
    '330', '331', '332', '333', '334', '335', '336', '337', '338', '339',  # FL
    '340', '341', '342', '344', '346', '347', '349',  # FL, VI
    '350', '351', '352', '354', '355', '356', '357', '358', '359',  # AL
    '360', '361', '362', '363', '364', '365', '366', '367', '368', '369',  # AL
    '370', '371', '372', '373', '374', '375', '376', '377', '378', '379',  # TN
    '380', '381', '382', '383', '384', '385',  # TN
    '386', '387', '388', '389',  # MS
    '390', '391', '392', '393', '394', '395', '396', '397',  # MS
    '398', '399',  # GA (military)
    
    # 4xx - IN, KY, MI, OH
    '400', '401', '402', '403', '404', '405', '406', '407', '408', '409',  # KY
    '410', '411', '412', '413', '414', '415', '416', '417', '418', '419',  # KY
    '420', '421', '422', '423', '424', '425', '426', '427',  # KY
    '430', '431', '432', '433', '434', '435', '436', '437', '438', '439',  # OH
    '440', '441', '442', '443', '444', '445', '446', '447', '448', '449',  # OH
    '450', '451', '452', '453', '454', '455', '456', '457', '458', '459',  # OH
    '460', '461', '462', '463', '464', '465', '466', '467', '468', '469',  # IN
    '470', '471', '472', '473', '474', '475', '476', '477', '478', '479',  # IN
    '480', '481', '482', '483', '484', '485', '486', '487', '488', '489',  # MI
    '490', '491', '492', '493', '494', '495', '496', '497', '498', '499',  # MI
    
    # 5xx - IA, MN, MT, ND, SD, WI
    '500', '501', '502', '503', '504', '505', '506', '507', '508', '509',  # IA
    '510', '511', '512', '513', '514', '515', '516', '520', '521', '522',  # IA
    '523', '524', '525', '526', '527', '528',  # IA
    '530', '531', '532', '534', '535', '537', '538', '539',  # WI
    '540', '541', '542', '543', '544', '545', '546', '547', '548', '549',  # WI
    '550', '551', '553', '554', '555', '556', '557', '558', '559',  # MN
    '560', '561', '562', '563', '564', '565', '566', '567',  # MN
    '570', '571', '572', '573', '574', '575', '576', '577',  # SD
    '580', '581', '582', '583', '584', '585', '586', '587', '588',  # ND
    '590', '591', '592', '593', '594', '595', '596', '597', '598', '599',  # MT
    
    # 6xx - IL, KS, MO, NE
    '600', '601', '602', '603', '604', '605', '606', '607', '608', '609',  # IL
    '610', '611', '612', '613', '614', '615', '616', '617', '618', '619',  # IL
    '620', '621', '622', '623', '624', '625', '626', '627', '628', '629',  # IL
    '630', '631', '632', '633', '634', '635', '636', '637', '638', '639',  # MO
    '640', '641', '644', '645', '646', '647', '648', '649',  # MO
    '650', '651', '652', '653', '654', '655', '656', '657', '658', '659',  # MO
    '660', '661', '662', '664', '665', '666', '667', '668', '669',  # KS
    '670', '671', '672', '673', '674', '675', '676', '677', '678', '679',  # KS
    '680', '681', '683', '684', '685', '686', '687', '688', '689',  # NE
    '690', '691', '692', '693',  # NE
    
    # 7xx - AR, LA, OK, TX
    '700', '701', '703', '704', '705', '706', '707', '708',  # LA
    '710', '711', '712', '713', '714',  # LA
    '716', '717', '718', '719',  # AR
    '720', '721', '722', '723', '724', '725', '726', '727', '728', '729',  # AR
    '730', '731', '733', '734', '735', '736', '737', '738', '739',  # OK
    '740', '741', '743', '744', '745', '746', '747', '748', '749',  # OK
    '750', '751', '752', '753', '754', '755', '756', '757', '758', '759',  # TX
    '760', '761', '762', '763', '764', '765', '766', '767', '768', '769',  # TX
    '770', '771', '772', '773', '774', '775', '776', '777', '778', '779',  # TX
    '780', '781', '782', '783', '784', '785', '786', '787', '788', '789',  # TX
    '790', '791', '792', '793', '794', '795', '796', '797', '798', '799',  # TX
    
    # 8xx - AZ, CO, ID, NM, NV, UT, WY
    '800', '801', '802', '803', '804', '805', '806', '807', '808', '809',  # CO
    '810', '811', '812', '813', '814', '815', '816',  # CO
    '820', '821', '822', '823', '824', '825', '826', '827', '828', '829',  # WY
    '830', '831', '832', '833', '834', '835', '836', '837', '838',  # ID
    '840', '841', '842', '843', '844', '845', '846', '847',  # UT
    '850', '851', '852', '853', '855', '856', '857', '859',  # AZ
    '860', '863', '864', '865',  # AZ
    '870', '871', '872', '873', '874', '875', '877', '878', '879',  # NM
    '880', '881', '882', '883', '884',  # NM, TX
    '889', '890', '891', '893', '894', '895', '897', '898',  # NV
    
    # 9xx - AK, AS, CA, GU, HI, OR, WA, Military APO/FPO
    '900', '901', '902', '903', '904', '905', '906', '907', '908',  # CA
    '910', '911', '912', '913', '914', '915', '916', '917', '918',  # CA
    '919', '920', '921', '922', '923', '924', '925', '926', '927', '928',  # CA
    '930', '931', '932', '933', '934', '935', '936', '937', '938', '939',  # CA
    '940', '941', '942', '943', '944', '945', '946', '947', '948', '949',  # CA
    '950', '951', '952', '953', '954', '955', '956', '957', '958', '959',  # CA
    '960', '961',  # CA, military
    '962', '963', '964', '965', '966',  # Military APO/FPO Pacific
    '967', '968',  # HI
    '969',  # GU, AS, other Pacific
    '970', '971', '972', '973', '974', '975', '976', '977', '978', '979',  # OR
    '980', '981', '982', '983', '984', '985', '986', '988', '989',  # WA
    '990', '991', '992', '993', '994',  # WA
    '995', '996', '997', '998', '999',  # AK
}


def validate_zip(zip_code: str) -> bool:
    """
    Validate US ZIP code using prefix lookup.
    Accepts 5-digit or 9-digit (ZIP+4) formats.
    """
    digits = ''.join(d for d in zip_code if d.isdigit())
    if len(digits) not in (5, 9):
        return False
    return digits[:3] in VALID_ZIP_PREFIXES


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
        """Find all matches in text, applying validator if present.
        
        If the pattern has a capture group, extracts the first group as the entity text.
        This allows patterns like r'(?:SSN|Social\s*Security)[:\s#]*(\d{3}-\d{2}-\d{4})'
        to capture just the SSN number, not the label.
        """
        entities = []
        for match in self.pattern.finditer(text):
            # Check if there's a capture group
            if match.lastindex and match.lastindex >= 1:
                # Use capture group 1 as the entity text
                matched_text = match.group(1)
                # Adjust start/end to the capture group span
                start, end = match.span(1)
            else:
                # No capture group, use full match
                matched_text = match.group()
                start, end = match.start(), match.end()
            
            # Apply validator if present
            if self.validator is not None:
                if not self.validator(matched_text):
                    continue
            
            entities.append(Entity(
                text=matched_text,
                start=start,
                end=end,
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
            pattern=re.compile(r'(?:SWIFT|BIC)[:\s#]*[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?', re.I),
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
            entity_type=EntityType.CRYPTO_ADDRESS,
            confidence=0.90,
        ))
        
        # Crypto - Bitcoin Bech32 (bc1)
        self.add_rule(Rule(
            name="bitcoin_bech32",
            pattern=re.compile(r'\bbc1[a-zA-HJ-NP-Z0-9]{39,59}\b'),
            entity_type=EntityType.CRYPTO_ADDRESS,
            confidence=0.92,
        ))
        
        # Crypto - Ethereum address
        self.add_rule(Rule(
            name="ethereum_address",
            pattern=re.compile(r'\b0x[a-fA-F0-9]{40}\b'),
            entity_type=EntityType.CRYPTO_ADDRESS,
            confidence=0.95,
        ))
        
        # Crypto - Litecoin address (L, M, or 3 prefix, or ltc1 for bech32)
        self.add_rule(Rule(
            name="litecoin_address",
            pattern=re.compile(r'\b[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}\b'),
            entity_type=EntityType.CRYPTO_ADDRESS,
            confidence=0.90,
        ))
        
        # Crypto - Litecoin Bech32 (ltc1)
        self.add_rule(Rule(
            name="litecoin_bech32",
            pattern=re.compile(r'\bltc1[a-zA-HJ-NP-Z0-9]{39,59}\b'),
            entity_type=EntityType.CRYPTO_ADDRESS,
            confidence=0.92,
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
        # FIXED: Require delimiter (: # =) to avoid matching "user agent", "handle with", etc.
        self.add_rule(Rule(
            name="username_labeled",
            pattern=re.compile(r'(?:user(?:name)?|login|screen\s*name)\s*[:=#]\s*[A-Za-z0-9_\-\.]{3,30}', re.I),
            entity_type=EntityType.USERNAME,
            confidence=0.88,
        ))
        # Handle with @ prefix (more specific)
        self.add_rule(Rule(
            name="handle_labeled",
            pattern=re.compile(r'(?:handle)\s*[:=#]\s*@?[A-Za-z0-9_\-\.]{3,30}', re.I),
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
                r'(?:DOB|D\.O\.B\.?|Date\s*of\s*Birth|Birth\s*Date|Born)[:\s]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
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
        
        # ZIP code (US 5+4) - high confidence, always valid
        self.add_rule(Rule(
            name="zip_code_plus4",
            pattern=re.compile(r'\b\d{5}-\d{4}\b'),
            entity_type=EntityType.ZIP,
            confidence=0.98,
            validator=validate_zip,
        ))
        
        # ZIP code (US 5-digit) - with prefix validation
        self.add_rule(Rule(
            name="zip_code_5",
            pattern=re.compile(r'\b\d{5}\b'),
            entity_type=EntityType.ZIP,
            confidence=0.85,
            validator=validate_zip,
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
