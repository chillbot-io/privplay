"""
Calibration module for BERT confidence scores.

Uses Platt scaling (logistic regression) to convert raw model confidences
to calibrated probabilities that reflect actual accuracy.

Usage:
    calibrator = Calibrator.load("calibration_models.json")
    calibrated = calibrator.calibrate_phi(raw_score=0.92)  # → 0.78
"""

import json
import math
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """Sigmoid function with overflow protection."""
    if x < -700:
        return 0.0
    if x > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class Calibrator:
    """
    Calibrates BERT confidence scores using Platt scaling.
    
    Raw BERT scores are often miscalibrated - a model saying 90% confident
    might only be correct 70% of the time. Platt scaling fits a logistic
    regression to convert raw scores to calibrated probabilities.
    
    Formula: calibrated = sigmoid(a * raw + b)
    """
    
    # Type authority mapping
    # PHI-BERT is authoritative for clinical types
    PHI_ONLY_TYPES = {
        "NAME_PATIENT", "NAME_PROVIDER", "MRN", "DATE_DOB", "DATE_ADMISSION",
        "DATE_DISCHARGE", "DIAGNOSIS", "LAB_TEST", "DRUG", "FACILITY",
        "AGE", "MEDICAL_RECORD", "HEALTH_PLAN",
    }
    
    # PII-BERT is authoritative for general PII types
    PII_ONLY_TYPES = {
        "USERNAME", "PASSWORD", "CREDIT_CARD", "BANK_ACCOUNT", "IP_ADDRESS",
        "MAC_ADDRESS", "IMEI", "URL", "USER_AGENT", "CRYPTO_ADDRESS",
        "DEVICE_ID", "IBAN", "SWIFT",
    }
    
    # Shared types - combine both BERT scores
    SHARED_TYPES = {
        "NAME_PERSON", "SSN", "PHONE", "FAX", "EMAIL", "ADDRESS", "ZIP",
        "DATE", "LOCATION", "GPS_COORDINATE", "ACCOUNT_NUMBER",
    }
    
    def __init__(
        self,
        phi_a: float = 1.0,
        phi_b: float = 0.0,
        pii_a: float = 1.0,
        pii_b: float = 0.0,
    ):
        """
        Initialize calibrator with Platt scaling coefficients.
        
        Default values (a=1, b=0) result in identity transform (no calibration).
        """
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.pii_a = pii_a
        self.pii_b = pii_b
        
        self._loaded = False
    
    @classmethod
    def load(cls, path: str = "calibration_models.json") -> "Calibrator":
        """Load calibration coefficients from file."""
        calibrator = cls()
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            if data.get("phi_bert"):
                calibrator.phi_a = data["phi_bert"]["a"]
                calibrator.phi_b = data["phi_bert"]["b"]
                logger.info(f"Loaded PHI-BERT calibration: a={calibrator.phi_a:.4f}, b={calibrator.phi_b:.4f}")
            
            if data.get("pii_bert"):
                calibrator.pii_a = data["pii_bert"]["a"]
                calibrator.pii_b = data["pii_bert"]["b"]
                logger.info(f"Loaded PII-BERT calibration: a={calibrator.pii_a:.4f}, b={calibrator.pii_b:.4f}")
            
            calibrator._loaded = True
            
        except FileNotFoundError:
            logger.warning(f"Calibration file not found: {path}. Using identity transform.")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}. Using identity transform.")
        
        return calibrator
    
    def is_loaded(self) -> bool:
        """Check if calibration was loaded successfully."""
        return self._loaded
    
    def calibrate_phi(self, raw_score: float) -> float:
        """Calibrate PHI-BERT confidence score."""
        return sigmoid(self.phi_a * raw_score + self.phi_b)
    
    def calibrate_pii(self, raw_score: float) -> float:
        """Calibrate PII-BERT confidence score."""
        return sigmoid(self.pii_a * raw_score + self.pii_b)
    
    def get_authority(self, entity_type: str) -> str:
        """
        Determine which BERT model is authoritative for an entity type.
        
        Returns: "phi", "pii", or "shared"
        """
        if entity_type in self.PHI_ONLY_TYPES:
            return "phi"
        elif entity_type in self.PII_ONLY_TYPES:
            return "pii"
        else:
            return "shared"
    
    def combine_scores(
        self,
        phi_raw: float,
        pii_raw: float,
        entity_type: Optional[str] = None,
    ) -> float:
        """
        Combine and calibrate BERT scores based on entity type authority.
        
        Args:
            phi_raw: Raw PHI-BERT confidence (0-1)
            pii_raw: Raw PII-BERT confidence (0-1)
            entity_type: Detected entity type (for routing)
            
        Returns:
            Combined calibrated confidence score
        """
        # Calibrate raw scores
        phi_cal = self.calibrate_phi(phi_raw) if phi_raw > 0 else 0.0
        pii_cal = self.calibrate_pii(pii_raw) if pii_raw > 0 else 0.0
        
        # Route based on entity type authority
        if entity_type:
            authority = self.get_authority(entity_type)
            
            if authority == "phi":
                return phi_cal if phi_cal > 0 else pii_cal
            elif authority == "pii":
                return pii_cal if pii_cal > 0 else phi_cal
        
        # Shared or unknown type: Noisy-OR combination
        # P(PII) = 1 - (1 - P_phi)(1 - P_pii)
        if phi_cal > 0 and pii_cal > 0:
            return 1.0 - (1.0 - phi_cal) * (1.0 - pii_cal)
        elif phi_cal > 0:
            return phi_cal
        elif pii_cal > 0:
            return pii_cal
        else:
            return 0.0
    
    def compute_agreement(self, phi_raw: float, pii_raw: float) -> float:
        """
        Compute BERT agreement score (for meta-classifier features).
        
        High when both models agree, low when they disagree.
        """
        if phi_raw == 0 or pii_raw == 0:
            return 0.0
        return phi_raw * pii_raw
    
    def compute_disagreement(self, phi_raw: float, pii_raw: float) -> float:
        """
        Compute BERT disagreement score (for meta-classifier features).
        
        High when models disagree significantly.
        """
        if phi_raw == 0 and pii_raw == 0:
            return 0.0
        return abs(phi_raw - pii_raw)


# Rule tiers for confidence boosting
class RuleTier:
    """Rule tier classification for confidence boosting."""
    
    NONE = 0       # No rule matched
    WEAK = 1       # Generic pattern match
    STRONG = 2     # Specific format or labeled context
    CHECKSUM = 3   # Algorithmically validated (Luhn, SSN, DEA, NPI)
    
    # Boost factors for each tier
    BOOST = {
        NONE: 0.0,
        WEAK: 0.10,
        STRONG: 0.25,
        CHECKSUM: 0.50,  # Will be overridden to accept directly
    }
    
    @classmethod
    def get_tier(cls, rule_conf: float, has_checksum: bool) -> int:
        """Determine rule tier from confidence and checksum flag."""
        if has_checksum:
            return cls.CHECKSUM
        elif rule_conf >= 0.90:
            return cls.STRONG
        elif rule_conf > 0:
            return cls.WEAK
        else:
            return cls.NONE
    
    @classmethod
    def apply_boost(cls, bert_score: float, tier: int) -> float:
        """
        Apply rule evidence boost to BERT score.
        
        Formula: boosted = bert + (1 - bert) * boost_factor
        This moves the score toward 1.0 proportionally.
        """
        boost = cls.BOOST.get(tier, 0.0)
        return bert_score + (1.0 - bert_score) * boost
