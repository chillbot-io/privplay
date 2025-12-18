"""
Simplified classification engine.

Stack:
- PHI-BERT (clinical NER)
- PII-BERT (general PII NER)
- Checksum rules (algorithmic validation only)
- Meta-classifier (learned decision)

No Presidio, no regex-only patterns, no entropy detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging
import uuid

from ..types import Entity, EntityType, SourceType
from ..config import get_config, Config
from .models.base import BaseModel
from .models.transformer import get_model
from .models.pii_transformer import get_pii_model, PIITransformerModel
from .rules.checksum_rules import ChecksumRuleEngine
from ..allowlist import is_allowed

logger = logging.getLogger(__name__)


# Filter OTHER type - not actionable
FILTER_OTHER = True

# Thresholds
CHECKSUM_CONFIDENCE = 0.99  # Checksum-validated = accept immediately
ACCEPT_THRESHOLD = 0.40     # Model confidence to accept
REJECT_THRESHOLD = 0.25     # Below this = reject


@dataclass
class SpanSignals:
    """Detection signals for a span - features for meta-classifier."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Span info
    span_start: int = 0
    span_end: int = 0
    span_text: str = ""
    
    # PHI-BERT signals
    phi_bert_detected: bool = False
    phi_bert_conf: float = 0.0
    phi_bert_type: str = ""
    
    # PII-BERT signals
    pii_bert_detected: bool = False
    pii_bert_conf: float = 0.0
    pii_bert_type: str = ""
    
    # Checksum rule signals
    checksum_validated: bool = False
    checksum_type: str = ""
    
    # Computed features
    sources_agree_count: int = 0
    bert_agreement: float = 0.0      # phi * pii
    bert_max: float = 0.0            # max(phi, pii)
    type_consensus: bool = False
    span_length: int = 0
    has_digits: bool = False
    has_letters: bool = False
    all_digits: bool = False
    
    # Output
    final_type: str = ""
    final_conf: float = 0.0
    accepted: bool = False
    
    # Ground truth (for training)
    ground_truth_type: Optional[str] = None
    
    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to feature dict for ML model."""
        return {
            "phi_bert_detected": int(self.phi_bert_detected),
            "phi_bert_conf": self.phi_bert_conf,
            "pii_bert_detected": int(self.pii_bert_detected),
            "pii_bert_conf": self.pii_bert_conf,
            "checksum_validated": int(self.checksum_validated),
            "sources_agree_count": self.sources_agree_count,
            "bert_agreement": self.bert_agreement,
            "bert_max": self.bert_max,
            "type_consensus": int(self.type_consensus),
            "span_length": self.span_length,
            "has_digits": int(self.has_digits),
            "has_letters": int(self.has_letters),
            "all_digits": int(self.all_digits),
        }


class ClassificationEngine:
    """
    Simplified PHI/PII classification engine.
    
    Pipeline:
    1. Run PHI-BERT, PII-BERT, Checksum rules in parallel
    2. Checksum matches → accept immediately (algorithmic certainty)
    3. BERT detections → meta-classifier decision
    4. Apply allowlist
    """
    
    def __init__(
        self,
        phi_model: Optional[BaseModel] = None,
        pii_model: Optional[PIITransformerModel] = None,
        config: Optional[Config] = None,
        use_mock_model: bool = False,
        capture_signals: bool = False,
        use_meta_classifier: bool = True,
    ):
        self.config = config or get_config()
        
        # PHI-BERT (Stanford clinical)
        self.phi_model = phi_model or get_model(use_mock=use_mock_model)
        
        # PII-BERT (general PII)
        self.pii_model = pii_model or get_pii_model()
        
        # Checksum rules only
        self.checksum_rules = ChecksumRuleEngine()
        
        # Meta-classifier
        self._meta_classifier = None
        self.use_meta_classifier = use_meta_classifier
        if use_meta_classifier:
            try:
                from ..training.meta_classifier import MetaClassifier
                self._meta_classifier = MetaClassifier()
                if self._meta_classifier.is_trained():
                    logger.info("Meta-classifier loaded")
            except Exception as e:
                logger.warning(f"Meta-classifier not available: {e}")
        
        # Signal capture for training
        self.capture_signals = capture_signals
        self.captured_signals: List[SpanSignals] = []
    
    def detect(
        self, 
        text: str, 
        verify: bool = False,  # Ignored - kept for API compatibility
        threshold: Optional[float] = None,
    ) -> List[Entity]:
        """
        Detect PHI/PII entities.
        
        Returns list of detected entities.
        """
        # Stage 1: Run all detectors
        phi_entities = self._run_phi_bert(text)
        pii_entities = self._run_pii_bert(text)
        checksum_entities = self._run_checksum_rules(text)
        
        # Stage 2: Checksum matches are certain - accept directly
        accepted = []
        for entity in checksum_entities:
            entity._checksum_validated = True
            accepted.append(entity)
            
            if self.capture_signals:
                signals = SpanSignals(
                    span_start=entity.start,
                    span_end=entity.end,
                    span_text=entity.text,
                    checksum_validated=True,
                    checksum_type=entity.entity_type.value,
                    final_type=entity.entity_type.value,
                    final_conf=CHECKSUM_CONFIDENCE,
                    accepted=True,
                )
                self.captured_signals.append(signals)
        
        # Build set of checksum-covered spans
        checksum_spans = set()
        for entity in checksum_entities:
            for pos in range(entity.start, entity.end):
                checksum_spans.add(pos)
        
        # Stage 3: Process BERT detections
        bert_entities = phi_entities + pii_entities
        
        # Group overlapping BERT entities
        groups = self._group_overlapping(bert_entities)
        
        for group in groups:
            # Skip if overlaps with checksum detection
            group_start = min(e.start for e in group)
            group_end = max(e.end for e in group)
            if any(pos in checksum_spans for pos in range(group_start, group_end)):
                continue
            
            # Build signals and decide
            signals = self._build_signals(group, text)
            
            if self.capture_signals:
                self.captured_signals.append(signals)
            
            if signals.accepted:
                entity = Entity(
                    text=signals.span_text,
                    start=signals.span_start,
                    end=signals.span_end,
                    entity_type=EntityType(signals.final_type) if signals.final_type else EntityType.OTHER,
                    confidence=signals.final_conf,
                    source=SourceType.MODEL,
                )
                accepted.append(entity)
        
        # Stage 4: Apply allowlist
        accepted = [e for e in accepted if not is_allowed(e.text)]
        
        # Filter OTHER type
        if FILTER_OTHER:
            accepted = [e for e in accepted if e.entity_type != EntityType.OTHER]
        
        return accepted
    
    def _run_phi_bert(self, text: str) -> List[Entity]:
        """Run PHI-BERT."""
        if not self.phi_model:
            return []
        
        try:
            entities = self.phi_model.detect(text)
            for e in entities:
                e._detector = "phi_bert"
            return entities
        except Exception as e:
            logger.error(f"PHI-BERT failed: {e}")
            return []
    
    def _run_pii_bert(self, text: str) -> List[Entity]:
        """Run PII-BERT."""
        if not self.pii_model:
            return []
        
        try:
            entities = self.pii_model.detect(text)
            for e in entities:
                e._detector = "pii_bert"
            return entities
        except Exception as e:
            logger.error(f"PII-BERT failed: {e}")
            return []
    
    def _run_checksum_rules(self, text: str) -> List[Entity]:
        """Run checksum-validated rules."""
        try:
            entities = self.checksum_rules.detect(text)
            for e in entities:
                e._detector = "checksum_rule"
            return entities
        except Exception as e:
            logger.error(f"Checksum rules failed: {e}")
            return []
    
    def _group_overlapping(self, entities: List[Entity]) -> List[List[Entity]]:
        """Group overlapping entities."""
        if not entities:
            return []
        
        entities = sorted(entities, key=lambda e: (e.start, -e.end))
        
        groups = []
        current_group = [entities[0]]
        current_end = entities[0].end
        
        for entity in entities[1:]:
            if entity.start < current_end:
                # Overlaps
                current_group.append(entity)
                current_end = max(current_end, entity.end)
            else:
                # New group
                groups.append(current_group)
                current_group = [entity]
                current_end = entity.end
        
        groups.append(current_group)
        return groups
    
    def _build_signals(self, group: List[Entity], text: str) -> SpanSignals:
        """Build signals from entity group and make decision."""
        # Find best span
        best = max(group, key=lambda e: e.confidence)
        
        signals = SpanSignals(
            span_start=best.start,
            span_end=best.end,
            span_text=text[best.start:best.end],
        )
        
        # Extract detector signals
        types_seen = set()
        
        for e in group:
            detector = getattr(e, "_detector", "")
            etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
            types_seen.add(etype)
            
            if detector == "phi_bert":
                signals.phi_bert_detected = True
                signals.phi_bert_conf = max(signals.phi_bert_conf, e.confidence)
                signals.phi_bert_type = etype
            elif detector == "pii_bert":
                signals.pii_bert_detected = True
                signals.pii_bert_conf = max(signals.pii_bert_conf, e.confidence)
                signals.pii_bert_type = etype
        
        # Compute features
        signals.sources_agree_count = sum([
            signals.phi_bert_detected,
            signals.pii_bert_detected,
        ])
        signals.bert_agreement = signals.phi_bert_conf * signals.pii_bert_conf
        signals.bert_max = max(signals.phi_bert_conf, signals.pii_bert_conf)
        signals.type_consensus = len(types_seen) == 1
        signals.span_length = len(signals.span_text)
        signals.has_digits = any(c.isdigit() for c in signals.span_text)
        signals.has_letters = any(c.isalpha() for c in signals.span_text)
        signals.all_digits = signals.span_text.replace('-', '').replace(' ', '').isdigit()
        
        # Make decision
        signals.accepted, signals.final_conf = self._decide(signals)
        
        # Determine type
        if signals.accepted:
            signals.final_type = self._resolve_type(signals)
        
        return signals
    
    def _decide(self, signals: SpanSignals) -> Tuple[bool, float]:
        """Decide whether to accept this detection."""
        
        # Try meta-classifier first
        if self._meta_classifier and self._meta_classifier.is_trained():
            try:
                is_entity, _, confidence = self._meta_classifier.predict(signals)
                return is_entity, confidence
            except Exception as e:
                logger.warning(f"Meta-classifier failed: {e}")
        
        # Fallback: threshold on max BERT confidence
        conf = signals.bert_max
        
        # Boost if both models agree
        if signals.sources_agree_count == 2:
            conf = min(0.99, conf + 0.1)
        
        if conf >= ACCEPT_THRESHOLD:
            return True, conf
        elif conf < REJECT_THRESHOLD:
            return False, conf
        else:
            # Uncertain - accept with lower confidence
            return True, conf
    
    def _resolve_type(self, signals: SpanSignals) -> str:
        """Resolve final entity type."""
        # If both agree, use that
        if signals.phi_bert_type and signals.pii_bert_type:
            if signals.phi_bert_type == signals.pii_bert_type:
                return signals.phi_bert_type
        
        # Use higher confidence model's type
        if signals.phi_bert_conf > signals.pii_bert_conf:
            return signals.phi_bert_type or signals.pii_bert_type
        else:
            return signals.pii_bert_type or signals.phi_bert_type
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_captured_signals(self) -> List[SpanSignals]:
        """Get captured signals for training."""
        return self.captured_signals
    
    def clear_captured_signals(self):
        """Clear captured signals."""
        self.captured_signals = []
    
    def get_stack_status(self) -> Dict[str, Any]:
        """Get status of detection components."""
        return {
            "phi_bert": {
                "available": self.phi_model.is_available() if self.phi_model else False,
            },
            "pii_bert": {
                "available": self.pii_model.is_available() if self.pii_model else False,
            },
            "checksum_rules": {
                "count": len(self.checksum_rules.rules),
            },
            "meta_classifier": {
                "trained": self._meta_classifier.is_trained() if self._meta_classifier else False,
            },
        }
