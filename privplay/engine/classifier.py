"""Main classification engine - combines models, rules, and verification."""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
import logging
import uuid

from ..types import Entity, EntityType, SourceType, VerificationResult
from ..config import get_config, Config
from .models.base import BaseModel
from .models.transformer import get_model
from .models.pii_transformer import get_pii_model, PIITransformerModel
from .models.presidio_detector import PresidioDetector, get_presidio_detector
from .rules.engine import RuleEngine
from ..verification.verifier import Verifier, get_verifier
from ..allowlist import is_allowed

logger = logging.getLogger(__name__)


# Filter all OTHER type entities - not actionable for redaction
FILTER_ALL_OTHER = True


@dataclass
class SpanSignals:
    """
    All detection signals for a single span.
    
    This is the feature vector for the meta-classifier.
    Each span gets one of these, capturing what every detector said.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Span info
    span_start: int = 0
    span_end: int = 0
    span_text: str = ""
    
    # Stanford PHI BERT signals
    phi_bert_detected: bool = False
    phi_bert_conf: float = 0.0
    phi_bert_type: str = ""
    
    # PII BERT signals (AI4Privacy trained)
    pii_bert_detected: bool = False
    pii_bert_conf: float = 0.0
    pii_bert_type: str = ""
    
    # Presidio signals
    presidio_detected: bool = False
    presidio_conf: float = 0.0
    presidio_type: str = ""
    
    # Rules signals
    rule_detected: bool = False
    rule_conf: float = 0.0
    rule_type: str = ""
    rule_has_checksum: bool = False
    
    # Ollama/LLM signals
    llm_verified: bool = False
    llm_decision: str = ""  # yes/no/uncertain
    llm_conf: float = 0.0
    
    # Computed features
    sources_agree_count: int = 0
    span_length: int = 0
    has_digits: bool = False
    has_letters: bool = False
    all_caps: bool = False
    all_digits: bool = False
    mixed_case: bool = False
    
    # Current classifier output (rule-based merge)
    merged_type: str = ""
    merged_conf: float = 0.0
    merged_source: str = ""
    
    # Ground truth (filled by human review or benchmark)
    ground_truth_type: Optional[str] = None  # None=unknown, "NONE"=not PHI
    ground_truth_source: Optional[str] = None  # "benchmark", "human"
    
    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to feature dict for ML model."""
        return {
            "phi_bert_detected": int(self.phi_bert_detected),
            "phi_bert_conf": self.phi_bert_conf,
            "pii_bert_detected": int(self.pii_bert_detected),
            "pii_bert_conf": self.pii_bert_conf,
            "presidio_detected": int(self.presidio_detected),
            "presidio_conf": self.presidio_conf,
            "rule_detected": int(self.rule_detected),
            "rule_conf": self.rule_conf,
            "rule_has_checksum": int(self.rule_has_checksum),
            "llm_verified": int(self.llm_verified),
            "llm_conf": self.llm_conf,
            "sources_agree_count": self.sources_agree_count,
            "span_length": self.span_length,
            "has_digits": int(self.has_digits),
            "has_letters": int(self.has_letters),
            "all_caps": int(self.all_caps),
            "all_digits": int(self.all_digits),
            "mixed_case": int(self.mixed_case),
        }


class ClassificationEngine:
    """
    Main PHI/PII classification engine.
    
    Defense-in-depth approach combining:
    - Stanford transformer (PHI-specific, clinical)
    - PII BERT (trained on AI4Privacy, general PII)
    - Presidio (Microsoft PII detector)
    - Rule-based detection (regex + validation)
    - LLM verification (for uncertain cases)
    
    Signal capture mode stores all detector outputs for meta-classifier training.
    """
    
    def __init__(
        self,
        phi_model: Optional[BaseModel] = None,
        pii_model: Optional[PIITransformerModel] = None,
        presidio: Optional[PresidioDetector] = None,
        rules: Optional[RuleEngine] = None,
        verifier: Optional[Verifier] = None,
        config: Optional[Config] = None,
        use_mock_model: bool = False,
        capture_signals: bool = False,
    ):
        self.config = config or get_config()
        
        # Stanford PHI model
        self.phi_model = phi_model or get_model(use_mock=use_mock_model)
        
        # PII BERT model (may not be available)
        self.pii_model = pii_model
        if self.pii_model is None:
            self.pii_model = get_pii_model()
        
        # Rules engine
        self.rules = rules or RuleEngine()
        
        # LLM verifier
        self.verifier = verifier or get_verifier()
        
        # Initialize Presidio if enabled
        self.presidio = presidio
        if self.presidio is None and self.config.presidio.enabled:
            self.presidio = get_presidio_detector(
                score_threshold=self.config.presidio.score_threshold
            )
        
        # Signal capture for meta-classifier training
        self.capture_signals = capture_signals
        self.captured_signals: List[SpanSignals] = []
    
    def detect(
        self, 
        text: str, 
        verify: bool = True,
        threshold: Optional[float] = None,
    ) -> List[Entity]:
        """
        Detect PHI/PII entities in text.
        
        Args:
            text: Input text to scan
            verify: Whether to run LLM verification
            threshold: Confidence threshold for verification
            
        Returns:
            List of detected entities
        """
        if threshold is None:
            threshold = self.config.confidence_threshold
        
        # Run all detectors
        phi_entities = self._run_phi_model(text)
        pii_entities = self._run_pii_model(text)
        presidio_entities = self._run_presidio(text)
        rule_entities = self._run_rules(text)
        
        # Merge all detections (captures signals if enabled)
        entities = self._merge_entities(
            text, phi_entities, pii_entities, presidio_entities, rule_entities
        )
        
        # Apply allowlist
        entities = self._apply_allowlist(entities)
        
        # Run LLM verification on uncertain entities
        if verify and self.verifier and self.verifier.is_available():
            entities = self._verify_uncertain(entities, text, threshold)
        
        return entities
    
    def _run_phi_model(self, text: str) -> List[Entity]:
        """Run Stanford PHI model."""
        if not self.phi_model:
            return []
        
        try:
            entities = self.phi_model.detect(text)
            # Filter OTHER if configured
            if FILTER_ALL_OTHER:
                entities = [e for e in entities if e.entity_type != EntityType.OTHER]
            return entities
        except Exception as e:
            logger.error(f"PHI model failed: {e}")
            return []
    
    def _run_pii_model(self, text: str) -> List[Entity]:
        """Run PII BERT model."""
        if not self.pii_model or not self.pii_model.is_available():
            return []
        
        try:
            entities = self.pii_model.detect(text)
            # Mark source as PII_MODEL to distinguish from PHI model
            for e in entities:
                e.source = SourceType.MODEL  # Could add PII_MODEL to SourceType
            return entities
        except Exception as e:
            logger.error(f"PII model failed: {e}")
            return []
    
    def _run_presidio(self, text: str) -> List[Entity]:
        """Run Presidio PII detection."""
        if not self.presidio:
            return []
        
        try:
            entities = self.presidio.detect(text)
            if FILTER_ALL_OTHER:
                entities = [e for e in entities if e.entity_type != EntityType.OTHER]
            return entities
        except Exception as e:
            logger.error(f"Presidio failed: {e}")
            return []
    
    def _run_rules(self, text: str) -> List[Entity]:
        """Run rule-based detection."""
        try:
            return self.rules.detect(text)
        except Exception as e:
            logger.error(f"Rules failed: {e}")
            return []
    
    def _merge_entities(
        self,
        text: str,
        phi_entities: List[Entity],
        pii_entities: List[Entity],
        presidio_entities: List[Entity],
        rule_entities: List[Entity],
    ) -> List[Entity]:
        """
        Merge entities from all sources using tiered authority model.
        
        If capture_signals is enabled, stores all signals for each span.
        """
        # Collect all entities with source tags
        all_entities = []
        
        for e in phi_entities:
            e._detector = "phi_bert"
            all_entities.append(e)
        
        for e in pii_entities:
            e._detector = "pii_bert"
            all_entities.append(e)
        
        for e in presidio_entities:
            e._detector = "presidio"
            all_entities.append(e)
        
        for e in rule_entities:
            e._detector = "rule"
            all_entities.append(e)
        
        if not all_entities:
            return []
        
        # Sort by start position
        all_entities.sort(key=lambda e: (e.start, -e.confidence))
        
        merged = []
        i = 0
        
        while i < len(all_entities):
            current = all_entities[i]
            
            # Find overlapping entities
            overlapping = [current]
            j = i + 1
            
            while j < len(all_entities):
                next_entity = all_entities[j]
                if next_entity.start < current.end:
                    overlapping.append(next_entity)
                    # Extend current span if next entity extends beyond
                    if next_entity.end > current.end:
                        current = Entity(
                            text=text[current.start:next_entity.end],
                            start=current.start,
                            end=next_entity.end,
                            entity_type=current.entity_type,
                            confidence=current.confidence,
                            source=current.source,
                        )
                        current._detector = overlapping[0]._detector
                    j += 1
                else:
                    break
            
            # Merge overlapping entities
            result = self._merge_overlapping(overlapping, text)
            merged.append(result)
            
            # Capture signals if enabled
            if self.capture_signals:
                signals = self._build_signals(overlapping, result, text)
                self.captured_signals.append(signals)
            
            i = j
        
        return merged
    
    def _build_signals(
        self, 
        overlapping: List[Entity], 
        result: Entity,
        text: str,
    ) -> SpanSignals:
        """Build signal vector from overlapping entities."""
        signals = SpanSignals(
            span_start=result.start,
            span_end=result.end,
            span_text=result.text,
        )
        
        # Extract signals from each detector
        for e in overlapping:
            detector = getattr(e, "_detector", "unknown")
            etype = e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)
            
            if detector == "phi_bert":
                signals.phi_bert_detected = True
                signals.phi_bert_conf = max(signals.phi_bert_conf, e.confidence)
                signals.phi_bert_type = etype
            elif detector == "pii_bert":
                signals.pii_bert_detected = True
                signals.pii_bert_conf = max(signals.pii_bert_conf, e.confidence)
                signals.pii_bert_type = etype
            elif detector == "presidio":
                signals.presidio_detected = True
                signals.presidio_conf = max(signals.presidio_conf, e.confidence)
                signals.presidio_type = etype
            elif detector == "rule":
                signals.rule_detected = True
                signals.rule_conf = max(signals.rule_conf, e.confidence)
                signals.rule_type = etype
                # High confidence rules typically have checksum validation
                signals.rule_has_checksum = e.confidence >= 0.95
        
        # Computed features
        detectors = set(getattr(e, "_detector", "") for e in overlapping)
        signals.sources_agree_count = len(detectors)
        signals.span_length = len(result.text)
        signals.has_digits = any(c.isdigit() for c in result.text)
        signals.has_letters = any(c.isalpha() for c in result.text)
        signals.all_caps = result.text.isupper() and signals.has_letters
        signals.all_digits = result.text.replace("-", "").replace(" ", "").isdigit()
        signals.mixed_case = (
            any(c.isupper() for c in result.text) and 
            any(c.islower() for c in result.text)
        )
        
        # Current merge output
        signals.merged_type = result.entity_type.value if hasattr(result.entity_type, "value") else str(result.entity_type)
        signals.merged_conf = result.confidence
        signals.merged_source = result.source.value if hasattr(result.source, "value") else str(result.source)
        
        return signals
    
    def _merge_overlapping(self, entities: List[Entity], text: str) -> Entity:
        """
        Merge overlapping entities using tiered authority model.
        
        Tier 1: Rules (algorithmic validation) → highest authority
        Tier 2: Presidio (pattern + context)
        Tier 3: PII BERT (learned on AI4Privacy)
        Tier 4: PHI BERT (Stanford, clinical focus)
        """
        sources = set(getattr(e, "_detector", e.source) for e in entities)
        n_sources = len(sources)
        
        # Tier 1: Rules with high confidence always win
        rules = [e for e in entities if getattr(e, "_detector", "") == "rule"]
        if rules:
            best = max(rules, key=lambda e: e.confidence)
            confidence = best.confidence
            if n_sources > 1:
                confidence = min(0.99, confidence + 0.05)
            return Entity(
                text=best.text,
                start=best.start,
                end=best.end,
                entity_type=best.entity_type,
                confidence=confidence,
                source=SourceType.MERGED if n_sources > 1 else SourceType.RULE,
            )
        
        # Tier 2: Presidio
        presidio = [e for e in entities if getattr(e, "_detector", "") == "presidio"]
        if presidio:
            best = max(presidio, key=lambda e: e.confidence)
            confidence = best.confidence
            if n_sources > 1:
                confidence = min(0.99, confidence + 0.05)
            return Entity(
                text=best.text,
                start=best.start,
                end=best.end,
                entity_type=best.entity_type,
                confidence=confidence,
                source=SourceType.MERGED if n_sources > 1 else SourceType.PRESIDIO,
            )
        
        # Tier 3: PII BERT (more entity types than PHI BERT)
        pii = [e for e in entities if getattr(e, "_detector", "") == "pii_bert"]
        if pii:
            specific = [e for e in pii if e.entity_type != EntityType.OTHER]
            best = max(specific or pii, key=lambda e: e.confidence)
            confidence = best.confidence
            if n_sources > 1:
                confidence = min(0.99, confidence + 0.05)
            return Entity(
                text=best.text,
                start=best.start,
                end=best.end,
                entity_type=best.entity_type,
                confidence=confidence,
                source=SourceType.MERGED if n_sources > 1 else SourceType.MODEL,
            )
        
        # Tier 4: PHI BERT (Stanford)
        phi = [e for e in entities if getattr(e, "_detector", "") == "phi_bert"]
        if phi:
            specific = [e for e in phi if e.entity_type != EntityType.OTHER]
            best = max(specific or phi, key=lambda e: e.confidence)
            return Entity(
                text=best.text,
                start=best.start,
                end=best.end,
                entity_type=best.entity_type,
                confidence=best.confidence,
                source=SourceType.MODEL,
            )
        
        # Fallback - shouldn't happen
        best = max(entities, key=lambda e: e.confidence)
        return best
    
    def _apply_allowlist(self, entities: List[Entity]) -> List[Entity]:
        """Filter out allowlisted terms."""
        return [e for e in entities if not is_allowed(e.text)]
    
    def _verify_uncertain(
        self, 
        entities: List[Entity], 
        text: str, 
        threshold: float,
    ) -> List[Entity]:
        """Run LLM verification on uncertain entities."""
        verified = []
        
        for entity in entities:
            if entity.confidence >= threshold:
                verified.append(entity)
            else:
                result = self.verifier.verify(entity, text)
                if result.decision == VerificationResult.YES:
                    entity.llm_confidence = result.confidence
                    entity.llm_reasoning = result.reasoning
                    verified.append(entity)
                elif result.decision == VerificationResult.UNCERTAIN:
                    entity.llm_confidence = result.confidence
                    entity.llm_reasoning = result.reasoning
                    verified.append(entity)
                # If NO, entity is dropped
        
        return verified
    
    def get_captured_signals(self) -> List[SpanSignals]:
        """Get captured signals for meta-classifier training."""
        return self.captured_signals
    
    def clear_captured_signals(self):
        """Clear captured signals."""
        self.captured_signals = []
    
    def get_stack_status(self) -> Dict[str, Any]:
        """Get status of all detection components."""
        return {
            "phi_bert": {
                "name": self.phi_model.name if self.phi_model else None,
                "available": self.phi_model.is_available() if self.phi_model else False,
            },
            "pii_bert": {
                "name": self.pii_model.name if self.pii_model else None,
                "available": self.pii_model.is_available() if self.pii_model else False,
            },
            "presidio": {
                "enabled": self.presidio is not None,
                "available": self.presidio.is_available() if self.presidio else False,
            },
            "rules": {
                "enabled": True,
                "rule_count": len(self.rules.rules) if self.rules else 0,
            },
            "verifier": {
                "provider": getattr(self.verifier, "provider", "unknown") if self.verifier else None,
                "available": self.verifier.is_available() if self.verifier else False,
            },
        }
