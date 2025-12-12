"""Main classification engine - combines models, rules, and verification."""

from typing import List, Optional, Set, Tuple
import logging

from ..types import Entity, EntityType, SourceType, VerificationResult
from ..config import get_config, Config
from .models.base import BaseModel
from .models.transformer import get_model
from .models.presidio_detector import PresidioDetector, get_presidio_detector
from .rules.engine import RuleEngine
from ..verification.verifier import Verifier, get_verifier

logger = logging.getLogger(__name__)


class ClassificationEngine:
    """
    Main PHI/PII classification engine.
    
    Defense-in-depth approach combining:
    - Transformer model (Stanford de-identifier) → PHI-specific
    - Presidio (Microsoft PII detector) → General PII (backs up PHI since most PHI contains PII)
    - Rule-based detection (regex patterns) → Known formats
    - LLM verification (for uncertain cases) → Human-like judgment
    """
    
    def __init__(
        self,
        model: Optional[BaseModel] = None,
        presidio: Optional[PresidioDetector] = None,
        rules: Optional[RuleEngine] = None,
        verifier: Optional[Verifier] = None,
        config: Optional[Config] = None,
        use_mock_model: bool = False,
    ):
        self.config = config or get_config()
        self.model = model or get_model(use_mock=use_mock_model)
        self.rules = rules or RuleEngine()
        self.verifier = verifier or get_verifier()
        
        # Initialize Presidio if enabled
        self.presidio = presidio
        if self.presidio is None and self.config.presidio.enabled:
            self.presidio = get_presidio_detector(
                score_threshold=self.config.presidio.score_threshold
            )
        
        # Allowlist - known non-PHI terms
        self.allowlist: Set[str] = set()
        
        # Blocklist - known PHI terms (force detection)
        self.blocklist: Set[str] = set()
    
    def detect(
        self, 
        text: str, 
        verify: bool = True,
        threshold: Optional[float] = None
    ) -> List[Entity]:
        """
        Detect PHI/PII entities in text.
        
        Defense-in-depth pipeline:
        1. Run transformer model (PHI-specific)
        2. Run Presidio (general PII - backs up PHI since most PHI has PII)
        3. Run rule-based detection (known formats)
        4. Merge all detections (boost confidence when sources agree)
        5. Apply allowlist/blocklist
        6. Run LLM verification on uncertain entities
        
        Args:
            text: Input text to scan
            verify: Whether to run LLM verification on uncertain entities
            threshold: Confidence threshold for verification (default from config)
            
        Returns:
            List of detected entities
        """
        if threshold is None:
            threshold = self.config.confidence_threshold
        
        # Run transformer model (PHI-specific)
        model_entities = self._run_model(text)
        
        # Run Presidio PII detection (defense in depth)
        presidio_entities = self._run_presidio(text)
        
        # Run rule-based detection
        rule_entities = self._run_rules(text)
        
        # Merge all detections
        entities = self._merge_entities(model_entities, presidio_entities, rule_entities)
        
        # Apply allowlist/blocklist
        entities = self._apply_lists(entities)
        
        # Run LLM verification on uncertain entities
        if verify and self.verifier.is_available():
            entities = self._verify_uncertain(entities, text, threshold)
        
        return entities
    
    def _run_model(self, text: str) -> List[Entity]:
        """Run transformer model detection (PHI-specific)."""
        try:
            return self.model.detect(text)
        except Exception as e:
            logger.error(f"Model detection failed: {e}")
            return []
    
    def _run_presidio(self, text: str) -> List[Entity]:
        """Run Presidio PII detection (defense in depth for general PII)."""
        if self.presidio is None:
            return []
        
        try:
            entities = self.presidio.detect(text)
            # Tag source as PRESIDIO
            for entity in entities:
                entity.source = SourceType.PRESIDIO
            return entities
        except Exception as e:
            logger.error(f"Presidio detection failed: {e}")
            return []
    
    def _run_rules(self, text: str) -> List[Entity]:
        """Run rule-based detection."""
        try:
            return self.rules.detect(text)
        except Exception as e:
            logger.error(f"Rule detection failed: {e}")
            return []
    
    def _merge_entities(
        self, 
        model_entities: List[Entity], 
        presidio_entities: List[Entity],
        rule_entities: List[Entity]
    ) -> List[Entity]:
        """
        Merge entities from different sources.
        
        Strategy:
        - Overlapping entities: keep higher confidence
        - Boost confidence when multiple sources agree
        - More sources agreeing = higher confidence boost
        """
        all_entities = model_entities + presidio_entities + rule_entities
        
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
                
                # Check for overlap
                if next_entity.start < current.end:
                    overlapping.append(next_entity)
                    j += 1
                else:
                    break
            
            # Merge overlapping entities
            if len(overlapping) == 1:
                merged.append(current)
            else:
                merged.append(self._merge_overlapping(overlapping))
            
            i = j
        
        return merged
    
    def _merge_overlapping(self, entities: List[Entity]) -> Entity:
        """
        Merge overlapping entities into one.
        
        Confidence boost strategy:
        - 2 sources agree on same type: +5% confidence
        - 3+ sources agree on same type: +10% confidence
        """
        sources = set(e.source for e in entities)
        types = set(e.entity_type for e in entities)
        
        # Use highest confidence as base
        best = max(entities, key=lambda e: e.confidence)
        
        # Boost if multiple sources agree
        confidence = best.confidence
        n_sources = len(sources)
        
        if n_sources > 1 and len(types) == 1:
            # Multiple sources agree on same type - boost confidence
            if n_sources >= 3:
                confidence = min(0.99, confidence + 0.10)
            else:
                confidence = min(0.99, confidence + 0.05)
        
        return Entity(
            text=best.text,
            start=best.start,
            end=best.end,
            entity_type=best.entity_type,
            confidence=confidence,
            source=SourceType.MERGED if n_sources > 1 else best.source,
        )
    
    def _apply_lists(self, entities: List[Entity]) -> List[Entity]:
        """Apply allowlist and blocklist."""
        result = []
        
        for entity in entities:
            text_lower = entity.text.lower()
            
            # Skip if in allowlist
            if text_lower in self.allowlist:
                continue
            
            # Boost if in blocklist
            if text_lower in self.blocklist:
                entity.confidence = 0.99
            
            result.append(entity)
        
        return result
    
    def _verify_uncertain(
        self, 
        entities: List[Entity], 
        text: str,
        threshold: float
    ) -> List[Entity]:
        """Run LLM verification on entities below threshold."""
        context_window = self.config.context_window
        
        for entity in entities:
            if entity.confidence >= threshold:
                continue
            
            # Get context
            start = max(0, entity.start - context_window)
            end = min(len(text), entity.end + context_window)
            context = text[start:end]
            
            # Verify with LLM
            response = self.verifier.verify(
                entity_text=entity.text,
                entity_type=entity.entity_type,
                context=context
            )
            
            # Update entity with LLM results
            entity.llm_reasoning = response.reasoning
            
            if response.decision == VerificationResult.YES:
                # LLM confirms PHI - boost confidence
                entity.llm_confidence = response.confidence
            elif response.decision == VerificationResult.NO:
                # LLM says not PHI - lower confidence
                entity.llm_confidence = 1.0 - response.confidence
            else:
                # Uncertain
                entity.llm_confidence = 0.5
        
        return entities
    
    def add_to_allowlist(self, term: str):
        """Add term to allowlist (not PHI)."""
        self.allowlist.add(term.lower())
    
    def add_to_blocklist(self, term: str):
        """Add term to blocklist (always PHI)."""
        self.blocklist.add(term.lower())
    
    def get_context(self, text: str, start: int, end: int) -> Tuple[str, str]:
        """Get context before and after an entity."""
        window = self.config.context_window
        
        context_before = text[max(0, start - window):start]
        context_after = text[end:min(len(text), end + window)]
        
        return context_before, context_after
    
    def get_stack_status(self) -> dict:
        """
        Get status of all detection components in the stack.
        
        Useful for diagnostics and understanding which layers are active.
        """
        status = {
            "transformer": {
                "name": self.model.name if hasattr(self.model, 'name') else "unknown",
                "available": self.model.is_available() if hasattr(self.model, 'is_available') else True,
            },
            "presidio": {
                "enabled": self.presidio is not None,
                "available": self.presidio.is_available() if self.presidio else False,
                "error": self.presidio.load_error if self.presidio and hasattr(self.presidio, 'load_error') else None,
            },
            "rules": {
                "enabled": self.rules is not None,
                "pattern_count": len(self.rules.patterns) if hasattr(self.rules, 'patterns') else 0,
            },
            "verifier": {
                "provider": self.config.verification.provider,
                "available": self.verifier.is_available() if self.verifier else False,
            },
        }
        return status
