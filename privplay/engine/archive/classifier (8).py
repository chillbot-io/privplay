"""
Main classification engine - staged pipeline architecture.

Stage 1: Parallel Detection (PHI-BERT, PII-BERT, Presidio, Rules)
Stage 2: Checksum Fast-Path (algorithmic validation bypasses ML)
Stage 3: BERT Score Calibration + Routing (Platt scaling, type authority)
Stage 4: Rule Evidence Boost (patterns increase confidence)
Stage 5: Meta-Classifier Decision (XGBoost accept/reject)
Stage 6: Type Assignment (consensus or authority-based)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any, Tuple
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

# Decision thresholds
ACCEPT_THRESHOLD = 0.35      # Above this → accept
REJECT_THRESHOLD = 0.20      # Below this → reject
# Between REJECT and ACCEPT → uncertain (could go to human review)


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
    
    # Stanford PHI BERT signals (raw)
    phi_bert_detected: bool = False
    phi_bert_conf: float = 0.0
    phi_bert_type: str = ""
    
    # PII BERT signals (raw)
    pii_bert_detected: bool = False
    pii_bert_conf: float = 0.0
    pii_bert_type: str = ""
    
    # Calibrated BERT signals (Stage 3 output)
    bert_calibrated: float = 0.0
    bert_authority: str = ""  # "phi", "pii", or "shared"
    
    # Presidio signals
    presidio_detected: bool = False
    presidio_conf: float = 0.0
    presidio_type: str = ""
    
    # Rules signals
    rule_detected: bool = False
    rule_conf: float = 0.0
    rule_type: str = ""
    rule_has_checksum: bool = False
    rule_tier: int = 0  # 0=none, 1=weak, 2=strong, 3=checksum
    
    # Stage 4 output (after rule boost)
    boosted_conf: float = 0.0
    
    # Ollama/LLM signals
    llm_verified: bool = False
    llm_decision: str = ""
    llm_conf: float = 0.0
    
    # Coreference signals
    in_coref_cluster: bool = False
    coref_cluster_id: Optional[int] = None
    coref_cluster_size: int = 0
    coref_anchor_type: Optional[str] = None
    coref_anchor_conf: float = 0.0
    coref_is_anchor: bool = False
    coref_is_pronoun: bool = False
    
    # Document-level features
    doc_type: str = ""
    doc_has_ssn: bool = False
    doc_has_dates: bool = False
    doc_has_medical_terms: bool = False
    doc_entity_count: int = 0
    doc_pii_density: float = 0.0
    
    # Computed features
    sources_agree_count: int = 0
    span_length: int = 0
    has_digits: bool = False
    has_letters: bool = False
    all_caps: bool = False
    all_digits: bool = False
    mixed_case: bool = False
    
    # BERT agreement features (for meta-classifier)
    bert_agreement: float = 0.0      # phi * pii (both agree)
    bert_disagreement: float = 0.0   # |phi - pii| (models disagree)
    bert_max: float = 0.0            # max(phi, pii)
    type_consensus: bool = False     # Do all sources agree on type?
    
    # Current classifier output
    merged_type: str = ""
    merged_conf: float = 0.0
    merged_source: str = ""
    
    # Meta-classifier output
    meta_decision: Optional[bool] = None  # True=accept, False=reject, None=uncertain
    meta_confidence: float = 0.0
    
    # Ground truth (for training)
    ground_truth_type: Optional[str] = None
    ground_truth_source: Optional[str] = None
    
    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to feature dict for ML model."""
        return {
            # Raw detector signals
            "phi_bert_detected": int(self.phi_bert_detected),
            "phi_bert_conf": self.phi_bert_conf,
            "pii_bert_detected": int(self.pii_bert_detected),
            "pii_bert_conf": self.pii_bert_conf,
            "presidio_detected": int(self.presidio_detected),
            "presidio_conf": self.presidio_conf,
            "rule_detected": int(self.rule_detected),
            "rule_conf": self.rule_conf,
            "rule_has_checksum": int(self.rule_has_checksum),
            "rule_tier": self.rule_tier,
            
            # Calibrated/boosted signals (NEW)
            "bert_calibrated": self.bert_calibrated,
            "boosted_conf": self.boosted_conf,
            
            # BERT agreement features (NEW)
            "bert_agreement": self.bert_agreement,
            "bert_disagreement": self.bert_disagreement,
            "bert_max": self.bert_max,
            "type_consensus": int(self.type_consensus),
            
            # LLM signals
            "llm_verified": int(self.llm_verified),
            "llm_conf": self.llm_conf,
            
            # Coreference signals
            "in_coref_cluster": int(self.in_coref_cluster),
            "coref_cluster_size": self.coref_cluster_size,
            "coref_anchor_conf": self.coref_anchor_conf,
            "coref_is_anchor": int(self.coref_is_anchor),
            "coref_is_pronoun": int(self.coref_is_pronoun),
            "coref_has_phi_anchor": int(self.coref_anchor_type is not None),
            
            # Document-level features
            "doc_has_ssn": int(self.doc_has_ssn),
            "doc_has_dates": int(self.doc_has_dates),
            "doc_has_medical_terms": int(self.doc_has_medical_terms),
            "doc_entity_count": self.doc_entity_count,
            "doc_pii_density": self.doc_pii_density,
            
            # Computed features
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
    Main PHI/PII classification engine with staged pipeline.
    
    Architecture:
    - Stage 1: Parallel detection (PHI-BERT, PII-BERT, Presidio, Rules)
    - Stage 2: Checksum fast-path (bypass ML for algorithmic validation)
    - Stage 3: BERT calibration + routing (Platt scaling, type authority)
    - Stage 4: Rule evidence boost (patterns increase confidence)
    - Stage 5: Meta-classifier decision (XGBoost accept/reject)
    - Stage 6: Type assignment (consensus or authority-based)
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
        use_coreference: bool = True,
        use_meta_classifier: bool = True,
        calibration_path: Optional[str] = None,
    ):
        self.config = config or get_config()
        
        # Load calibrator
        self._calibrator = None
        self._load_calibrator(calibration_path)
        
        # Meta-classifier for learned merge decisions
        self._meta_classifier = None
        self.use_meta_classifier = use_meta_classifier
        if use_meta_classifier:
            try:
                from ..training.meta_classifier import MetaClassifier
                self._meta_classifier = MetaClassifier()
                if self._meta_classifier.is_trained():
                    logger.info("Meta-classifier loaded and ready")
                else:
                    logger.info("Meta-classifier not trained, using rule-based merge")
            except Exception as e:
                logger.warning(f"Failed to load meta-classifier: {e}")
        
        # Stanford PHI model
        self.phi_model = phi_model or get_model(use_mock=use_mock_model)
        
        # PII BERT model
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
        
        # Coreference resolver (lazy loaded)
        self.use_coreference = use_coreference
        self._coref_resolver = None
        
        # Signal capture for training
        self.capture_signals = capture_signals
        self.captured_signals: List[SpanSignals] = []
    
    def _load_calibrator(self, path: Optional[str] = None):
        """Load calibration model for BERT scores."""
        try:
            from .calibration import Calibrator
            
            if path:
                self._calibrator = Calibrator.load(path)
            else:
                # Try default locations
                from pathlib import Path
                default_paths = [
                    Path.cwd() / "calibration_models.json",
                    Path.home() / ".privplay" / "calibration_models.json",
                ]
                for p in default_paths:
                    if p.exists():
                        self._calibrator = Calibrator.load(str(p))
                        break
                
                if self._calibrator is None:
                    self._calibrator = Calibrator()  # Identity transform
                    logger.info("No calibration model found, using raw BERT scores")
        except ImportError:
            logger.warning("Calibration module not available")
            self._calibrator = None
    
    def _get_coref_resolver(self):
        """Lazy load coreference resolver."""
        if self._coref_resolver is None and self.use_coreference:
            try:
                from .coreference import CoreferenceResolver
                self._coref_resolver = CoreferenceResolver(device='cpu')
            except Exception as e:
                logger.warning(f"Failed to load coreference resolver: {e}")
                self.use_coreference = False
        return self._coref_resolver
    
    def detect(
        self, 
        text: str, 
        verify: bool = True,
        threshold: Optional[float] = None,
    ) -> List[Entity]:
        """
        Detect PHI/PII entities in text using staged pipeline.
        
        Args:
            text: Input text to scan
            verify: Whether to run LLM verification
            threshold: Confidence threshold for verification
            
        Returns:
            List of detected entities
        """
        if threshold is None:
            threshold = self.config.confidence_threshold
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: PARALLEL DETECTION
        # ═══════════════════════════════════════════════════════════════
        phi_entities = self._run_phi_model(text)
        pii_entities = self._run_pii_model(text)
        presidio_entities = self._run_presidio(text)
        rule_entities = self._run_rules(text)
        
        # Tag sources
        for e in phi_entities:
            e._detector = "phi_bert"
        for e in pii_entities:
            e._detector = "pii_bert"
        for e in presidio_entities:
            e._detector = "presidio"
        for e in rule_entities:
            e._detector = "rule"
        
        # Combine all entities
        all_entities = phi_entities + pii_entities + presidio_entities + rule_entities
        
        if not all_entities:
            return []
        
        # Validate positions
        all_entities = self._validate_entity_positions(all_entities, text)
        if not all_entities:
            return []
        
        # Group overlapping spans
        all_entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        groups = self._group_overlapping_smart(all_entities)
        
        # Process each group through stages 2-6
        final_entities = []
        
        for group in groups:
            result = self._process_span_group(group, text)
            if result is not None:
                final_entities.append(result)
        
        # ═══════════════════════════════════════════════════════════════
        # POST-PROCESSING
        # ═══════════════════════════════════════════════════════════════
        
        # Run coreference to find additional mentions
        if self.use_coreference:
            final_entities = self._enrich_with_coreference(text, final_entities)
        
        # Apply allowlist
        final_entities = self._apply_allowlist(final_entities)
        
        # Run LLM verification on uncertain entities
        if verify and self.verifier and self.verifier.is_available():
            final_entities = self._verify_uncertain(final_entities, text, threshold)
        
        # Correct type confusion
        final_entities = self._correct_type_confusion(final_entities)
        
        return final_entities
    
    def _process_span_group(
        self, 
        group: List[Entity], 
        text: str
    ) -> Optional[Entity]:
        """
        Process a group of overlapping entities through stages 2-6.
        
        Returns final entity or None if rejected.
        """
        # Build signals from group
        signals = self._build_signals(group, text)
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: CHECKSUM FAST-PATH
        # ═══════════════════════════════════════════════════════════════
        if signals.rule_has_checksum and signals.rule_conf >= 0.95:
            # Algorithmic validation = near-certain, bypass ML
            result = self._create_entity_from_signals(signals, group, text)
            result.confidence = 0.99
            
            if self.capture_signals:
                signals.meta_decision = True
                signals.meta_confidence = 0.99
                self.captured_signals.append(signals)
            
            return result
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: BERT CALIBRATION + ROUTING
        # ═══════════════════════════════════════════════════════════════
        self._apply_calibration(signals)
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 4: RULE EVIDENCE BOOST
        # ═══════════════════════════════════════════════════════════════
        self._apply_rule_boost(signals)
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 5: META-CLASSIFIER DECISION
        # ═══════════════════════════════════════════════════════════════
        is_entity, confidence = self._meta_classify(signals)
        
        signals.meta_decision = is_entity
        signals.meta_confidence = confidence
        
        if self.capture_signals:
            self.captured_signals.append(signals)
        
        if not is_entity:
            return None
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 6: TYPE ASSIGNMENT
        # ═══════════════════════════════════════════════════════════════
        entity_type = self._resolve_type(signals, group)
        
        result = self._create_entity_from_signals(signals, group, text)
        result.entity_type = entity_type
        result.confidence = confidence
        
        return result
    
    def _build_signals(self, group: List[Entity], text: str) -> SpanSignals:
        """Build signal vector from overlapping entities."""
        # Use the longest span as the canonical span
        canonical = max(group, key=lambda e: e.end - e.start)
        
        signals = SpanSignals(
            span_start=canonical.start,
            span_end=canonical.end,
            span_text=text[canonical.start:canonical.end],
        )
        
        # Collect types for consensus check
        detected_types = set()
        
        # Extract signals from each detector
        for e in group:
            detector = getattr(e, "_detector", "unknown")
            etype = e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)
            detected_types.add(etype)
            
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
                # Check for checksum validation
                if e.confidence >= 0.95:
                    signals.rule_has_checksum = True
        
        # Compute agreement features
        signals.bert_agreement = signals.phi_bert_conf * signals.pii_bert_conf
        signals.bert_disagreement = abs(signals.phi_bert_conf - signals.pii_bert_conf)
        signals.bert_max = max(signals.phi_bert_conf, signals.pii_bert_conf)
        signals.type_consensus = len(detected_types) == 1
        
        # Compute rule tier
        from .calibration import RuleTier
        signals.rule_tier = RuleTier.get_tier(signals.rule_conf, signals.rule_has_checksum)
        
        # Computed features
        detectors = set(getattr(e, "_detector", "") for e in group)
        signals.sources_agree_count = len(detectors)
        signals.span_length = len(signals.span_text)
        signals.has_digits = any(c.isdigit() for c in signals.span_text)
        signals.has_letters = any(c.isalpha() for c in signals.span_text)
        signals.all_caps = signals.span_text.isupper() and signals.has_letters
        signals.all_digits = signals.span_text.replace("-", "").replace(" ", "").isdigit()
        signals.mixed_case = (
            any(c.isupper() for c in signals.span_text) and 
            any(c.islower() for c in signals.span_text)
        )
        
        # Store merged type (most common or highest confidence)
        best_entity = max(group, key=lambda e: e.confidence)
        signals.merged_type = best_entity.entity_type.value if hasattr(best_entity.entity_type, "value") else str(best_entity.entity_type)
        signals.merged_conf = best_entity.confidence
        signals.merged_source = getattr(best_entity, "_detector", "unknown")
        
        return signals
    
    def _apply_calibration(self, signals: SpanSignals):
        """
        Stage 3: Apply Platt scaling calibration to BERT scores.
        
        Routes to PHI-BERT, PII-BERT, or combines based on entity type authority.
        """
        if self._calibrator is None:
            # No calibration - use raw scores
            signals.bert_calibrated = max(signals.phi_bert_conf, signals.pii_bert_conf)
            return
        
        # Determine inferred type for routing
        inferred_type = signals.merged_type
        
        # Get calibrated combined score
        signals.bert_calibrated = self._calibrator.combine_scores(
            phi_raw=signals.phi_bert_conf,
            pii_raw=signals.pii_bert_conf,
            entity_type=inferred_type,
        )
        
        signals.bert_authority = self._calibrator.get_authority(inferred_type)
    
    def _apply_rule_boost(self, signals: SpanSignals):
        """
        Stage 4: Apply rule evidence boost to calibrated score.
        
        Strong patterns increase confidence, weak patterns give minor boost.
        """
        from .calibration import RuleTier
        
        base_score = signals.bert_calibrated
        
        # If no BERT detection but rule detected, use rule confidence as base
        if base_score == 0 and signals.rule_detected:
            base_score = signals.rule_conf
        
        # Apply tier-based boost
        signals.boosted_conf = RuleTier.apply_boost(base_score, signals.rule_tier)
    
    def _meta_classify(self, signals: SpanSignals) -> Tuple[bool, float]:
        """
        Stage 5: Meta-classifier decision.
        
        Returns (is_entity, confidence).
        """
        # Try trained meta-classifier first
        if self._meta_classifier and self._meta_classifier.is_trained():
            is_entity, _, confidence = self._meta_classifier.predict(signals)
            return is_entity, confidence
        
        # Fallback: threshold-based decision on boosted confidence
        conf = signals.boosted_conf
        
        if conf >= ACCEPT_THRESHOLD:
            return True, conf
        elif conf < REJECT_THRESHOLD:
            return False, conf
        else:
            # Uncertain zone - accept but flag
            # Could route to human review in production
            return True, conf
    
    def _resolve_type(self, signals: SpanSignals, group: List[Entity]) -> EntityType:
        """
        Stage 6: Resolve final entity type.
        
        Priority:
        1. Checksum-validated type (rule with validation)
        2. Type consensus (all sources agree)
        3. Authority BERT's type
        4. Highest confidence source's type
        """
        # 1. Checksum-validated
        if signals.rule_has_checksum and signals.rule_type:
            try:
                return EntityType(signals.rule_type)
            except ValueError:
                pass
        
        # 2. Type consensus
        if signals.type_consensus:
            try:
                return EntityType(signals.merged_type)
            except ValueError:
                pass
        
        # 3. Authority BERT's type
        if signals.bert_authority == "phi" and signals.phi_bert_type:
            try:
                return EntityType(signals.phi_bert_type)
            except ValueError:
                pass
        elif signals.bert_authority == "pii" and signals.pii_bert_type:
            try:
                return EntityType(signals.pii_bert_type)
            except ValueError:
                pass
        
        # 4. Highest confidence source
        best = max(group, key=lambda e: e.confidence)
        return best.entity_type
    
    def _create_entity_from_signals(
        self, 
        signals: SpanSignals, 
        group: List[Entity],
        text: str,
    ) -> Entity:
        """Create Entity object from signals."""
        return Entity(
            text=signals.span_text,
            start=signals.span_start,
            end=signals.span_end,
            entity_type=EntityType(signals.merged_type) if signals.merged_type else EntityType.OTHER,
            confidence=signals.boosted_conf,
            source=SourceType.MERGED if signals.sources_agree_count > 1 else SourceType.MODEL,
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # DETECTOR METHODS (unchanged from original)
    # ═══════════════════════════════════════════════════════════════════
    
    def _run_phi_model(self, text: str) -> List[Entity]:
        """Run Stanford PHI model."""
        if not self.phi_model:
            return []
        
        try:
            entities = self.phi_model.detect(text)
            if FILTER_ALL_OTHER:
                entities = [e for e in entities if e.entity_type != EntityType.OTHER]
            return entities
        except Exception as e:
            logger.error(f"PHI model failed: {e}")
            return []
    
    def _run_pii_model(self, text: str) -> List[Entity]:
        """Run PII BERT model."""
        if not self.pii_model:
            return []
        
        try:
            entities = self.pii_model.detect(text)
            for e in entities:
                e._detector = "pii_bert"
            if FILTER_ALL_OTHER:
                entities = [e for e in entities if e.entity_type != EntityType.OTHER]
            return entities
        except Exception as e:
            logger.error(f"PII model failed: {e}")
            return []
    
    def _run_presidio(self, text: str) -> List[Entity]:
        """Run Presidio detector."""
        if not self.presidio:
            return []
        
        try:
            entities = self.presidio.detect(text)
            for e in entities:
                e._detector = "presidio"
            return entities
        except Exception as e:
            logger.error(f"Presidio failed: {e}")
            return []
    
    def _run_rules(self, text: str) -> List[Entity]:
        """Run rule-based detection."""
        if not self.rules:
            return []
        
        try:
            entities = self.rules.detect(text)
            for e in entities:
                e._detector = "rule"
            return entities
        except Exception as e:
            logger.error(f"Rules failed: {e}")
            return []
    
    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def _group_overlapping_smart(self, entities: List[Entity]) -> List[List[Entity]]:
        """Group entities that truly overlap with each other."""
        if not entities:
            return []
        
        n = len(entities)
        overlaps = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = entities[i], entities[j]
                if e1.start < e2.end and e2.start < e1.end:
                    overlaps[i].add(j)
                    overlaps[j].add(i)
        
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            group_indices = []
            queue = [i]
            visited[i] = True
            
            while queue:
                curr = queue.pop(0)
                group_indices.append(curr)
                
                for neighbor in overlaps[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            group_indices.sort(key=lambda idx: entities[idx].start)
            groups.append([entities[idx] for idx in group_indices])
        
        groups.sort(key=lambda g: g[0].start)
        return groups
    
    def _validate_entity_positions(self, entities: List[Entity], text: str) -> List[Entity]:
        """Validate that entity positions match entity text."""
        validated = []
        
        for entity in entities:
            if entity.start < 0 or entity.end > len(text) or entity.start >= entity.end:
                continue
            
            actual_text = text[entity.start:entity.end]
            
            if actual_text == entity.text:
                validated.append(entity)
            elif actual_text.strip() == entity.text.strip():
                entity.text = actual_text
                validated.append(entity)
            else:
                # Try to find nearby
                search_start = max(0, entity.start - 20)
                search_end = min(len(text), entity.end + 20)
                search_region = text[search_start:search_end]
                
                idx = search_region.find(entity.text)
                if idx >= 0:
                    entity.start = search_start + idx
                    entity.end = entity.start + len(entity.text)
                    validated.append(entity)
        
        return validated
    
    def _apply_allowlist(self, entities: List[Entity]) -> List[Entity]:
        """Filter out allowlisted terms."""
        return [e for e in entities if not is_allowed(e.text)]
    
    def _correct_type_confusion(self, entities: List[Entity]) -> List[Entity]:
        """Post-process entities to fix common type confusions."""
        corrected = []
        
        for entity in entities:
            digits_only = ''.join(c for c in entity.text if c.isdigit())
            text_clean = entity.text.replace('-', '').replace(' ', '')
            
            # 16-digit MRN → CREDIT_CARD
            if (entity.entity_type == EntityType.MRN and 
                len(digits_only) in (15, 16) and
                len(text_clean) <= 19):
                entity = Entity(
                    text=entity.text,
                    start=entity.start,
                    end=entity.end,
                    entity_type=EntityType.CREDIT_CARD,
                    confidence=entity.confidence * 0.95,
                    source=entity.source,
                )
            
            # DATE with 15-16 digits → CREDIT_CARD
            elif (entity.entity_type == EntityType.DATE and
                  len(digits_only) in (15, 16) and
                  text_clean.isdigit()):
                entity = Entity(
                    text=entity.text,
                    start=entity.start,
                    end=entity.end,
                    entity_type=EntityType.CREDIT_CARD,
                    confidence=entity.confidence * 0.90,
                    source=entity.source,
                )
            
            corrected.append(entity)
        
        return corrected
    
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
                result = self.verifier.verify(
                    entity_text=entity.text,
                    entity_type=entity.entity_type,
                    context=text
                )
                if result.decision in (VerificationResult.YES, VerificationResult.UNCERTAIN):
                    entity.llm_confidence = result.confidence
                    entity.llm_reasoning = result.reasoning
                    verified.append(entity)
        
        return verified
    
    def _enrich_with_coreference(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> List[Entity]:
        """Use coreference to find additional PHI mentions."""
        resolver = self._get_coref_resolver()
        if resolver is None:
            return entities
        
        try:
            from .coreference import CorefResult
            
            coref_result = resolver.resolve(text)
            
            detected_spans = [
                {
                    'start': e.start,
                    'end': e.end,
                    'text': e.text,
                    'entity_type': e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type),
                    'confidence': e.confidence,
                }
                for e in entities
            ]
            
            coref_result = resolver.enrich_with_phi(coref_result, detected_spans)
            new_spans = resolver.get_additional_phi_spans(coref_result, detected_spans)
            
            for span in new_spans:
                anchor_type = span.get('coref_anchor_type', 'OTHER')
                try:
                    entity_type = EntityType(anchor_type)
                except ValueError:
                    entity_type = EntityType.OTHER
                
                new_entity = Entity(
                    text=span['text'],
                    start=span['start'],
                    end=span['end'],
                    entity_type=entity_type,
                    confidence=span.get('coref_anchor_conf', 0.5) * 0.8,
                    source=SourceType.MERGED,
                )
                new_entity._detector = "coreference"
                entities.append(new_entity)
            
        except Exception as e:
            logger.warning(f"Coreference enrichment failed: {e}")
        
        return entities
    
    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════
    
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
            "coreference": {
                "enabled": self.use_coreference,
                "available": self._coref_resolver is not None,
            },
            "calibration": {
                "loaded": self._calibrator.is_loaded() if self._calibrator else False,
            },
            "meta_classifier": {
                "trained": self._meta_classifier.is_trained() if self._meta_classifier else False,
            },
            "verifier": {
                "provider": getattr(self.verifier, "provider", "unknown") if self.verifier else None,
                "available": self.verifier.is_available() if self.verifier else False,
            },
        }
