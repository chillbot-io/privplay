"""Integration between ClassificationEngine and Continuous Learning.

Wires detection signals to the SignalStore for continuous learning.

Usage:
    from privplay.learning.integration import LearningClassifier
    
    # Create classifier with learning enabled
    classifier = LearningClassifier(
        capture_for_learning=True,
        document_id="doc_123",
    )
    
    # Detection automatically captures signals
    entities = classifier.detect(text)
    
    # Or wrap existing engine
    from privplay.engine.classifier import ClassificationEngine
    from privplay.learning.integration import wrap_for_learning
    
    engine = ClassificationEngine()
    learning_engine = wrap_for_learning(engine)
"""

import secrets
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..engine.classifier import ClassificationEngine, SpanSignals
from ..types import Entity
from ..config import Config, get_config
from .signal_store import (
    SignalStore,
    CapturedSignal,
    get_signal_store,
)

logger = logging.getLogger(__name__)


def span_signals_to_captured(
    signal: SpanSignals,
    document_id: str,
    doc_type: str = "",
    conversation_id: Optional[str] = None,
    model_version: str = "",
) -> CapturedSignal:
    """Convert SpanSignals to CapturedSignal for storage."""
    return CapturedSignal(
        id=signal.id,
        created_at=datetime.utcnow(),
        
        # Source
        document_id=document_id,
        doc_type=doc_type,
        conversation_id=conversation_id,
        
        # Span
        span_start=signal.span_start,
        span_end=signal.span_end,
        span_text=signal.span_text,
        
        # Detector signals
        phi_bert_detected=signal.phi_bert_detected,
        phi_bert_conf=signal.phi_bert_conf,
        phi_bert_type=signal.phi_bert_type,
        
        pii_bert_detected=signal.pii_bert_detected,
        pii_bert_conf=signal.pii_bert_conf,
        pii_bert_type=signal.pii_bert_type,
        
        presidio_detected=signal.presidio_detected,
        presidio_conf=signal.presidio_conf,
        presidio_type=signal.presidio_type,
        
        rule_detected=signal.rule_detected,
        rule_conf=signal.rule_conf,
        rule_type=signal.rule_type,
        rule_has_checksum=signal.rule_has_checksum,
        
        # Coreference
        in_coref_cluster=signal.in_coref_cluster,
        coref_cluster_size=signal.coref_cluster_size,
        coref_anchor_conf=signal.coref_anchor_conf,
        coref_is_anchor=signal.coref_is_anchor,
        coref_is_pronoun=signal.coref_is_pronoun,
        coref_anchor_type=signal.coref_anchor_type,
        
        # Document-level
        doc_has_ssn=signal.doc_has_ssn,
        doc_has_dates=signal.doc_has_dates,
        doc_has_medical_terms=signal.doc_has_medical_terms,
        doc_entity_count=signal.doc_entity_count,
        doc_pii_density=signal.doc_pii_density,
        
        # Computed
        sources_agree_count=signal.sources_agree_count,
        span_length=signal.span_length,
        has_digits=signal.has_digits,
        has_letters=signal.has_letters,
        all_caps=signal.all_caps,
        all_digits=signal.all_digits,
        mixed_case=signal.mixed_case,
        
        # Merged output
        merged_type=signal.merged_type,
        merged_conf=signal.merged_conf,
        merged_source=signal.merged_source,
        
        # Model
        model_version=model_version,
    )


def compute_document_features_from_signals(signals: List[SpanSignals]) -> Dict[str, Any]:
    """Compute document-level features from all signals in a document."""
    detected_types = set()
    for signal in signals:
        if signal.merged_type:
            detected_types.add(signal.merged_type.upper())
        if signal.rule_type:
            detected_types.add(signal.rule_type.upper())
        if signal.phi_bert_type:
            detected_types.add(signal.phi_bert_type.upper())
        if signal.pii_bert_type:
            detected_types.add(signal.pii_bert_type.upper())
    
    doc_has_ssn = "SSN" in detected_types
    
    date_types = {"DATE", "DATE_DOB", "DATE_ADMISSION", "DATE_DISCHARGE"}
    doc_has_dates = bool(detected_types & date_types)
    
    medical_types = {"DRUG", "DIAGNOSIS", "LAB_TEST", "FACILITY", "NAME_PROVIDER", "MRN"}
    doc_has_medical_terms = bool(detected_types & medical_types)
    
    doc_entity_count = len(signals)
    
    return {
        "doc_has_ssn": doc_has_ssn,
        "doc_has_dates": doc_has_dates,
        "doc_has_medical_terms": doc_has_medical_terms,
        "doc_entity_count": doc_entity_count,
    }


class LearningClassifier:
    """ClassificationEngine wrapper that captures signals for continuous learning.
    
    Wraps the detection flow to:
    1. Run detection with signal capture enabled
    2. Compute document-level features
    3. Store signals with confidence < 0.95 in SignalStore
    
    Usage:
        classifier = LearningClassifier()
        
        # Set document context before detection
        classifier.set_document_context(
            document_id="doc_123",
            doc_type="clinical_note",
            conversation_id="conv_456",
        )
        
        # Detect (signals automatically captured)
        entities = classifier.detect(text)
    """
    
    def __init__(
        self,
        engine: Optional[ClassificationEngine] = None,
        signal_store: Optional[SignalStore] = None,
        config: Optional[Config] = None,
        model_version: str = "",
    ):
        """Initialize learning classifier.
        
        Args:
            engine: ClassificationEngine to wrap (creates new if not provided)
            signal_store: SignalStore for persisting signals
            config: Configuration
            model_version: Current model version string
        """
        self.config = config or get_config()
        
        # Create or wrap engine with signal capture enabled
        if engine:
            self._engine = engine
            self._engine.capture_signals = True
        else:
            self._engine = ClassificationEngine(
                config=self.config,
                capture_signals=True,
            )
        
        self._signal_store = signal_store or get_signal_store()
        self._model_version = model_version
        
        # Document context (set before detection)
        self._document_id: Optional[str] = None
        self._doc_type: str = ""
        self._conversation_id: Optional[str] = None
    
    def set_document_context(
        self,
        document_id: Optional[str] = None,
        doc_type: str = "",
        conversation_id: Optional[str] = None,
    ):
        """Set context for the next detection.
        
        Args:
            document_id: Unique document identifier
            doc_type: Document type (clinical_note, etc.)
            conversation_id: Conversation ID if in chat context
        """
        self._document_id = document_id or secrets.token_hex(8)
        self._doc_type = doc_type
        self._conversation_id = conversation_id
    
    def detect(
        self,
        text: str,
        verify: bool = True,
        threshold: Optional[float] = None,
    ) -> List[Entity]:
        """Detect entities and capture signals for learning.
        
        Args:
            text: Text to scan
            verify: Whether to run LLM verification
            threshold: Confidence threshold
            
        Returns:
            List of detected entities
        """
        # Ensure we have a document ID
        if not self._document_id:
            self._document_id = secrets.token_hex(8)
        
        # Clear previous signals
        self._engine.clear_captured_signals()
        
        # Run detection
        entities = self._engine.detect(text, verify=verify, threshold=threshold)
        
        # Get captured signals
        signals = self._engine.get_captured_signals()
        
        if signals:
            # Compute document-level features
            doc_features = compute_document_features_from_signals(signals)
            
            # Also compute PII density
            total_pii_chars = sum(s.span_end - s.span_start for s in signals)
            doc_pii_density = min(1.0, total_pii_chars / max(1, len(text)))
            doc_features["doc_pii_density"] = doc_pii_density
            
            # Update signals with document features and store
            captured_count = 0
            for signal in signals:
                # Add document features
                signal.doc_has_ssn = doc_features["doc_has_ssn"]
                signal.doc_has_dates = doc_features["doc_has_dates"]
                signal.doc_has_medical_terms = doc_features["doc_has_medical_terms"]
                signal.doc_entity_count = doc_features["doc_entity_count"]
                signal.doc_pii_density = doc_features["doc_pii_density"]
                signal.doc_type = self._doc_type
                
                # Convert and store
                captured = span_signals_to_captured(
                    signal,
                    document_id=self._document_id,
                    doc_type=self._doc_type,
                    conversation_id=self._conversation_id,
                    model_version=self._model_version,
                )
                
                # Store captures uncertain signals (conf < 0.95)
                self._signal_store.capture_signal(captured)
                if captured.merged_conf < 0.95:
                    captured_count += 1
            
            if captured_count > 0:
                logger.debug(f"Captured {captured_count} uncertain signals for review")
        
        # Reset context for next detection
        self._document_id = None
        
        return entities
    
    @property
    def engine(self) -> ClassificationEngine:
        """Access underlying engine."""
        return self._engine
    
    def get_stack_status(self) -> Dict[str, Any]:
        """Get detection stack status."""
        status = self._engine.get_stack_status()
        status["learning"] = {
            "enabled": True,
            "pending_review": self._signal_store.get_pending_count(),
            "labeled_signals": self._signal_store.get_labeled_signal_count(),
        }
        return status


def wrap_for_learning(
    engine: ClassificationEngine,
    signal_store: Optional[SignalStore] = None,
    model_version: str = "",
) -> LearningClassifier:
    """Wrap an existing ClassificationEngine for continuous learning.
    
    Args:
        engine: Engine to wrap
        signal_store: Signal store to use
        model_version: Current model version
        
    Returns:
        LearningClassifier wrapping the engine
    """
    return LearningClassifier(
        engine=engine,
        signal_store=signal_store,
        model_version=model_version,
    )
