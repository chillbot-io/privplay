"""Coreference resolution for PHI detection.

Enriches detected PHI spans with coreference information so that
pronouns and references ("He", "the patient", "Mr. Smith") linked
to confirmed PHI get flagged.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class CorefCluster:
    """A cluster of coreferent mentions."""
    cluster_id: int
    mentions: List[Tuple[int, int, str]]  # (start, end, text)
    
    # PHI anchor info (if any mention in cluster is detected PHI)
    anchor_span: Optional[Tuple[int, int]] = None
    anchor_type: Optional[str] = None
    anchor_confidence: float = 0.0


@dataclass 
class CorefResult:
    """Result of coreference resolution on a document."""
    clusters: List[CorefCluster]
    span_to_cluster: Dict[Tuple[int, int], int]  # span -> cluster_id


class CoreferenceResolver:
    """Resolve coreferences using FastCoref."""
    
    _instance = None
    
    def __new__(cls, device: str = 'cpu'):
        """Singleton pattern - only load model once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, device: str = 'cpu'):
        if self._initialized:
            return
            
        logger.info(f"Loading FastCoref model on {device}...")
        from fastcoref import FCoref
        self.model = FCoref(device=device)
        self._initialized = True
        logger.info("FastCoref model loaded")
    
    def resolve(self, text: str) -> CorefResult:
        """
        Run coreference resolution on text.
        
        Returns:
            CorefResult with clusters and span mapping
        """
        preds = self.model.predict(texts=[text])
        
        clusters = []
        span_to_cluster = {}
        
        # Get clusters as spans (character offsets)
        raw_clusters = preds[0].get_clusters(as_strings=False)
        raw_strings = preds[0].get_clusters(as_strings=True)
        
        for cluster_id, (spans, strings) in enumerate(zip(raw_clusters, raw_strings)):
            mentions = []
            for (start, end), text_str in zip(spans, strings):
                mentions.append((start, end, text_str))
                span_to_cluster[(start, end)] = cluster_id
            
            clusters.append(CorefCluster(
                cluster_id=cluster_id,
                mentions=mentions,
            ))
        
        return CorefResult(clusters=clusters, span_to_cluster=span_to_cluster)
    
    def enrich_with_phi(
        self,
        coref_result: CorefResult,
        detected_spans: List[Dict],  # [{start, end, entity_type, confidence}, ...]
    ) -> CorefResult:
        """
        Enrich coref clusters with PHI detection info.
        
        For each cluster, find if any mention overlaps with detected PHI.
        If so, mark that as the "anchor" and propagate its type/confidence
        to the whole cluster.
        """
        # Build lookup for detected PHI
        phi_lookup = {}
        for span in detected_spans:
            key = (span['start'], span['end'])
            phi_lookup[key] = span
        
        # For each cluster, find PHI anchors
        for cluster in coref_result.clusters:
            best_anchor = None
            best_conf = 0.0
            
            for mention_start, mention_end, mention_text in cluster.mentions:
                # Check for exact match
                key = (mention_start, mention_end)
                if key in phi_lookup:
                    phi = phi_lookup[key]
                    if phi.get('confidence', 0) > best_conf:
                        best_anchor = phi
                        best_conf = phi.get('confidence', 0)
                    continue
                
                # Check for overlapping match (50%+ overlap)
                for (phi_start, phi_end), phi in phi_lookup.items():
                    overlap = self._compute_overlap(
                        mention_start, mention_end,
                        phi_start, phi_end
                    )
                    if overlap > 0.5:
                        if phi.get('confidence', 0) > best_conf:
                            best_anchor = phi
                            best_conf = phi.get('confidence', 0)
            
            if best_anchor:
                cluster.anchor_span = (best_anchor['start'], best_anchor['end'])
                cluster.anchor_type = best_anchor.get('entity_type')
                cluster.anchor_confidence = best_anchor.get('confidence', 0.0)
        
        return coref_result
    
    def _compute_overlap(
        self, 
        start1: int, end1: int,
        start2: int, end2: int
    ) -> float:
        """Compute overlap ratio between two spans."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_len = overlap_end - overlap_start
        span1_len = end1 - start1
        span2_len = end2 - start2
        min_len = min(span1_len, span2_len)
        
        return overlap_len / min_len if min_len > 0 else 0.0
    
    # Pronouns - NEVER surface as standalone entities
    # HIPAA Safe Harbor requires removing the 18 identifiers, not references to them.
    # Once "John Smith" is redacted, "he" becomes non-identifying.
    # These are still useful as SIGNALS for the meta-classifier, but should not
    # become standalone Entity objects.
    PRONOUNS = {
        # First person
        'i', 'me', 'my', 'mine', 'myself',
        # Second person
        'you', 'your', 'yours', 'yourself', 'yourselves',
        # Third person singular
        'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself',
        # Third person plural
        'they', 'them', 'their', 'theirs', 'themselves',
        # First person plural
        'we', 'us', 'our', 'ours', 'ourselves',
        # Demonstratives
        'this', 'that', 'these', 'those',
        # Relative/interrogative
        'who', 'whom', 'whose', 'which', 'what',
        # Indefinite (common ones that get flagged)
        'one', 'ones',
    }
    
    def get_additional_phi_spans(
        self,
        coref_result: CorefResult,
        detected_spans: List[Dict],
    ) -> List[Dict]:
        """
        Get NEW spans that should be flagged as PHI based on coreference.
        
        Returns substantive mentions like "the patient", "Mr. Smith", "Dr. Jones"
        that corefer with detected PHI.
        
        IMPORTANT: Pure pronouns (he/she/your/etc.) are NOT surfaced as entities.
        Per HIPAA Safe Harbor, we only need to redact the 18 identifiers themselves,
        not references to them. Once "John Smith" is redacted, "he" is non-identifying.
        Pronouns are still captured as SIGNALS for the meta-classifier.
        
        Returns:
            List of new span dicts with coref signals (excluding pronouns)
        """
        # Build set of already-detected spans
        detected_set = {(s['start'], s['end']) for s in detected_spans}
        
        new_spans = []
        
        for cluster in coref_result.clusters:
            # Skip clusters without PHI anchor
            if cluster.anchor_type is None:
                continue
            
            for mention_start, mention_end, mention_text in cluster.mentions:
                key = (mention_start, mention_end)
                
                # Skip if already detected
                if key in detected_set:
                    continue
                
                # Skip if this IS the anchor
                if cluster.anchor_span == key:
                    continue
                
                # SKIP pure pronouns - they shouldn't be standalone entities
                mention_clean = mention_text.lower().strip()
                if mention_clean in self.PRONOUNS:
                    logger.debug(f"Skipping pronoun '{mention_text}' (corefers with {cluster.anchor_type})")
                    continue
                
                # SKIP very short spans (likely fragments or missed pronouns)
                if len(mention_clean) <= 2:
                    logger.debug(f"Skipping short span '{mention_text}' (len <= 2)")
                    continue
                
                # This is a substantive NEW span we should flag
                # Examples: "the patient", "Mr. Smith", "Dr. Jones", "the defendant"
                new_spans.append({
                    'start': mention_start,
                    'end': mention_end,
                    'text': mention_text,
                    'entity_type': cluster.anchor_type,  # Inherit from anchor
                    'confidence': 0.0,  # No direct detection
                    'source': 'coreference',
                    # Coref signals for meta-classifier
                    'in_coref_cluster': True,
                    'coref_cluster_id': cluster.cluster_id,
                    'coref_cluster_size': len(cluster.mentions),
                    'coref_anchor_type': cluster.anchor_type,
                    'coref_anchor_conf': cluster.anchor_confidence,
                })
        
        logger.debug(f"Coreference: {len(new_spans)} substantive mentions (pronouns filtered)")
        return new_spans
    
    def enrich_signals(
        self,
        text: str,
        signals: List[Dict],
    ) -> List[Dict]:
        """
        Full pipeline: resolve coreference and enrich signals.
        
        Args:
            text: The document text
            signals: List of SpanSignal dicts from detectors
            
        Returns:
            Enriched signals + new coref-derived signals
        """
        # Run coreference
        coref_result = self.resolve(text)
        
        # Build detected spans list
        detected_spans = [
            {
                'start': s.get('start'),
                'end': s.get('end'),
                'text': s.get('text'),
                'entity_type': s.get('entity_type') or s.get('detected_type'),
                'confidence': s.get('confidence', s.get('meta_confidence', 0.5)),
            }
            for s in signals
        ]
        
        # Enrich with PHI info
        coref_result = self.enrich_with_phi(coref_result, detected_spans)
        
        # Add coref signals to existing spans
        enriched_signals = []
        for signal in signals:
            signal = dict(signal)  # Copy
            key = (signal.get('start'), signal.get('end'))
            
            if key in coref_result.span_to_cluster:
                cluster_id = coref_result.span_to_cluster[key]
                cluster = coref_result.clusters[cluster_id]
                
                signal['in_coref_cluster'] = True
                signal['coref_cluster_id'] = cluster_id
                signal['coref_cluster_size'] = len(cluster.mentions)
                signal['coref_anchor_type'] = cluster.anchor_type
                signal['coref_anchor_conf'] = cluster.anchor_confidence
            else:
                signal['in_coref_cluster'] = False
                signal['coref_cluster_id'] = None
                signal['coref_cluster_size'] = 0
                signal['coref_anchor_type'] = None
                signal['coref_anchor_conf'] = 0.0
            
            enriched_signals.append(signal)
        
        # Get new spans from coreference
        new_spans = self.get_additional_phi_spans(coref_result, detected_spans)
        
        # Add new spans as signals
        for span in new_spans:
            enriched_signals.append(span)
        
        return enriched_signals


# Convenience function
_resolver = None

def get_coreference_resolver(device: str = 'cpu') -> CoreferenceResolver:
    """Get singleton coreference resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = CoreferenceResolver(device=device)
    return _resolver


def enrich_signals_with_coreference(
    text: str,
    signals: List[Dict],
    device: str = 'cpu',
) -> List[Dict]:
    """
    Convenience function to enrich signals with coreference.
    
    Args:
        text: Document text
        signals: SpanSignal dicts from detectors
        device: 'cpu' or 'cuda:0'
        
    Returns:
        Enriched signals + new coref-derived signals
    """
    resolver = get_coreference_resolver(device)
    return resolver.enrich_signals(text, signals)