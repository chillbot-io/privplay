"""SHDM Tokenizer - Convert detected PHI entities to tokens.

Takes detection results and replaces PHI with consistent tokens.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedEntity:
    """An entity detected by the classification engine."""
    start: int
    end: int
    text: str
    entity_type: str
    confidence: float = 1.0


def refine_person_type(text: str, entity: DetectedEntity) -> str:
    """
    Refine NAME_PERSON to NAME_PATIENT or NAME_PROVIDER based on context.
    
    Args:
        text: Full document text
        entity: The detected entity
        
    Returns:
        Refined entity type string
    """
    etype = entity.entity_type
    if etype not in ("NAME_PERSON", "PERSON"):
        return etype
    
    # Get context before the entity (up to 50 chars)
    context_start = max(0, entity.start - 50)
    context_before = text[context_start:entity.start].lower()
    
    # Get context after the entity (up to 30 chars)
    context_end = min(len(text), entity.end + 30)
    context_after = text[entity.end:context_end].lower()
    
    # Provider indicators (before the name)
    provider_before = ['dr.', 'dr ', 'doctor ', 'physician ', 'nurse ', 'np ', 
                       'pa ', 'practitioner', 'specialist', 'surgeon', 
                       'therapist', 'pharmacist', 'attending ', 'resident ',
                       'fellow ', 'intern ']
    
    # Provider indicators (after the name)
    provider_after = [', md', ', do', ', rn', ', np', ', pa', 'm.d.', 'd.o.',
                      ', phd', ', pharmd', ', dds', ', dpm']
    
    for indicator in provider_before:
        if indicator in context_before:
            return "NAME_PROVIDER"
    
    for indicator in provider_after:
        if context_after.startswith(indicator) or indicator in context_after[:15]:
            return "NAME_PROVIDER"
    
    # Patient indicators (before the name)
    patient_before = ['patient ', 'pt ', 'pt.', 'client ', 'resident ', 
                      'admitted ', 'presented ', 'complains ', 'reports ',
                      'states ', 'the ', 'mr. ', 'mrs. ', 'ms. ', 'miss ']
    
    # Patient indicators (after the name)
    patient_after = ['was admitted', 'presented', 'complains', 'reports', 
                     'denies', 'is a ', 'year old', 'y/o', 'yo ', 
                     ' was seen', ' came in', ' called']
    
    for indicator in patient_before:
        if indicator in context_before:
            return "NAME_PATIENT"
    
    for indicator in patient_after:
        if indicator in context_after:
            return "NAME_PATIENT"
    
    # Default to PATIENT in clinical context (safer for privacy)
    return "NAME_PATIENT"


class Tokenizer:
    """Converts detected PHI entities to tokens in text.
    
    Usage:
        from privplay.shdm import SHDMStore, Tokenizer
        
        store = SHDMStore(db_path, key)
        tokenizer = Tokenizer(store)
        
        # After detection
        entities = [
            DetectedEntity(8, 18, "John Smith", "NAME_PATIENT", 0.95),
            DetectedEntity(35, 46, "123-45-6789", "SSN", 0.99),
        ]
        
        original = "Patient John Smith has SSN 123-45-6789"
        redacted = tokenizer.tokenize(conv_id, original, entities)
        # "Patient [PATIENT_1] has SSN [SSN_1]"
    """
    
    def __init__(self, store, refine_person_types: bool = True):
        """Initialize with SHDM store.
        
        Args:
            store: SHDMStore instance for token persistence
            refine_person_types: If True, refine NAME_PERSON to PATIENT/PROVIDER
        """
        self.store = store
        self.refine_person_types = refine_person_types
    
    def tokenize(
        self, 
        conv_id: str, 
        text: str, 
        entities: List[DetectedEntity],
    ) -> str:
        """Replace detected entities with tokens.
        
        Args:
            conv_id: Conversation ID for consistent token mapping
            text: Original text containing PHI
            entities: List of detected entities with positions
            
        Returns:
            Text with PHI replaced by tokens
        """
        if not entities:
            return text
        
        # Sort entities by start position (descending) so we can replace
        # from end to start without messing up offsets
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        result = text
        for entity in sorted_entities:
            # Refine person types based on context
            entity_type = entity.entity_type
            if self.refine_person_types and entity_type in ("NAME_PERSON", "PERSON"):
                entity_type = refine_person_type(text, entity)
            
            # Get or create token for this entity
            token = self.store.get_or_create_token(
                conv_id, 
                entity.text, 
                entity_type
            )
            
            # Replace in text
            result = result[:entity.start] + token + result[entity.end:]
        
        return result
    
    def tokenize_with_map(
        self,
        conv_id: str,
        text: str,
        entities: List[DetectedEntity],
    ) -> Tuple[str, dict]:
        """Replace entities and return the token mapping.
        
        Returns:
            Tuple of (tokenized_text, {original: token} mapping)
        """
        if not entities:
            return text, {}
        
        token_map = {}
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        result = text
        for entity in sorted_entities:
            # Refine person types based on context
            entity_type = entity.entity_type
            if self.refine_person_types and entity_type in ("NAME_PERSON", "PERSON"):
                entity_type = refine_person_type(text, entity)
            
            token = self.store.get_or_create_token(
                conv_id,
                entity.text,
                entity_type
            )
            token_map[entity.text] = token
            result = result[:entity.start] + token + result[entity.end:]
        
        return result, token_map


class Restorer:
    """Restores tokens back to original PHI values.
    
    Usage:
        restorer = Restorer(store)
        
        llm_response = "I recommend [PATIENT_1] take medication..."
        restored = restorer.restore(conv_id, llm_response)
        # "I recommend John Smith take medication..."
    """
    
    # Pattern to match tokens like [PATIENT_1], [SSN_2], [PROVIDER_1]
    TOKEN_PATTERN = re.compile(r'\[([A-Z_]+)_(\d+)\]')
    
    def __init__(self, store):
        """Initialize with SHDM store.
        
        Args:
            store: SHDMStore instance for token lookup
        """
        self.store = store
    
    def restore(self, conv_id: str, text: str) -> str:
        """Replace tokens with original PHI values.
        
        Args:
            conv_id: Conversation ID to look up tokens
            text: Text containing tokens from LLM response
            
        Returns:
            Text with tokens replaced by original PHI
        """
        def replace_token(match):
            token = match.group(0)  # e.g., "[PATIENT_1]"
            original = self.store.lookup_token(conv_id, token)
            return original if original else token  # Keep token if not found
        
        return self.TOKEN_PATTERN.sub(replace_token, text)
    
    def restore_with_map(self, conv_id: str, text: str) -> Tuple[str, dict]:
        """Replace tokens and return what was restored.
        
        Returns:
            Tuple of (restored_text, {token: original} mapping of what was replaced)
        """
        restored_map = {}
        
        def replace_token(match):
            token = match.group(0)
            original = self.store.lookup_token(conv_id, token)
            if original:
                restored_map[token] = original
                return original
            return token
        
        restored = self.TOKEN_PATTERN.sub(replace_token, text)
        return restored, restored_map
    
    def find_tokens(self, text: str) -> List[str]:
        """Find all tokens in text without restoring.
        
        Useful for checking if LLM response contains expected tokens.
        """
        return self.TOKEN_PATTERN.findall(text)
    
    def has_tokens(self, text: str) -> bool:
        """Check if text contains any tokens."""
        return bool(self.TOKEN_PATTERN.search(text))


# =============================================================================
# INTEGRATION WITH DETECTION ENGINE
# =============================================================================

def entities_from_detection(detection_results: List) -> List[DetectedEntity]:
    """Convert detection engine results to DetectedEntity objects.
    
    This bridges the detection engine output format to SHDM input format.
    
    Args:
        detection_results: List of entities from ClassificationEngine.detect()
        
    Returns:
        List of DetectedEntity objects
    """
    entities = []
    
    for result in detection_results:
        # Handle different result formats
        if hasattr(result, 'start') and hasattr(result, 'end'):
            # Object with attributes
            entities.append(DetectedEntity(
                start=result.start,
                end=result.end,
                text=result.text if hasattr(result, 'text') else "",
                entity_type=result.entity_type if hasattr(result, 'entity_type') else str(result.type),
                confidence=result.confidence if hasattr(result, 'confidence') else 1.0,
            ))
        elif isinstance(result, dict):
            # Dictionary format
            entities.append(DetectedEntity(
                start=result.get('start', 0),
                end=result.get('end', 0),
                text=result.get('text', ''),
                entity_type=result.get('entity_type', result.get('type', '')),
                confidence=result.get('confidence', 1.0),
            ))
        elif isinstance(result, tuple) and len(result) >= 4:
            # Tuple format (start, end, text, type, [confidence])
            entities.append(DetectedEntity(
                start=result[0],
                end=result[1],
                text=result[2],
                entity_type=result[3],
                confidence=result[4] if len(result) > 4 else 1.0,
            ))
    
    return entities
