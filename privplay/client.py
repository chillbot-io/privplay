"""Privacy Client - High-level interface for safe LLM interaction.

Wires together:
- ClassificationEngine (PHI/PII detection)
- SHDM (tokenization, storage, restoration)
- SessionManager (conversation management)

This is the main entry point for applications that want to safely
interact with LLMs while protecting PHI/PII.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

from .shdm import (
    SHDMStore,
    SessionManager,
    Encryptor,
    DetectedEntity,
    LLMContext,
    Conversation,
)
from .engine.classifier import ClassificationEngine, SpanSignals
from .types import Entity, EntityType
from .config import get_config, Config

logger = logging.getLogger(__name__)


@dataclass
class PrivacyResponse:
    """Response from the privacy client."""
    # For user display (PHI restored)
    content: str
    
    # What was sent to LLM (tokenized)
    content_redacted: str
    
    # Token mappings used
    tokens_used: Dict[str, str]  # token -> original
    
    # Conversation ID
    conversation_id: str


@dataclass 
class ProcessedInput:
    """Processed user input ready for LLM."""
    # Tokenized text
    redacted: str
    
    # Full context for LLM
    context: LLMContext
    
    # Entities detected
    entities: List[Entity]
    
    # Token mappings created
    tokens_created: Dict[str, str]  # original -> token
    
    # Conversation ID
    conversation_id: str


class PrivacyClient:
    """High-level client for privacy-safe LLM interactions.
    
    This is the main interface for applications. It handles:
    - PHI/PII detection in user input
    - Tokenization (PHI -> tokens) before sending to LLM
    - Token restoration (tokens -> PHI) in LLM responses
    - Conversation management with compaction
    
    Usage:
        # Initialize
        client = PrivacyClient()
        
        # Start conversation
        conv_id = client.create_conversation()
        
        # Process user input
        processed = client.process_input(
            conv_id,
            "What about John Smith's lab results?",
            system_prompt="You are a medical assistant."
        )
        
        # Send to your LLM
        llm_response = your_llm.chat(processed.context.to_messages())
        
        # Process response (restores tokens)
        response = client.process_response(conv_id, llm_response)
        
        # Show response.content to user (has real names)
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        config: Optional[Config] = None,
        use_mock_model: bool = False,
        max_context_tokens: int = 8000,
        keep_recent_messages: int = 10,
    ):
        """Initialize the privacy client.
        
        Args:
            db_path: Path to SHDM database. Defaults to ~/.privplay/shdm.db
            encryption_key: 32-byte key for encrypting PHI. Generated if not provided.
            config: Privplay config. Uses default if not provided.
            use_mock_model: Use mock detection model (for testing)
            max_context_tokens: Trigger compaction at this token count
            keep_recent_messages: Keep this many recent messages uncompacted
        """
        self.config = config or get_config()
        
        # Database path
        if db_path is None:
            db_path = str(self.config.data_dir / "shdm.db")
        
        # Encryption key
        if encryption_key is None:
            # Try to load from file or generate new
            key_path = Path(db_path).parent / ".shdm_key"
            if key_path.exists():
                encryption_key = key_path.read_bytes()
            else:
                encryption_key = Encryptor.generate_key()
                key_path.parent.mkdir(parents=True, exist_ok=True)
                key_path.write_bytes(encryption_key)
                os.chmod(key_path, 0o600)  # Owner read/write only
                logger.info(f"Generated new encryption key: {key_path}")
        
        # Initialize components
        self._store = SHDMStore(db_path, encryption_key)
        
        self._engine = ClassificationEngine(
            config=self.config,
            use_mock_model=use_mock_model,
        )
        
        self._session = SessionManager(
            self._store,
            detect_fn=self._detect_entities,
            max_tokens=max_context_tokens,
            keep_recent=keep_recent_messages,
        )
        
        logger.info(f"PrivacyClient initialized with db: {db_path}")
    
    def _detect_entities(self, text: str) -> List[Dict]:
        """Detection function for SessionManager.
        
        Converts Entity objects to dicts for SHDM.
        """
        entities = self._engine.detect(text, verify=False)
        
        return [
            {
                'start': e.start,
                'end': e.end,
                'text': e.text,
                'entity_type': e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type),
                'confidence': e.confidence,
            }
            for e in entities
        ]
    
    # =========================================================================
    # CONVERSATION MANAGEMENT
    # =========================================================================
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation.
        
        Returns conversation ID.
        """
        return self._session.create_conversation(title)
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self._session.get_conversation(conv_id)
    
    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """List recent conversations."""
        return self._session.list_conversations(limit)
    
    def delete_conversation(self, conv_id: str):
        """Delete a conversation and all its data."""
        self._session.delete_conversation(conv_id)
    
    # =========================================================================
    # MESSAGE PROCESSING
    # =========================================================================
    
    def process_input(
        self,
        conv_id: str,
        content: str,
        system_prompt: Optional[str] = None,
    ) -> ProcessedInput:
        """Process user input for sending to LLM.
        
        1. Detects PHI/PII in the input
        2. Tokenizes detected entities
        3. Stores message
        4. Builds context with history
        
        Args:
            conv_id: Conversation ID
            content: User's message (may contain PHI)
            system_prompt: System prompt for LLM
            
        Returns:
            ProcessedInput with tokenized text and context
        """
        # Detect entities
        entities = self._engine.detect(content, verify=False)
        
        # Convert to DetectedEntity format
        detected = [
            DetectedEntity(
                start=e.start,
                end=e.end,
                text=e.text,
                entity_type=e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type),
                confidence=e.confidence,
            )
            for e in entities
        ]
        
        # Process through session manager
        redacted, context = self._session.process_user_message(
            conv_id,
            content,
            system_prompt=system_prompt,
            entities=detected,
        )
        
        # Get tokens created for this message
        all_tokens = self._session.get_token_mappings(conv_id)
        tokens_created = {}
        for entity in detected:
            for token, original in all_tokens.items():
                if original.lower() == entity.text.lower():
                    tokens_created[entity.text] = token
                    break
        
        return ProcessedInput(
            redacted=redacted,
            context=context,
            entities=entities,
            tokens_created=tokens_created,
            conversation_id=conv_id,
        )
    
    def process_response(
        self,
        conv_id: str,
        llm_response: str,
    ) -> PrivacyResponse:
        """Process LLM response for showing to user.
        
        1. Restores tokens to original PHI values
        2. Stores message
        
        Args:
            conv_id: Conversation ID
            llm_response: Response from LLM (may contain tokens)
            
        Returns:
            PrivacyResponse with restored content
        """
        # Restore through session manager
        restored = self._session.process_assistant_message(conv_id, llm_response)
        
        # Get all tokens for reference
        all_tokens = self._session.get_token_mappings(conv_id)
        
        # Find which tokens were in the response
        import re
        tokens_used = {}
        for token in re.findall(r'\[[A-Z_]+_\d+\]', llm_response):
            if token in all_tokens:
                tokens_used[token] = all_tokens[token]
        
        return PrivacyResponse(
            content=restored,
            content_redacted=llm_response,
            tokens_used=tokens_used,
            conversation_id=conv_id,
        )
    
    # =========================================================================
    # COMPACTION
    # =========================================================================
    
    def compact_if_needed(
        self,
        conv_id: str,
        llm_fn: Callable[[List[dict]], str],
    ) -> Optional[str]:
        """Compact conversation if context is too large.
        
        Args:
            conv_id: Conversation ID
            llm_fn: Function that calls LLM with messages, returns response
            
        Returns:
            Summary if compaction happened, None otherwise
        """
        return self._session.auto_compact_if_needed(conv_id, llm_fn)
    
    def needs_compaction(self, conv_id: str) -> bool:
        """Check if conversation needs compaction."""
        return self._session.needs_compaction(conv_id)
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def detect(self, text: str, verify: bool = False) -> List[Entity]:
        """Detect PHI/PII without storing (for preview/testing).
        
        Args:
            text: Text to scan
            verify: Run LLM verification on uncertain entities
            
        Returns:
            List of detected entities
        """
        return self._engine.detect(text, verify=verify)
    
    def tokenize(self, conv_id: str, text: str) -> str:
        """Tokenize text without storing as a message.
        
        Useful for processing additional content like file attachments.
        """
        return self._session.tokenize_text(conv_id, text)
    
    def restore(self, conv_id: str, text: str) -> str:
        """Restore tokens in text without storing as a message."""
        return self._session.restore_text(conv_id, text)
    
    def get_context(
        self,
        conv_id: str,
        system_prompt: Optional[str] = None,
    ) -> LLMContext:
        """Get current context without processing a new message.
        
        Useful for re-generating context or debugging.
        """
        return self._session.get_context(conv_id, system_prompt)
    
    def get_messages(
        self,
        conv_id: str,
        for_display: bool = True,
    ) -> List[dict]:
        """Get message history.
        
        Args:
            conv_id: Conversation ID
            for_display: If True, returns user-visible content (PHI restored).
                        If False, returns tokenized content.
        """
        return self._session.get_messages(conv_id, for_display=for_display)
    
    def get_token_mappings(self, conv_id: str) -> Dict[str, str]:
        """Get all token mappings for a conversation.
        
        Returns dict mapping token -> original value.
        """
        return self._session.get_token_mappings(conv_id)
    
    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================
    
    def get_stack_status(self) -> Dict[str, Any]:
        """Get status of detection components."""
        return self._engine.get_stack_status()
    
    def get_stats(self, conv_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation."""
        conv = self._session.get_conversation(conv_id)
        if not conv:
            return {}
        
        messages = self._session.get_messages(conv_id, include_compacted=True)
        tokens = self._session.get_token_mappings(conv_id)
        context = self._session.get_context(conv_id)
        
        return {
            "conversation_id": conv_id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": len(messages),
            "token_count": len(tokens),
            "context_tokens": context.estimate_tokens(),
            "has_summary": conv.active_summary is not None,
        }


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

_default_client: Optional[PrivacyClient] = None


def get_privacy_client(**kwargs) -> PrivacyClient:
    """Get or create the default privacy client.
    
    Keyword arguments are passed to PrivacyClient() on first call.
    """
    global _default_client
    
    if _default_client is None:
        _default_client = PrivacyClient(**kwargs)
    
    return _default_client


def reset_privacy_client():
    """Reset the default client (for testing)."""
    global _default_client
    _default_client = None
