"""SHDM Session Manager - Orchestrates conversation flow with compaction.

Manages the full lifecycle of a conversation:
- Token mappings for consistent PHI redaction
- Message history with compaction
- Context building for LLM requests
"""

import re
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

from .store import SHDMStore, Conversation, Message
from .tokenizer import Tokenizer, Restorer, DetectedEntity, entities_from_detection


@dataclass
class LLMContext:
    """Context to send to the LLM."""
    system_prompt: Optional[str]
    summary: Optional[str]  # Compacted history
    messages: List[dict]  # Recent messages [{role, content}]
    
    def to_messages(self) -> List[dict]:
        """Convert to LLM message format."""
        result = []
        
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        
        if self.summary:
            result.append({
                "role": "system", 
                "content": f"Previous conversation summary:\n{self.summary}"
            })
        
        result.extend(self.messages)
        return result
    
    def estimate_tokens(self) -> int:
        """Rough token estimate (chars / 4)."""
        total = 0
        if self.system_prompt:
            total += len(self.system_prompt)
        if self.summary:
            total += len(self.summary)
        for msg in self.messages:
            total += len(msg.get("content", ""))
        return total // 4


class SessionManager:
    """Manages conversation sessions with SHDM integration.
    
    This is the main interface for the chat client. It handles:
    - Creating and managing conversations
    - Tokenizing outbound messages (PHI -> tokens)
    - Building context for LLM (with compaction)
    - Restoring inbound responses (tokens -> PHI)
    
    Usage:
        # Initialize
        store = SHDMStore(db_path, key)
        session = SessionManager(store, detect_fn=my_detection_function)
        
        # Start conversation
        conv_id = session.create_conversation()
        
        # User sends message
        redacted, context = session.process_user_message(
            conv_id, 
            "What about John Smith's test results?",
            system_prompt="You are a helpful medical assistant."
        )
        
        # Send context.to_messages() to LLM...
        llm_response = call_llm(context.to_messages())
        
        # Process LLM response
        restored = session.process_assistant_message(conv_id, llm_response)
        
        # Show `restored` to user (PHI restored)
    """
    
    # Default compaction settings
    DEFAULT_MAX_TOKENS = 8000  # Trigger compaction at this threshold
    DEFAULT_KEEP_RECENT = 10   # Keep this many recent messages uncompacted
    
    def __init__(
        self, 
        store: SHDMStore,
        detect_fn: Optional[Callable[[str], List]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        keep_recent: int = DEFAULT_KEEP_RECENT,
    ):
        """Initialize session manager.
        
        Args:
            store: SHDMStore instance
            detect_fn: Function to detect PHI in text, returns list of entities.
                       If None, no automatic detection (must provide entities manually).
            max_tokens: Trigger compaction when context exceeds this
            keep_recent: Number of recent messages to keep uncompacted
        """
        self.store = store
        self.detect_fn = detect_fn
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        
        self._tokenizer = Tokenizer(store)
        self._restorer = Restorer(store)
    
    # =========================================================================
    # CONVERSATION LIFECYCLE
    # =========================================================================
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation."""
        return self.store.create_conversation(title)
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self.store.get_conversation(conv_id)
    
    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """List recent conversations."""
        return self.store.list_conversations(limit)
    
    def delete_conversation(self, conv_id: str):
        """Delete a conversation and all its data."""
        self.store.delete_conversation(conv_id)
    
    # =========================================================================
    # MESSAGE PROCESSING
    # =========================================================================
    
    def process_user_message(
        self,
        conv_id: str,
        content: str,
        system_prompt: Optional[str] = None,
        entities: Optional[List[DetectedEntity]] = None,
    ) -> Tuple[str, LLMContext]:
        """Process a user message and build LLM context.
        
        1. Detect PHI (if detect_fn provided and entities not given)
        2. Tokenize PHI -> tokens
        3. Store message
        4. Check for compaction need
        5. Build context for LLM
        
        Args:
            conv_id: Conversation ID
            content: User's message (may contain PHI)
            system_prompt: System prompt for LLM
            entities: Pre-detected entities (if None, uses detect_fn)
            
        Returns:
            Tuple of (redacted_message, context_for_llm)
        """
        # Detect PHI if not provided
        if entities is None and self.detect_fn:
            detection_results = self.detect_fn(content)
            entities = entities_from_detection(detection_results)
        
        # Tokenize
        if entities:
            redacted = self._tokenizer.tokenize(conv_id, content, entities)
        else:
            redacted = content
        
        # Store message
        self.store.add_message(conv_id, "user", content, redacted)
        
        # Build context
        context = self._build_context(conv_id, system_prompt)
        
        return redacted, context
    
    def process_assistant_message(
        self,
        conv_id: str,
        content: str,
    ) -> str:
        """Process an assistant (LLM) response.
        
        1. Restore tokens -> PHI
        2. Store message (both versions)
        
        Args:
            conv_id: Conversation ID
            content: LLM response (may contain tokens)
            
        Returns:
            Restored message with PHI for user display
        """
        # Restore tokens to PHI
        restored = self._restorer.restore(conv_id, content)
        
        # Store message (content = what user sees, redacted = what LLM said)
        self.store.add_message(conv_id, "assistant", restored, content)
        
        return restored
    
    def tokenize_text(
        self,
        conv_id: str,
        text: str,
        entities: Optional[List[DetectedEntity]] = None,
    ) -> str:
        """Tokenize text without storing as a message.
        
        Useful for processing system prompts or other text.
        """
        if entities is None and self.detect_fn:
            detection_results = self.detect_fn(text)
            entities = entities_from_detection(detection_results)
        
        if entities:
            return self._tokenizer.tokenize(conv_id, text, entities)
        return text
    
    def restore_text(self, conv_id: str, text: str) -> str:
        """Restore tokens in text without storing as a message."""
        return self._restorer.restore(conv_id, text)
    
    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================
    
    def _build_context(
        self, 
        conv_id: str, 
        system_prompt: Optional[str] = None,
    ) -> LLMContext:
        """Build context for LLM request.
        
        Combines:
        - System prompt
        - Compacted summary (if any)
        - Recent uncompacted messages
        """
        conv = self.store.get_conversation(conv_id)
        messages = self.store.get_uncompacted_messages(conv_id)
        
        # Convert to LLM format (use redacted content)
        llm_messages = [
            {"role": msg.role, "content": msg.content_redacted}
            for msg in messages
        ]
        
        return LLMContext(
            system_prompt=system_prompt,
            summary=conv.active_summary if conv else None,
            messages=llm_messages,
        )
    
    def get_context(
        self,
        conv_id: str,
        system_prompt: Optional[str] = None,
    ) -> LLMContext:
        """Get current context without processing a new message."""
        return self._build_context(conv_id, system_prompt)
    
    # =========================================================================
    # COMPACTION
    # =========================================================================
    
    def needs_compaction(self, conv_id: str) -> bool:
        """Check if conversation needs compaction."""
        context = self._build_context(conv_id)
        return context.estimate_tokens() > self.max_tokens
    
    def compact(
        self,
        conv_id: str,
        llm_fn: Callable[[List[dict]], str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Compact old messages into a summary.
        
        Args:
            conv_id: Conversation ID
            llm_fn: Function that takes messages and returns LLM response
            system_prompt: Optional system prompt for summarization
            
        Returns:
            The generated summary
        """
        messages = self.store.get_uncompacted_messages(conv_id)
        
        if len(messages) <= self.keep_recent:
            return ""  # Nothing to compact
        
        # Split: old messages to compact, recent to keep
        to_compact = messages[:-self.keep_recent]
        
        # Build summarization prompt
        conv_text = "\n".join([
            f"{msg.role.upper()}: {msg.content_redacted}"
            for msg in to_compact
        ])
        
        summarize_prompt = system_prompt or """You are summarizing a conversation for context continuity.
Preserve ALL important information including:
- Patient names, conditions, medications
- Decisions made
- Action items
- Key dates and numbers
Be concise but complete. Use the same entity tokens (like [PATIENT_1]) that appear in the conversation."""
        
        summarize_messages = [
            {"role": "system", "content": summarize_prompt},
            {"role": "user", "content": f"Summarize this conversation:\n\n{conv_text}"}
        ]
        
        # Get summary from LLM
        summary = llm_fn(summarize_messages)
        
        # Get existing summary and append
        conv = self.store.get_conversation(conv_id)
        if conv and conv.active_summary:
            combined_summary = f"{conv.active_summary}\n\n---\n\n{summary}"
        else:
            combined_summary = summary
        
        # Update conversation with new summary
        self.store.update_conversation(conv_id, active_summary=combined_summary)
        
        # Mark old messages as compacted
        message_ids = [msg.id for msg in to_compact]
        self.store.mark_messages_compacted(message_ids)
        
        return summary
    
    def auto_compact_if_needed(
        self,
        conv_id: str,
        llm_fn: Callable[[List[dict]], str],
    ) -> Optional[str]:
        """Compact if context exceeds threshold.
        
        Returns summary if compaction happened, None otherwise.
        """
        if self.needs_compaction(conv_id):
            return self.compact(conv_id, llm_fn)
        return None
    
    # =========================================================================
    # TOKEN ACCESS
    # =========================================================================
    
    def get_token_mappings(self, conv_id: str) -> dict:
        """Get all token mappings for a conversation.
        
        Returns dict mapping token -> original value.
        """
        return self.store.get_all_tokens(conv_id)
    
    def lookup_token(self, conv_id: str, token: str) -> Optional[str]:
        """Look up a single token."""
        return self.store.lookup_token(conv_id, token)
    
    # =========================================================================
    # MESSAGE HISTORY
    # =========================================================================
    
    def get_messages(
        self, 
        conv_id: str, 
        include_compacted: bool = False,
        for_display: bool = True,
    ) -> List[dict]:
        """Get messages for display or export.
        
        Args:
            conv_id: Conversation ID
            include_compacted: Include compacted messages
            for_display: If True, returns user-visible content. 
                        If False, returns redacted (tokenized) content.
        
        Returns:
            List of {role, content, timestamp} dicts
        """
        messages = self.store.get_messages(conv_id, include_compacted)
        
        return [
            {
                "role": msg.role,
                "content": msg.content if for_display else msg.content_redacted,
                "timestamp": msg.created_at.isoformat(),
            }
            for msg in messages
        ]
    
    def get_full_history(self, conv_id: str) -> List[Message]:
        """Get full message history including compacted messages."""
        return self.store.get_messages(conv_id, include_compacted=True)
