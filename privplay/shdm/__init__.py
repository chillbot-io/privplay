"""SHDM - Safe Harbor De-Identification Module.

Provides reversible tokenization for PHI/PII, enabling safe LLM usage
while preserving the ability to restore original values in responses.

Components:
- store: Encrypted SQLite storage for token mappings and conversations
- tokenizer: Text tokenization (PHI -> tokens)  
- restorer: Token restoration (tokens -> PHI)
- session: Conversation and context management

Usage:
    from privplay.shdm import SessionManager, SHDMStore, Encryptor
    
    # Initialize with encryption key
    key = Encryptor.generate_key()
    store = SHDMStore("~/.privplay/shdm.db", key)
    
    # Create session manager with detection function
    session = SessionManager(store, detect_fn=engine.detect)
    
    # Create conversation
    conv_id = session.create_conversation()
    
    # Process user message (auto-detects and tokenizes PHI)
    redacted, context = session.process_user_message(
        conv_id, 
        "What about John Smith's lab results?",
        system_prompt="You are a medical assistant."
    )
    
    # Send to LLM
    response = llm.chat(context.to_messages())
    
    # Process response (restores tokens to PHI)
    restored = session.process_assistant_message(conv_id, response)
    
    # Show `restored` to user - they see "John Smith", LLM only saw "[PATIENT_1]"
"""

from .store import (
    SHDMStore,
    Encryptor,
    Conversation,
    Message,
    TokenMapping,
    get_shdm_store,
    reset_shdm_store,
    hash_for_lookup,
)

from .tokenizer import (
    Tokenizer,
    Restorer,
    DetectedEntity,
    entities_from_detection,
)

from .session import (
    SessionManager,
    LLMContext,
)

__all__ = [
    # Store
    "SHDMStore",
    "Encryptor",
    "Conversation", 
    "Message",
    "TokenMapping",
    "get_shdm_store",
    "reset_shdm_store",
    "hash_for_lookup",
    # Tokenizer
    "Tokenizer",
    "Restorer",
    "DetectedEntity",
    "entities_from_detection",
    # Session
    "SessionManager",
    "LLMContext",
]
