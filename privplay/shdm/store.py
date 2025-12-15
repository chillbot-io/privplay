"""SHDM Store - Encrypted storage for token mappings and conversations.

This module provides the persistence layer for the Safe Harbor De-Identification
Module, storing encrypted PHI-to-token mappings and conversation history.

Security model:
- Original PHI values are AES-256-GCM encrypted before storage
- A SHA-256 hash of the original enables consistent token lookup without decryption
- Encryption key is derived from user/session credentials (not stored here)
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Conversation:
    """A conversation with compaction support."""
    id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    active_summary: Optional[str] = None
    token_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "active_summary": self.active_summary,
            "token_count": self.token_count,
        }


@dataclass
class Message:
    """A single message in a conversation."""
    id: str
    conversation_id: str
    role: str  # user, assistant, system
    content: str  # Original content (what user sees)
    content_redacted: str  # Tokenized content (what LLM sees)
    created_at: datetime
    compacted: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "content_redacted": self.content_redacted,
            "created_at": self.created_at.isoformat(),
            "compacted": self.compacted,
        }


@dataclass
class TokenMapping:
    """A mapping from token to encrypted original value."""
    id: str
    conversation_id: str
    token: str  # e.g., [PATIENT_1]
    entity_type: str  # e.g., NAME_PATIENT
    original_hash: str  # SHA-256 for lookup
    created_at: datetime
    # Note: original_encrypted is handled separately (bytes)


# =============================================================================
# ENCRYPTION HELPERS
# =============================================================================

class Encryptor:
    """AES-256-GCM encryption for PHI values."""
    
    def __init__(self, key: bytes):
        """Initialize with a 32-byte key."""
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self._aesgcm = AESGCM(key)
    
    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt a string, returns nonce + ciphertext."""
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        return nonce + ciphertext
    
    def decrypt(self, data: bytes) -> str:
        """Decrypt nonce + ciphertext back to string."""
        nonce = data[:12]
        ciphertext = data[12:]
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode('utf-8')
    
    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derive a 32-byte key from password using PBKDF2."""
        import hashlib
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 600000, dklen=32)
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a random 32-byte key."""
        return os.urandom(32)


def hash_for_lookup(text: str) -> str:
    """Create a consistent hash for token lookup.
    
    Normalizes text (lowercase, stripped) before hashing to ensure
    "John Smith" and "john smith" map to the same token.
    """
    normalized = text.lower().strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


# =============================================================================
# SHDM STORE
# =============================================================================

class SHDMStore:
    """Persistent storage for SHDM token mappings and conversations.
    
    Usage:
        key = Encryptor.generate_key()  # Or derive from user password
        store = SHDMStore("/path/to/shdm.db", key)
        
        conv_id = store.create_conversation()
        token = store.get_or_create_token(conv_id, "John Smith", "NAME_PATIENT")
        # Returns "[PATIENT_1]"
        
        original = store.lookup_token(conv_id, "[PATIENT_1]")
        # Returns "John Smith"
    """
    
    def __init__(self, db_path: str, encryption_key: bytes):
        """Initialize the store.
        
        Args:
            db_path: Path to SQLite database file
            encryption_key: 32-byte key for AES-256-GCM encryption
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._encryptor = Encryptor(encryption_key)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                -- Conversations
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    title TEXT,
                    active_summary TEXT,
                    token_count INTEGER DEFAULT 0
                );
                
                -- Messages
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_redacted TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    compacted BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv 
                    ON messages(conversation_id, created_at);
                
                -- Token mappings
                CREATE TABLE IF NOT EXISTS token_mappings (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    original_encrypted BLOB NOT NULL,
                    original_hash TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                    UNIQUE(conversation_id, token)
                );
                CREATE INDEX IF NOT EXISTS idx_tokens_conv 
                    ON token_mappings(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_tokens_hash 
                    ON token_mappings(conversation_id, original_hash);
                CREATE INDEX IF NOT EXISTS idx_tokens_lookup
                    ON token_mappings(conversation_id, token);
                
                -- Token counters (for generating [TYPE_N] tokens)
                CREATE TABLE IF NOT EXISTS token_counters (
                    conversation_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    counter INTEGER DEFAULT 0,
                    PRIMARY KEY (conversation_id, entity_type)
                );
            """)
    
    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    # =========================================================================
    # CONVERSATION MANAGEMENT
    # =========================================================================
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation, returns conversation ID."""
        conv_id = secrets.token_hex(8)
        now = datetime.utcnow()
        
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO conversations (id, created_at, updated_at, title)
                   VALUES (?, ?, ?, ?)""",
                (conv_id, now, now, title)
            )
        
        return conv_id
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            
            if row:
                return Conversation(
                    id=row["id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    title=row["title"],
                    active_summary=row["active_summary"],
                    token_count=row["token_count"],
                )
        return None
    
    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """List recent conversations."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM conversations 
                   ORDER BY updated_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            
            return [
                Conversation(
                    id=row["id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    title=row["title"],
                    active_summary=row["active_summary"],
                    token_count=row["token_count"],
                )
                for row in rows
            ]
    
    def update_conversation(
        self, 
        conv_id: str, 
        title: Optional[str] = None,
        active_summary: Optional[str] = None,
        token_count: Optional[int] = None,
    ):
        """Update conversation metadata."""
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if active_summary is not None:
            updates.append("active_summary = ?")
            params.append(active_summary)
        if token_count is not None:
            updates.append("token_count = ?")
            params.append(token_count)
        
        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.utcnow())
            params.append(conv_id)
            
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
                    params
                )
    
    def delete_conversation(self, conv_id: str):
        """Delete a conversation and all associated data."""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM token_mappings WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM token_counters WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    
    # =========================================================================
    # MESSAGE MANAGEMENT
    # =========================================================================
    
    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        content_redacted: str,
    ) -> str:
        """Add a message to a conversation, returns message ID."""
        msg_id = secrets.token_hex(8)
        now = datetime.utcnow()
        
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO messages 
                   (id, conversation_id, role, content, content_redacted, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (msg_id, conv_id, role, content, content_redacted, now)
            )
            
            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id)
            )
        
        return msg_id
    
    def get_messages(
        self, 
        conv_id: str, 
        include_compacted: bool = False,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages for a conversation.
        
        Args:
            conv_id: Conversation ID
            include_compacted: Include messages that have been compacted
            limit: Max number of recent messages to return
        """
        query = "SELECT * FROM messages WHERE conversation_id = ?"
        params: List = [conv_id]
        
        if not include_compacted:
            query += " AND compacted = FALSE"
        
        query += " ORDER BY created_at ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            
            return [
                Message(
                    id=row["id"],
                    conversation_id=row["conversation_id"],
                    role=row["role"],
                    content=row["content"],
                    content_redacted=row["content_redacted"],
                    created_at=row["created_at"],
                    compacted=bool(row["compacted"]),
                )
                for row in rows
            ]
    
    def get_uncompacted_messages(self, conv_id: str) -> List[Message]:
        """Get all uncompacted messages for a conversation."""
        return self.get_messages(conv_id, include_compacted=False)
    
    def mark_messages_compacted(self, message_ids: List[str]):
        """Mark messages as compacted (summarized)."""
        if not message_ids:
            return
        
        placeholders = ",".join("?" * len(message_ids))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE messages SET compacted = TRUE WHERE id IN ({placeholders})",
                message_ids
            )
    
    def count_messages(self, conv_id: str, include_compacted: bool = False) -> int:
        """Count messages in a conversation."""
        query = "SELECT COUNT(*) FROM messages WHERE conversation_id = ?"
        params: List = [conv_id]
        
        if not include_compacted:
            query += " AND compacted = FALSE"
        
        with self._connect() as conn:
            return conn.execute(query, params).fetchone()[0]
    
    # =========================================================================
    # TOKEN MAPPING (SHDM CORE)
    # =========================================================================
    
    def get_or_create_token(
        self, 
        conv_id: str, 
        original_text: str, 
        entity_type: str,
    ) -> str:
        """Get existing token or create new one for the given text.
        
        This is the core SHDM function. Given original PHI text, it:
        1. Checks if we've seen this text before in this conversation
        2. If yes, returns the existing token
        3. If no, creates a new token, encrypts the original, stores mapping
        
        Args:
            conv_id: Conversation ID
            original_text: The PHI text to tokenize (e.g., "John Smith")
            entity_type: The entity type (e.g., "NAME_PATIENT")
            
        Returns:
            Token string like "[PATIENT_1]"
        """
        text_hash = hash_for_lookup(original_text)
        
        with self._connect() as conn:
            # Check for existing mapping
            row = conn.execute(
                """SELECT token FROM token_mappings 
                   WHERE conversation_id = ? AND original_hash = ?""",
                (conv_id, text_hash)
            ).fetchone()
            
            if row:
                return row["token"]
            
            # Get next counter for this entity type
            counter_row = conn.execute(
                """SELECT counter FROM token_counters 
                   WHERE conversation_id = ? AND entity_type = ?""",
                (conv_id, entity_type)
            ).fetchone()
            
            if counter_row:
                counter = counter_row["counter"] + 1
                conn.execute(
                    """UPDATE token_counters SET counter = ? 
                       WHERE conversation_id = ? AND entity_type = ?""",
                    (counter, conv_id, entity_type)
                )
            else:
                counter = 1
                conn.execute(
                    """INSERT INTO token_counters (conversation_id, entity_type, counter)
                       VALUES (?, ?, ?)""",
                    (conv_id, entity_type, counter)
                )
            
            # Create token
            # Simplify entity type for token: NAME_PATIENT -> PATIENT
            short_type = self._shorten_type(entity_type)
            token = f"[{short_type}_{counter}]"
            
            # Encrypt original
            encrypted = self._encryptor.encrypt(original_text)
            
            # Store mapping
            mapping_id = secrets.token_hex(8)
            conn.execute(
                """INSERT INTO token_mappings 
                   (id, conversation_id, token, entity_type, original_encrypted, original_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (mapping_id, conv_id, token, entity_type, encrypted, text_hash, datetime.utcnow())
            )
            
            return token
    
    def lookup_token(self, conv_id: str, token: str) -> Optional[str]:
        """Look up the original value for a token.
        
        Args:
            conv_id: Conversation ID
            token: Token like "[PATIENT_1]"
            
        Returns:
            Original text like "John Smith", or None if not found
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT original_encrypted FROM token_mappings 
                   WHERE conversation_id = ? AND token = ?""",
                (conv_id, token)
            ).fetchone()
            
            if row:
                return self._encryptor.decrypt(row["original_encrypted"])
        
        return None
    
    def get_all_tokens(self, conv_id: str) -> Dict[str, str]:
        """Get all token mappings for a conversation.
        
        Returns:
            Dict mapping token -> original text
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT token, original_encrypted FROM token_mappings WHERE conversation_id = ?",
                (conv_id,)
            ).fetchall()
            
            return {
                row["token"]: self._encryptor.decrypt(row["original_encrypted"])
                for row in rows
            }
    
    def get_token_count(self, conv_id: str) -> int:
        """Count tokens in a conversation."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM token_mappings WHERE conversation_id = ?",
                (conv_id,)
            ).fetchone()[0]
    
    def _shorten_type(self, entity_type: str) -> str:
        """Shorten entity type for readable tokens.
        
        NAME_PATIENT -> PATIENT
        NAME_PROVIDER -> PROVIDER
        DATE_DOB -> DOB
        SSN -> SSN
        """
        # Remove common prefixes
        for prefix in ["NAME_", "DATE_", "ID_"]:
            if entity_type.startswith(prefix):
                return entity_type[len(prefix):]
        return entity_type
    
    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================
    
    def tokenize_all(
        self, 
        conv_id: str, 
        entities: List[Tuple[str, str]],
    ) -> Dict[str, str]:
        """Tokenize multiple entities at once.
        
        Args:
            conv_id: Conversation ID
            entities: List of (text, entity_type) tuples
            
        Returns:
            Dict mapping original text -> token
        """
        result = {}
        for text, entity_type in entities:
            token = self.get_or_create_token(conv_id, text, entity_type)
            result[text] = token
        return result
    
    def restore_all(self, conv_id: str, tokens: List[str]) -> Dict[str, str]:
        """Restore multiple tokens at once.
        
        Args:
            conv_id: Conversation ID
            tokens: List of tokens like ["[PATIENT_1]", "[SSN_1]"]
            
        Returns:
            Dict mapping token -> original text
        """
        result = {}
        for token in tokens:
            original = self.lookup_token(conv_id, token)
            if original:
                result[token] = original
        return result
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def cleanup_old_conversations(self, days: int = 30):
        """Delete conversations older than N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self._connect() as conn:
            # Get old conversation IDs
            rows = conn.execute(
                "SELECT id FROM conversations WHERE updated_at < ?",
                (cutoff,)
            ).fetchall()
            
            for row in rows:
                self.delete_conversation(row["id"])
            
            return len(rows)


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================

_default_store: Optional[SHDMStore] = None


def get_shdm_store(db_path: Optional[str] = None, key: Optional[bytes] = None) -> SHDMStore:
    """Get or create the default SHDM store.
    
    On first call, db_path and key must be provided.
    Subsequent calls return the cached instance.
    """
    global _default_store
    
    if _default_store is None:
        if db_path is None or key is None:
            raise ValueError("db_path and key required on first call")
        _default_store = SHDMStore(db_path, key)
    
    return _default_store


def reset_shdm_store():
    """Reset the default store (for testing)."""
    global _default_store
    _default_store = None
