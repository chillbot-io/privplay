"""Privplay SDK - Healthcare AI Privacy Infrastructure.

PHI/PII detection and de-identification for safe LLM usage in healthcare.

Quick Start:
    import privplay.sdk as privplay
    
    # Simple one-liner
    safe_text = await privplay.redact("Patient John Smith, SSN 123-45-6789")
    
    # With more control
    result = await privplay.scan(text)
    for entity in result.entities:
        print(f"{entity.entity_type}: {entity.text} ({entity.confidence:.0%})")
    
    # Multi-turn conversations
    async with privplay.Session(password="secret") as session:
        result = await session.redact("Dr. Smith prescribed Lisinopril")
        # Send result.safe_text to LLM...
        restored = await session.restore(llm_response)

Sync Usage:
    safe_text = privplay.sync_redact("Patient John Smith...")
    result = privplay.sync_scan(text)
"""

__version__ = "0.1.0"

import asyncio
import secrets
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Use existing types - no duplication
from .types import Entity, EntityType, SourceType
from .config import Config, get_config, set_config

# =============================================================================
# EXCEPTIONS
# =============================================================================

class PrivplayError(Exception):
    """Base exception for all Privplay errors."""
    pass

class ConfigurationError(PrivplayError):
    """Configuration is invalid or missing."""
    pass

class ModelNotFoundError(PrivplayError):
    """A required model is not available."""
    pass

class SessionError(PrivplayError):
    """Session-related error."""
    pass

class SessionExpiredError(SessionError):
    """Session has expired."""
    pass

class DetectionError(PrivplayError):
    """Error during PHI/PII detection."""
    pass


# =============================================================================
# RESULT TYPES (thin wrappers around existing Entity)
# =============================================================================

@dataclass
class ScanResult:
    """Result of scanning text for PHI/PII."""
    text: str
    entities: List[Entity]
    
    @property
    def has_phi(self) -> bool:
        return len(self.entities) > 0
    
    @property
    def entity_count(self) -> int:
        return len(self.entities)
    
    def redact(self, session_id: Optional[str] = None) -> "RedactResult":
        """Convert scan to redacted text."""
        return _redact_from_scan(self, session_id)
    
    def __iter__(self):
        return iter(self.entities)
    
    def __len__(self):
        return len(self.entities)


@dataclass
class RedactResult:
    """Result of redacting PHI/PII from text."""
    original: str
    safe_text: str
    entities: List[Entity]
    token_map: Dict[str, str]
    _session_id: Optional[str] = field(default=None, repr=False)
    _store: Any = field(default=None, repr=False)
    
    @property
    def has_phi(self) -> bool:
        return len(self.entities) > 0
    
    def restore(self, text: str) -> str:
        """Restore tokens in text (e.g., LLM response)."""
        if self._store and self._session_id:
            return _restore_with_store(text, self._session_id, self._store)
        return _restore_with_map(text, self.token_map)


@dataclass
class StackStatus:
    """Status of detection components."""
    phi_bert: Dict[str, Any]
    pii_bert: Dict[str, Any]
    rules: Dict[str, Any]
    coreference: Dict[str, Any]
    verifier: Dict[str, Any]
    
    @property
    def all_available(self) -> bool:
        return all([
            self.phi_bert.get("available", False),
            self.pii_bert.get("available", False),
            self.rules.get("available", True),
        ])


# =============================================================================
# ENGINE MANAGEMENT
# =============================================================================

_engine = None
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="privplay_")
    return _executor


def _get_engine():
    """Get or create the classification engine."""
    global _engine
    if _engine is None:
        from .engine.classifier import ClassificationEngine
        config = get_config()
        _engine = ClassificationEngine(
            config=config,
            use_coreference=True,
        )
    return _engine


def clear_engine():
    """Clear cached engine (for testing)."""
    global _engine
    _engine = None


# =============================================================================
# CORE API - ASYNC
# =============================================================================

async def scan(
    text: str,
    *,
    verify: bool = True,
    threshold: Optional[float] = None,
) -> ScanResult:
    """Scan text for PHI/PII entities.
    
    Args:
        text: Text to scan
        verify: Use LLM verification for uncertain entities
        threshold: Confidence threshold (default from config)
    
    Returns:
        ScanResult with detected entities
    """
    config = get_config()
    engine = _get_engine()
    
    loop = asyncio.get_event_loop()
    try:
        entities = await loop.run_in_executor(
            _get_executor(),
            lambda: engine.detect(
                text,
                verify=verify,
                threshold=threshold or config.confidence_threshold,
            ),
        )
    except Exception as e:
        raise DetectionError(f"Detection failed: {e}")
    
    return ScanResult(text=text, entities=entities)


async def redact(
    text: str,
    *,
    full: bool = False,
    verify: bool = True,
    threshold: Optional[float] = None,
) -> Union[str, RedactResult]:
    """Scan and redact PHI/PII in text.
    
    Args:
        text: Text to redact
        full: If True, return RedactResult. If False, return just safe_text.
        verify: Use LLM verification
        threshold: Confidence threshold
    
    Returns:
        str (safe_text) if full=False, RedactResult if full=True
    """
    scan_result = await scan(text, verify=verify, threshold=threshold)
    
    if not scan_result.entities:
        if full:
            return RedactResult(
                original=text,
                safe_text=text,
                entities=[],
                token_map={},
            )
        return text
    
    result = _redact_from_scan(scan_result)
    
    if full:
        return result
    return result.safe_text


async def restore(text: str, token_map: Dict[str, str]) -> str:
    """Restore tokens in text using token map.
    
    Args:
        text: Text containing tokens
        token_map: Mapping from original text to token
    
    Returns:
        Text with tokens restored
    """
    return _restore_with_map(text, token_map)


async def scan_batch(
    texts: List[str],
    *,
    verify: bool = True,
    threshold: Optional[float] = None,
    max_concurrent: int = 4,
) -> List[ScanResult]:
    """Batch scan multiple texts."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scan_one(t: str) -> ScanResult:
        async with semaphore:
            return await scan(t, verify=verify, threshold=threshold)
    
    return await asyncio.gather(*[scan_one(t) for t in texts])


async def scan_stream(
    text: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 100,
    verify: bool = False,
    threshold: Optional[float] = None,
) -> AsyncIterator[ScanResult]:
    """Stream detection for large documents."""
    if len(text) <= chunk_size:
        yield await scan(text, verify=verify, threshold=threshold)
        return
    
    position = 0
    while position < len(text):
        end = min(position + chunk_size, len(text))
        chunk = text[position:end]
        
        result = await scan(chunk, verify=verify, threshold=threshold)
        
        # Adjust positions to full document
        for entity in result.entities:
            entity.start += position
            entity.end += position
        
        yield result
        position = end - overlap if end < len(text) else end


def get_stack_status() -> StackStatus:
    """Get status of all detection components."""
    engine = _get_engine()
    raw = engine.get_stack_status()
    return StackStatus(
        phi_bert=raw.get("phi_bert", {}),
        pii_bert=raw.get("pii_bert", {}),
        rules=raw.get("rules", {}),
        coreference=raw.get("coreference", {}),
        verifier=raw.get("verifier", {}),
    )


# =============================================================================
# SYNC WRAPPERS
# =============================================================================

def _run_sync(coro):
    """Run coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def sync_scan(text: str, **kwargs) -> ScanResult:
    """Sync version of scan()."""
    return _run_sync(scan(text, **kwargs))


def sync_redact(text: str, **kwargs) -> Union[str, RedactResult]:
    """Sync version of redact()."""
    return _run_sync(redact(text, **kwargs))


def sync_restore(text: str, token_map: Dict[str, str]) -> str:
    """Sync version of restore()."""
    return _run_sync(restore(text, token_map))


# =============================================================================
# SESSION
# =============================================================================

# PBKDF2 iterations - OWASP 2024 recommendation
PBKDF2_ITERATIONS = 600_000


class Session:
    """Multi-turn conversation session with token consistency.
    
    Usage:
        async with Session(password="secret") as session:
            result = await session.redact("Patient John Smith...")
            # Send result.safe_text to LLM
            restored = await session.restore(llm_response)
    """
    
    def __init__(
        self,
        password: str,
        session_id: Optional[str] = None,
        persist: bool = True,
    ):
        self._password = password
        self._session_id = session_id or secrets.token_hex(16)
        self._persist = persist
        
        self._key: Optional[bytes] = None
        self._store = None
        self._engine = None
        self._initialized = False
        self._closed = False
        
        # In-memory fallback
        self._local_tokens: Dict[str, str] = {}
        self._local_counters: Dict[str, int] = {}
    
    @property
    def id(self) -> str:
        return self._session_id
    
    async def __aenter__(self) -> "Session":
        await self._initialize()
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    async def _initialize(self):
        if self._initialized:
            return
        
        # Derive key
        salt = self._session_id.encode()
        loop = asyncio.get_event_loop()
        self._key = await loop.run_in_executor(
            None,
            lambda: hashlib.pbkdf2_hmac('sha256', self._password.encode(), salt, PBKDF2_ITERATIONS, dklen=32)
        )
        
        if self._persist:
            await self._init_store()
        
        self._initialized = True
    
    async def _init_store(self):
        try:
            from .shdm.store import SHDMStore
            config = get_config()
            db_path = config.data_dir / "sessions" / f"{self._session_id}.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._store = SHDMStore(str(db_path), self._key)
            self._store.create_conversation(self._session_id)
        except Exception:
            self._persist = False
    
    async def scan(self, text: str, **kwargs) -> ScanResult:
        """Scan text for PHI/PII."""
        self._check_open()
        return await scan(text, **kwargs)
    
    async def redact(self, text: str, **kwargs) -> RedactResult:
        """Scan and redact, with session token consistency."""
        self._check_open()
        
        scan_result = await scan(text, **kwargs)
        if not scan_result.entities:
            return RedactResult(
                original=text,
                safe_text=text,
                entities=[],
                token_map={},
                _session_id=self._session_id,
                _store=self._store,
            )
        
        # Tokenize with session consistency
        safe_text, token_map = await self._tokenize(text, scan_result.entities)
        
        return RedactResult(
            original=text,
            safe_text=safe_text,
            entities=scan_result.entities,
            token_map=token_map,
            _session_id=self._session_id,
            _store=self._store,
        )
    
    async def restore(self, text: str) -> str:
        """Restore tokens to original values."""
        self._check_open()
        
        if self._store:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: _restore_from_store(text, self._session_id, self._store)
            )
        return _restore_with_map(text, {v: k for k, v in self._local_tokens.items()})
    
    async def _tokenize(self, text: str, entities: List[Entity]) -> tuple:
        """Tokenize entities with session consistency."""
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        token_map = {}
        result_text = text
        
        for entity in sorted_entities:
            token = await self._get_or_create_token(entity.text, entity.entity_type.value)
            
            actual = result_text[entity.start:entity.end]
            if actual == entity.text:
                result_text = result_text[:entity.start] + token + result_text[entity.end:]
            
            token_map[entity.text] = token
        
        return result_text, token_map
    
    async def _get_or_create_token(self, text: str, entity_type: str) -> str:
        """Get existing or create new token."""
        normalized = text.lower().strip()
        
        if self._store:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._store.get_or_create_token(self._session_id, text, entity_type)
            )
        
        # In-memory fallback
        if normalized in self._local_tokens:
            return self._local_tokens[normalized]
        
        short_type = _shorten_type(entity_type)
        self._local_counters[short_type] = self._local_counters.get(short_type, 0) + 1
        token = f"[{short_type}_{self._local_counters[short_type]}]"
        self._local_tokens[normalized] = token
        return token
    
    def _check_open(self):
        if self._closed:
            raise SessionExpiredError("Session closed")
        if not self._initialized:
            raise SessionError("Session not initialized - use 'async with'")
    
    async def close(self):
        self._key = None
        self._password = ""
        self._closed = True


class SyncSession:
    """Synchronous session wrapper."""
    
    def __init__(self, password: str, **kwargs):
        self._async = Session(password, **kwargs)
    
    def __enter__(self):
        _run_sync(self._async._initialize())
        return self
    
    def __exit__(self, *args):
        _run_sync(self._async.close())
    
    @property
    def id(self):
        return self._async.id
    
    def scan(self, text: str, **kwargs) -> ScanResult:
        return _run_sync(self._async.scan(text, **kwargs))
    
    def redact(self, text: str, **kwargs) -> RedactResult:
        return _run_sync(self._async.redact(text, **kwargs))
    
    def restore(self, text: str) -> str:
        return _run_sync(self._async.restore(text))


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

TOKEN_PATTERN = re.compile(r'\[([A-Z_]+)_(\d+)\]')


def _shorten_type(entity_type: str) -> str:
    """NAME_PATIENT -> PATIENT"""
    for prefix in ["NAME_", "DATE_", "ID_"]:
        if entity_type.startswith(prefix):
            return entity_type[len(prefix):]
    return entity_type


def _redact_from_scan(scan_result: ScanResult, session_id: Optional[str] = None) -> RedactResult:
    """Convert ScanResult to RedactResult."""
    if not scan_result.entities:
        return RedactResult(
            original=scan_result.text,
            safe_text=scan_result.text,
            entities=[],
            token_map={},
        )
    
    session_id = session_id or secrets.token_hex(8)
    sorted_entities = sorted(scan_result.entities, key=lambda e: e.start, reverse=True)
    
    type_counters: Dict[str, int] = {}
    token_map: Dict[str, str] = {}
    text_to_token: Dict[str, str] = {}
    result_text = scan_result.text
    
    for entity in sorted_entities:
        normalized = entity.text.lower().strip()
        
        if normalized in text_to_token:
            token = text_to_token[normalized]
        else:
            short_type = _shorten_type(entity.entity_type.value)
            type_counters[short_type] = type_counters.get(short_type, 0) + 1
            token = f"[{short_type}_{type_counters[short_type]}]"
            text_to_token[normalized] = token
        
        actual = result_text[entity.start:entity.end]
        if actual == entity.text:
            result_text = result_text[:entity.start] + token + result_text[entity.end:]
        
        token_map[entity.text] = token
    
    return RedactResult(
        original=scan_result.text,
        safe_text=result_text,
        entities=scan_result.entities,
        token_map=token_map,
        _session_id=session_id,
    )


def _restore_with_map(text: str, token_map: Dict[str, str]) -> str:
    """Restore using token map."""
    reverse = {v: k for k, v in token_map.items()}
    result = text
    for token in sorted(reverse.keys(), key=len, reverse=True):
        result = result.replace(token, reverse[token])
    return result


def _restore_with_store(text: str, session_id: str, store) -> str:
    """Restore using SHDM store."""
    def replace(match):
        token = match.group(0)
        original = store.lookup_token(session_id, token)
        return original if original else token
    return TOKEN_PATTERN.sub(replace, text)


def _restore_from_store(text: str, session_id: str, store) -> str:
    """Alias for _restore_with_store."""
    return _restore_with_store(text, session_id, store)


# =============================================================================
# CONFIGURE HELPER
# =============================================================================

def configure(**kwargs) -> Config:
    """Configure Privplay globally.
    
    Args:
        confidence_threshold: Detection threshold
        data_dir: Path to data directory
        **kwargs: Other config options
    
    Returns:
        Updated Config
    """
    config = get_config()
    
    if "confidence_threshold" in kwargs:
        config.confidence_threshold = kwargs["confidence_threshold"]
    if "data_dir" in kwargs:
        from pathlib import Path
        config.data_dir = Path(kwargs["data_dir"])
    
    set_config(config)
    clear_engine()  # Reset engine to pick up new config
    return config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Core async API
    "scan",
    "redact",
    "restore",
    "scan_batch",
    "scan_stream",
    "get_stack_status",
    
    # Sync API
    "sync_scan",
    "sync_redact",
    "sync_restore",
    
    # Session
    "Session",
    "SyncSession",
    
    # Configuration
    "configure",
    
    # Result types
    "ScanResult",
    "RedactResult",
    "StackStatus",
    
    # Re-export core types
    "Entity",
    "EntityType",
    "SourceType",
    "Config",
    "get_config",
    
    # Errors
    "PrivplayError",
    "ConfigurationError",
    "ModelNotFoundError",
    "SessionError",
    "SessionExpiredError",
    "DetectionError",
    
    # Utils
    "clear_engine",
]
