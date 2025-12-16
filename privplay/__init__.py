"""Privplay - Healthcare AI Privacy Infrastructure.

PHI/PII detection and de-identification for safe LLM usage in healthcare.

Quick Start:
    from privplay import scan, redact, Session
    
    # Simple redaction
    safe_text = sync_redact("Patient John Smith, SSN 123-45-6789")
    
    # Async API
    result = await scan("Dr. Wilson at jane@hospital.org")
    safe = await redact(text)
    
    # Multi-turn sessions
    async with Session(password="secret") as session:
        result = await session.redact("Patient data here")
        restored = await session.restore(llm_response)
"""

__version__ = "0.1.0"

# Core async API
from .sdk import (
    scan,
    redact,
    restore,
    scan_batch,
    scan_stream,
)

# Sync API
from .sdk import (
    sync_scan,
    sync_redact,
    sync_restore,
)

# Sessions
from .sdk import (
    Session,
    SyncSession,
)

# Result types
from .sdk import (
    ScanResult,
    RedactResult,
)

# Core types (re-exported from types.py)
from .types import (
    Entity,
    EntityType,
    SourceType,
)

# Exceptions
from .sdk import (
    PrivplayError,
    ConfigurationError,
    ModelNotFoundError,
    SessionError,
    SessionExpiredError,
    DetectionError,
)

# Configuration
from .sdk import (
    configure,
    get_config,
    get_stack_status,
)

__all__ = [
    # Version
    "__version__",
    
    # Async API
    "scan",
    "redact", 
    "restore",
    "scan_batch",
    "scan_stream",
    
    # Sync API
    "sync_scan",
    "sync_redact",
    "sync_restore",
    
    # Sessions
    "Session",
    "SyncSession",
    
    # Result types
    "ScanResult",
    "RedactResult",
    
    # Core types
    "Entity",
    "EntityType",
    "SourceType",
    
    # Exceptions
    "PrivplayError",
    "ConfigurationError",
    "ModelNotFoundError",
    "SessionError",
    "SessionExpiredError",
    "DetectionError",
    
    # Configuration
    "configure",
    "get_config",
    "get_stack_status",
]
