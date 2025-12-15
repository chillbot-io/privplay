"""Audit Logging - Tamper-evident logging for HIPAA compliance.

Provides:
- Immutable audit log entries
- Hash chain for tamper detection
- Structured event logging
- Export capabilities

HIPAA requires maintaining audit logs of PHI access for 6 years.

Usage:
    from privplay.audit import AuditLogger, AuditEvent
    
    logger = AuditLogger("~/.privplay/audit.db")
    
    # Log an event
    logger.log(AuditEvent(
        event_type="PHI_DETECTED",
        user_id="user_123",
        conversation_id="conv_456",
        details={"entity_count": 5, "entity_types": ["SSN", "NAME"]},
    ))
    
    # Query logs
    events = logger.query(
        start_date=datetime(2024, 1, 1),
        event_type="PHI_DETECTED"
    )
    
    # Verify integrity
    is_valid = logger.verify_chain()
    
    # Export for compliance
    logger.export_csv("audit_2024.csv", year=2024)
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """Standard audit event types."""
    # PHI Detection & Processing
    PHI_DETECTED = "PHI_DETECTED"
    PHI_REDACTED = "PHI_REDACTED"
    PHI_RESTORED = "PHI_RESTORED"
    
    # Conversation Events
    CONVERSATION_CREATED = "CONVERSATION_CREATED"
    CONVERSATION_DELETED = "CONVERSATION_DELETED"
    CONVERSATION_EXPORTED = "CONVERSATION_EXPORTED"
    
    # Access Events
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    DATA_ACCESSED = "DATA_ACCESSED"
    DATA_EXPORTED = "DATA_EXPORTED"
    
    # System Events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    KEY_ROTATED = "KEY_ROTATED"
    
    # Compliance Events
    AUDIT_QUERIED = "AUDIT_QUERIED"
    AUDIT_EXPORTED = "AUDIT_EXPORTED"
    INTEGRITY_CHECK = "INTEGRITY_CHECK"
    
    # Error Events
    ERROR = "ERROR"
    SECURITY_ALERT = "SECURITY_ALERT"


# =============================================================================
# AUDIT EVENT
# =============================================================================

@dataclass
class AuditEvent:
    """A single audit log entry."""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Set by AuditLogger
    id: Optional[str] = None
    sequence: Optional[int] = None
    prev_hash: Optional[str] = None
    hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'sequence': self.sequence,
            'prev_hash': self.prev_hash,
            'hash': self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary."""
        return cls(
            id=data.get('id'),
            event_type=data.get('event_type', ''),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None,
            user_id=data.get('user_id'),
            conversation_id=data.get('conversation_id'),
            details=data.get('details', {}),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            sequence=data.get('sequence'),
            prev_hash=data.get('prev_hash'),
            hash=data.get('hash'),
        )


# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """Tamper-evident audit logger with hash chain.
    
    Each log entry includes a hash of the previous entry, creating an
    immutable chain. Any tampering breaks the chain and is detectable.
    
    Storage: SQLite database with append-only semantics.
    """
    
    GENESIS_HASH = "0" * 64  # SHA-256 of nothing
    
    def __init__(self, db_path: str):
        """Initialize audit logger.
        
        Args:
            db_path: Path to SQLite audit database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                -- Audit log entries (append-only)
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    sequence INTEGER UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    conversation_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                );
                
                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                    ON audit_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_type 
                    ON audit_log(event_type);
                CREATE INDEX IF NOT EXISTS idx_audit_user 
                    ON audit_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_conversation 
                    ON audit_log(conversation_id);
                
                -- Sequence tracker
                CREATE TABLE IF NOT EXISTS audit_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
    
    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _get_last_hash(self) -> str:
        """Get hash of the last log entry."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT hash FROM audit_log ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            return row['hash'] if row else self.GENESIS_HASH
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(sequence) as max_seq FROM audit_log"
            ).fetchone()
            return (row['max_seq'] or 0) + 1
    
    def _compute_hash(self, event: AuditEvent) -> str:
        """Compute SHA-256 hash for an event."""
        # Create deterministic string representation
        hash_input = json.dumps({
            'id': event.id,
            'sequence': event.sequence,
            'event_type': event.event_type,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'conversation_id': event.conversation_id,
            'details': event.details,
            'prev_hash': event.prev_hash,
        }, sort_keys=True)
        
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    def log(self, event: AuditEvent) -> AuditEvent:
        """Log an audit event.
        
        Args:
            event: The event to log
            
        Returns:
            The event with id, sequence, and hashes filled in
        """
        # Generate ID if not set
        if not event.id:
            event.id = str(uuid.uuid4())
        
        # Set timestamp if not set
        if not event.timestamp:
            event.timestamp = datetime.utcnow()
        
        # Get chain info
        event.prev_hash = self._get_last_hash()
        event.sequence = self._get_next_sequence()
        
        # Compute hash
        event.hash = self._compute_hash(event)
        
        # Insert into database
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO audit_log 
                (id, sequence, event_type, timestamp, user_id, conversation_id, 
                 details, ip_address, user_agent, prev_hash, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.sequence,
                event.event_type,
                event.timestamp.isoformat(),
                event.user_id,
                event.conversation_id,
                json.dumps(event.details),
                event.ip_address,
                event.user_agent,
                event.prev_hash,
                event.hash,
            ))
        
        return event
    
    def log_simple(
        self,
        event_type: str,
        user_id: str = None,
        conversation_id: str = None,
        **details,
    ) -> AuditEvent:
        """Convenience method to log an event."""
        return self.log(AuditEvent(
            event_type=event_type,
            user_id=user_id,
            conversation_id=conversation_id,
            details=details,
        ))
    
    # =========================================================================
    # QUERYING
    # =========================================================================
    
    def query(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        event_type: str = None,
        user_id: str = None,
        conversation_id: str = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Query audit log entries.
        
        Args:
            start_date: Filter events after this date
            end_date: Filter events before this date
            event_type: Filter by event type
            user_id: Filter by user
            conversation_id: Filter by conversation
            limit: Maximum results
            offset: Skip this many results
            
        Returns:
            List of matching AuditEvent objects
        """
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())
        
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if conversation_id:
            conditions.append("conversation_id = ?")
            params.append(conversation_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT * FROM audit_log 
            WHERE {where_clause}
            ORDER BY sequence DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        events = []
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            for row in rows:
                events.append(AuditEvent(
                    id=row['id'],
                    sequence=row['sequence'],
                    event_type=row['event_type'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_id=row['user_id'],
                    conversation_id=row['conversation_id'],
                    details=json.loads(row['details']) if row['details'] else {},
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    prev_hash=row['prev_hash'],
                    hash=row['hash'],
                ))
        
        return events
    
    def get_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific event by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM audit_log WHERE id = ?", (event_id,)
            ).fetchone()
            
            if row:
                return AuditEvent(
                    id=row['id'],
                    sequence=row['sequence'],
                    event_type=row['event_type'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_id=row['user_id'],
                    conversation_id=row['conversation_id'],
                    details=json.loads(row['details']) if row['details'] else {},
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    prev_hash=row['prev_hash'],
                    hash=row['hash'],
                )
        return None
    
    def count(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        event_type: str = None,
    ) -> int:
        """Count matching log entries."""
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())
        
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM audit_log WHERE {where_clause}",
                params
            ).fetchone()
            return row['cnt']
    
    # =========================================================================
    # INTEGRITY VERIFICATION
    # =========================================================================
    
    def verify_chain(self, start_seq: int = 1, end_seq: int = None) -> Dict[str, Any]:
        """Verify integrity of the hash chain.
        
        Checks that:
        1. Each entry's hash matches its computed hash
        2. Each entry's prev_hash matches the previous entry's hash
        3. No gaps in sequence numbers
        
        Returns:
            Dict with 'valid' bool and details
        """
        result = {
            'valid': True,
            'entries_checked': 0,
            'first_invalid_sequence': None,
            'error': None,
        }
        
        with self._connect() as conn:
            if end_seq:
                rows = conn.execute(
                    """SELECT * FROM audit_log 
                       WHERE sequence >= ? AND sequence <= ?
                       ORDER BY sequence ASC""",
                    (start_seq, end_seq)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE sequence >= ? ORDER BY sequence ASC",
                    (start_seq,)
                ).fetchall()
            
            prev_hash = self.GENESIS_HASH if start_seq == 1 else None
            prev_seq = start_seq - 1
            
            for row in rows:
                result['entries_checked'] += 1
                
                # Check sequence continuity
                if row['sequence'] != prev_seq + 1:
                    result['valid'] = False
                    result['first_invalid_sequence'] = row['sequence']
                    result['error'] = f"Sequence gap: expected {prev_seq + 1}, got {row['sequence']}"
                    break
                
                # Check prev_hash matches
                if prev_hash is not None and row['prev_hash'] != prev_hash:
                    result['valid'] = False
                    result['first_invalid_sequence'] = row['sequence']
                    result['error'] = f"prev_hash mismatch at sequence {row['sequence']}"
                    break
                
                # Reconstruct and verify hash
                event = AuditEvent(
                    id=row['id'],
                    sequence=row['sequence'],
                    event_type=row['event_type'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_id=row['user_id'],
                    conversation_id=row['conversation_id'],
                    details=json.loads(row['details']) if row['details'] else {},
                    prev_hash=row['prev_hash'],
                )
                computed_hash = self._compute_hash(event)
                
                if computed_hash != row['hash']:
                    result['valid'] = False
                    result['first_invalid_sequence'] = row['sequence']
                    result['error'] = f"Hash mismatch at sequence {row['sequence']}"
                    break
                
                prev_hash = row['hash']
                prev_seq = row['sequence']
        
        # Log the verification
        self.log_simple(
            EventType.INTEGRITY_CHECK.value,
            entries_checked=result['entries_checked'],
            valid=result['valid'],
            error=result.get('error'),
        )
        
        return result
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export_csv(
        self,
        output_path: str,
        start_date: datetime = None,
        end_date: datetime = None,
        event_type: str = None,
    ) -> int:
        """Export audit log to CSV.
        
        Returns number of records exported.
        """
        import csv
        
        events = self.query(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            limit=1000000,  # High limit for export
        )
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'id', 'sequence', 'event_type', 'timestamp',
                'user_id', 'conversation_id', 'details',
                'ip_address', 'user_agent', 'prev_hash', 'hash'
            ])
            
            for event in events:
                writer.writerow([
                    event.id,
                    event.sequence,
                    event.event_type,
                    event.timestamp.isoformat(),
                    event.user_id or '',
                    event.conversation_id or '',
                    json.dumps(event.details),
                    event.ip_address or '',
                    event.user_agent or '',
                    event.prev_hash,
                    event.hash,
                ])
        
        # Log the export
        self.log_simple(
            EventType.AUDIT_EXPORTED.value,
            records_exported=len(events),
            output_path=output_path,
        )
        
        return len(events)
    
    def export_json(
        self,
        output_path: str,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> int:
        """Export audit log to JSON."""
        events = self.query(
            start_date=start_date,
            end_date=end_date,
            limit=1000000,
        )
        
        data = {
            'exported_at': datetime.utcnow().isoformat(),
            'record_count': len(events),
            'events': [e.to_dict() for e in events],
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.log_simple(
            EventType.AUDIT_EXPORTED.value,
            records_exported=len(events),
            output_path=output_path,
            format='json',
        )
        
        return len(events)
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get audit log statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self._connect() as conn:
            # Total count
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            
            # Recent count
            recent = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE timestamp >= ?",
                (cutoff.isoformat(),)
            ).fetchone()[0]
            
            # By event type
            by_type = {}
            rows = conn.execute("""
                SELECT event_type, COUNT(*) as cnt 
                FROM audit_log 
                WHERE timestamp >= ?
                GROUP BY event_type
            """, (cutoff.isoformat(),)).fetchall()
            for row in rows:
                by_type[row['event_type']] = row['cnt']
            
            # First and last entry
            first = conn.execute(
                "SELECT timestamp FROM audit_log ORDER BY sequence ASC LIMIT 1"
            ).fetchone()
            last = conn.execute(
                "SELECT timestamp FROM audit_log ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        
        return {
            'total_entries': total,
            f'entries_last_{days}_days': recent,
            'by_event_type': by_type,
            'first_entry': first['timestamp'] if first else None,
            'last_entry': last['timestamp'] if last else None,
        }


# =============================================================================
# CONVENIENCE
# =============================================================================

_default_logger: Optional[AuditLogger] = None


def get_audit_logger(db_path: str = None) -> AuditLogger:
    """Get or create default audit logger."""
    global _default_logger
    if _default_logger is None:
        if db_path is None:
            raise ValueError("db_path required on first call")
        _default_logger = AuditLogger(db_path)
    return _default_logger


def audit(event_type: str, **details) -> AuditEvent:
    """Quick audit logging."""
    return get_audit_logger().log_simple(event_type, **details)
