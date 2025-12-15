"""Signal Store for Continuous Learning.

Two-tier storage:
- Active DB: Fast queries, 90-day rolling window for review queue
- Archive DB: Cold storage, all signals forever, compacted

Signals flow:
    Detection → Active (if <0.95 conf) → Review Queue
                                      → Human Feedback
                                      → Labeled Signal
                                      → Training Corpus
    
    After 90 days: Active → Archive (compacted)
"""

import sqlite3
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATACLASSES
# =============================================================================

class FeedbackType(str, Enum):
    """Human feedback classifications."""
    TRUE_POSITIVE = "tp"      # Correct detection, correct label
    WRONG_LABEL = "wl"        # Correct detection, wrong label
    FALSE_POSITIVE = "fp"     # Not PHI at all
    FALSE_NEGATIVE = "fn"     # Missed PHI
    SKIPPED = "skip"          # No decision made


class SignalStatus(str, Enum):
    """Signal review status."""
    PENDING = "pending"       # Awaiting review
    REVIEWED = "reviewed"     # Human reviewed
    EXPIRED = "expired"       # Moved to archive without review
    TRAINING = "training"     # Used in training run


@dataclass
class CapturedSignal:
    """A captured detection signal with all features."""
    id: str
    created_at: datetime
    
    # Source info
    document_id: str
    doc_type: str
    conversation_id: Optional[str] = None
    
    # Span info
    span_start: int = 0
    span_end: int = 0
    span_text: str = ""
    
    # Detector signals
    phi_bert_detected: bool = False
    phi_bert_conf: float = 0.0
    phi_bert_type: str = ""
    
    pii_bert_detected: bool = False
    pii_bert_conf: float = 0.0
    pii_bert_type: str = ""
    
    presidio_detected: bool = False
    presidio_conf: float = 0.0
    presidio_type: str = ""
    
    rule_detected: bool = False
    rule_conf: float = 0.0
    rule_type: str = ""
    rule_has_checksum: bool = False
    
    # Coreference signals
    in_coref_cluster: bool = False
    coref_cluster_size: int = 0
    coref_anchor_conf: float = 0.0
    coref_is_anchor: bool = False
    coref_is_pronoun: bool = False
    coref_anchor_type: Optional[str] = None
    
    # Document-level features
    doc_has_ssn: bool = False
    doc_has_dates: bool = False
    doc_has_medical_terms: bool = False
    doc_entity_count: int = 0
    doc_pii_density: float = 0.0
    
    # Computed features
    sources_agree_count: int = 0
    span_length: int = 0
    has_digits: bool = False
    has_letters: bool = False
    all_caps: bool = False
    all_digits: bool = False
    mixed_case: bool = False
    
    # Merged output
    merged_type: str = ""
    merged_conf: float = 0.0
    merged_source: str = ""
    
    # Model info
    model_version: str = ""
    
    # Review status
    status: str = SignalStatus.PENDING.value
    
    # Feedback (filled after review)
    feedback_type: Optional[str] = None
    feedback_correct_type: Optional[str] = None  # For WRONG_LABEL
    feedback_at: Optional[datetime] = None
    feedback_by: Optional[str] = None  # User/session ID
    
    # Ground truth (for training)
    ground_truth_type: Optional[str] = None
    ground_truth_source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        # Convert datetimes to ISO strings
        if d['created_at']:
            d['created_at'] = d['created_at'].isoformat()
        if d['feedback_at']:
            d['feedback_at'] = d['feedback_at'].isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CapturedSignal':
        """Create from dictionary."""
        # Convert ISO strings to datetimes
        if d.get('created_at') and isinstance(d['created_at'], str):
            d['created_at'] = datetime.fromisoformat(d['created_at'])
        if d.get('feedback_at') and isinstance(d['feedback_at'], str):
            d['feedback_at'] = datetime.fromisoformat(d['feedback_at'])
        return cls(**d)
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for training."""
        return [
            # Detector signals
            int(self.phi_bert_detected),
            self.phi_bert_conf,
            int(self.pii_bert_detected),
            self.pii_bert_conf,
            int(self.presidio_detected),
            self.presidio_conf,
            int(self.rule_detected),
            self.rule_conf,
            int(self.rule_has_checksum),
            
            # Coreference signals
            int(self.in_coref_cluster),
            self.coref_cluster_size,
            self.coref_anchor_conf,
            int(self.coref_is_anchor),
            int(self.coref_is_pronoun),
            int(self.coref_anchor_type is not None),
            
            # Document-level features
            int(self.doc_has_ssn),
            int(self.doc_has_dates),
            int(self.doc_has_medical_terms),
            self.doc_entity_count,
            self.doc_pii_density,
            
            # Computed span features
            self.sources_agree_count,
            self.span_length,
            int(self.has_digits),
            int(self.has_letters),
            int(self.all_caps),
            int(self.all_digits),
            int(self.mixed_case),
        ]


@dataclass
class FalseNegativeReport:
    """User-reported missed PHI."""
    id: str
    created_at: datetime
    document_id: str
    conversation_id: Optional[str]
    
    # What was missed
    missed_text: str
    missed_start: Optional[int]  # If we can locate it
    missed_end: Optional[int]
    correct_type: str  # What it should have been labeled as
    
    # Context
    context_before: str = ""
    context_after: str = ""
    
    # Status
    status: str = "pending"  # pending, incorporated, rejected
    incorporated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d['created_at']:
            d['created_at'] = d['created_at'].isoformat()
        if d['incorporated_at']:
            d['incorporated_at'] = d['incorporated_at'].isoformat()
        return d


@dataclass 
class ModelVersion:
    """Trained model metadata."""
    id: str
    created_at: datetime
    
    # Training info
    training_signals: int
    adversarial_signals: int
    synthetic_signals: int
    
    # Performance
    f1_score: float
    precision: float
    recall: float
    
    # Status
    status: str = "staged"  # staged, shadow, active, rolled_back
    promoted_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    rollback_reason: Optional[str] = None
    
    # File location
    model_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d['created_at']:
            d['created_at'] = d['created_at'].isoformat()
        if d['promoted_at']:
            d['promoted_at'] = d['promoted_at'].isoformat()
        if d['rolled_back_at']:
            d['rolled_back_at'] = d['rolled_back_at'].isoformat()
        return d


# =============================================================================
# ACTIVE SIGNAL STORE
# =============================================================================

ACTIVE_SCHEMA = """
-- Captured signals awaiting or completed review
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    
    -- Source
    document_id TEXT NOT NULL,
    doc_type TEXT,
    conversation_id TEXT,
    
    -- Span
    span_start INTEGER,
    span_end INTEGER,
    span_text TEXT,
    
    -- All features as JSON (flexible, avoids schema migrations)
    features TEXT NOT NULL,
    
    -- Merged output
    merged_type TEXT,
    merged_conf REAL,
    
    -- Model that produced this
    model_version TEXT,
    
    -- Review status
    status TEXT DEFAULT 'pending',
    
    -- Feedback
    feedback_type TEXT,
    feedback_correct_type TEXT,
    feedback_at TEXT,
    feedback_by TEXT,
    
    -- Ground truth (derived from feedback)
    ground_truth_type TEXT,
    ground_truth_source TEXT
);

-- False negative reports
CREATE TABLE IF NOT EXISTS false_negatives (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    document_id TEXT NOT NULL,
    conversation_id TEXT,
    
    missed_text TEXT NOT NULL,
    missed_start INTEGER,
    missed_end INTEGER,
    correct_type TEXT NOT NULL,
    
    context_before TEXT,
    context_after TEXT,
    
    status TEXT DEFAULT 'pending',
    incorporated_at TEXT
);

-- Model versions
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    
    training_signals INTEGER,
    adversarial_signals INTEGER,
    synthetic_signals INTEGER,
    
    f1_score REAL,
    precision_score REAL,
    recall_score REAL,
    
    status TEXT DEFAULT 'staged',
    promoted_at TEXT,
    rolled_back_at TEXT,
    rollback_reason TEXT,
    
    model_path TEXT
);

-- Training runs
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    
    signals_used INTEGER,
    model_id TEXT,
    
    status TEXT DEFAULT 'running',
    error TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_conf ON signals(merged_conf);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_doc ON signals(document_id);
CREATE INDEX IF NOT EXISTS idx_fn_status ON false_negatives(status);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
"""


class SignalStore:
    """Active signal store for continuous learning."""
    
    CONFIDENCE_THRESHOLD = 0.95  # Below this → review queue
    ARCHIVE_AFTER_DAYS = 90
    
    def __init__(self, db_path: str, archive_path: Optional[str] = None):
        """Initialize signal store.
        
        Args:
            db_path: Path to active SQLite database
            archive_path: Path to archive database (defaults to db_path + '.archive')
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if archive_path:
            self.archive_path = Path(archive_path)
        else:
            self.archive_path = self.db_path.with_suffix('.archive.db')
        
        self._init_db()
        self._init_archive()
    
    def _init_db(self):
        """Initialize active database schema."""
        with self._connect() as conn:
            conn.executescript(ACTIVE_SCHEMA)
    
    def _init_archive(self):
        """Initialize archive database schema."""
        with self._connect_archive() as conn:
            conn.executescript(ACTIVE_SCHEMA)  # Same schema, different db
    
    @contextmanager
    def _connect(self):
        """Context manager for active database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    @contextmanager
    def _connect_archive(self):
        """Context manager for archive database."""
        conn = sqlite3.connect(self.archive_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    # =========================================================================
    # SIGNAL CAPTURE
    # =========================================================================
    
    def capture_signal(self, signal: CapturedSignal) -> str:
        """Capture a detection signal.
        
        Only stores if confidence < threshold (needs review).
        Returns signal ID.
        """
        # Only capture uncertain signals
        if signal.merged_conf >= self.CONFIDENCE_THRESHOLD:
            logger.debug(f"Signal {signal.id} above threshold, not capturing")
            return signal.id
        
        features = signal.to_dict()
        
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO signals (
                    id, created_at, document_id, doc_type, conversation_id,
                    span_start, span_end, span_text, features,
                    merged_type, merged_conf, model_version, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.id,
                signal.created_at.isoformat(),
                signal.document_id,
                signal.doc_type,
                signal.conversation_id,
                signal.span_start,
                signal.span_end,
                signal.span_text,
                json.dumps(features),
                signal.merged_type,
                signal.merged_conf,
                signal.model_version,
                SignalStatus.PENDING.value,
            ))
        
        logger.debug(f"Captured signal {signal.id} (conf={signal.merged_conf:.2f})")
        return signal.id
    
    def capture_signals_batch(self, signals: List[CapturedSignal]) -> int:
        """Capture multiple signals. Returns count captured."""
        captured = 0
        for signal in signals:
            if signal.merged_conf < self.CONFIDENCE_THRESHOLD:
                self.capture_signal(signal)
                captured += 1
        return captured
    
    # =========================================================================
    # REVIEW QUEUE
    # =========================================================================
    
    def get_pending_count(self) -> int:
        """Get count of signals awaiting review."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE status = ?",
                (SignalStatus.PENDING.value,)
            ).fetchone()
            return row[0]
    
    def get_pending_signals(
        self, 
        limit: int = 50,
        oldest_first: bool = True,
    ) -> List[CapturedSignal]:
        """Get signals awaiting review."""
        order = "ASC" if oldest_first else "DESC"
        
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT features FROM signals 
                WHERE status = ?
                ORDER BY created_at {order}
                LIMIT ?
            """, (SignalStatus.PENDING.value, limit)).fetchall()
            
            return [
                CapturedSignal.from_dict(json.loads(row['features']))
                for row in rows
            ]
    
    def get_signal(self, signal_id: str) -> Optional[CapturedSignal]:
        """Get a specific signal."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT features FROM signals WHERE id = ?",
                (signal_id,)
            ).fetchone()
            
            if row:
                return CapturedSignal.from_dict(json.loads(row['features']))
        return None
    
    # =========================================================================
    # FEEDBACK
    # =========================================================================
    
    def record_feedback(
        self,
        signal_id: str,
        feedback_type: FeedbackType,
        correct_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Record human feedback on a signal.
        
        Args:
            signal_id: Signal to update
            feedback_type: TP, WL, FP, FN, or SKIP
            correct_type: For WRONG_LABEL, the correct entity type
            user_id: Who provided feedback
            
        Returns:
            True if updated, False if signal not found
        """
        now = datetime.utcnow()
        
        # Determine ground truth from feedback
        ground_truth_type = None
        ground_truth_source = "human_feedback"
        
        if feedback_type == FeedbackType.TRUE_POSITIVE:
            # Get merged_type from signal
            signal = self.get_signal(signal_id)
            if signal:
                ground_truth_type = signal.merged_type
        elif feedback_type == FeedbackType.WRONG_LABEL:
            ground_truth_type = correct_type
        elif feedback_type == FeedbackType.FALSE_POSITIVE:
            ground_truth_type = "NONE"
        # FN handled separately via report_false_negative
        
        with self._connect() as conn:
            result = conn.execute("""
                UPDATE signals SET
                    status = ?,
                    feedback_type = ?,
                    feedback_correct_type = ?,
                    feedback_at = ?,
                    feedback_by = ?,
                    ground_truth_type = ?,
                    ground_truth_source = ?
                WHERE id = ?
            """, (
                SignalStatus.REVIEWED.value,
                feedback_type.value,
                correct_type,
                now.isoformat(),
                user_id,
                ground_truth_type,
                ground_truth_source,
                signal_id,
            ))
            
            return result.rowcount > 0
    
    def report_false_negative(
        self,
        document_id: str,
        missed_text: str,
        correct_type: str,
        conversation_id: Optional[str] = None,
        context_before: str = "",
        context_after: str = "",
        missed_start: Optional[int] = None,
        missed_end: Optional[int] = None,
    ) -> str:
        """Report a missed PHI entity (false negative).
        
        Returns the report ID.
        """
        report_id = secrets.token_hex(8)
        now = datetime.utcnow()
        
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO false_negatives (
                    id, created_at, document_id, conversation_id,
                    missed_text, missed_start, missed_end, correct_type,
                    context_before, context_after, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                now.isoformat(),
                document_id,
                conversation_id,
                missed_text,
                missed_start,
                missed_end,
                correct_type,
                context_before,
                context_after,
                "pending",
            ))
        
        logger.info(f"Recorded false negative report {report_id}: '{missed_text}' as {correct_type}")
        return report_id
    
    def get_pending_false_negatives(self, limit: int = 100) -> List[FalseNegativeReport]:
        """Get unincorporated false negative reports."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM false_negatives
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
            """, (limit,)).fetchall()
            
            reports = []
            for row in rows:
                reports.append(FalseNegativeReport(
                    id=row['id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    document_id=row['document_id'],
                    conversation_id=row['conversation_id'],
                    missed_text=row['missed_text'],
                    missed_start=row['missed_start'],
                    missed_end=row['missed_end'],
                    correct_type=row['correct_type'],
                    context_before=row['context_before'] or "",
                    context_after=row['context_after'] or "",
                    status=row['status'],
                ))
            return reports
    
    # =========================================================================
    # TRAINING DATA
    # =========================================================================
    
    def get_labeled_signals(self, limit: Optional[int] = None) -> List[CapturedSignal]:
        """Get all reviewed signals with ground truth for training."""
        query = """
            SELECT features FROM signals
            WHERE status = ? AND ground_truth_type IS NOT NULL
            ORDER BY feedback_at DESC
        """
        params: List[Any] = [SignalStatus.REVIEWED.value]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            
            return [
                CapturedSignal.from_dict(json.loads(row['features']))
                for row in rows
            ]
    
    def get_labeled_signal_count(self) -> int:
        """Get count of labeled signals ready for training."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) FROM signals
                WHERE status = ? AND ground_truth_type IS NOT NULL
            """, (SignalStatus.REVIEWED.value,)).fetchone()
            return row[0]
    
    def mark_signals_used_for_training(self, signal_ids: List[str], run_id: str):
        """Mark signals as used in a training run."""
        with self._connect() as conn:
            placeholders = ",".join("?" * len(signal_ids))
            conn.execute(f"""
                UPDATE signals SET status = ?
                WHERE id IN ({placeholders})
            """, [SignalStatus.TRAINING.value] + signal_ids)
    
    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================
    
    def register_model(self, model: ModelVersion) -> str:
        """Register a new trained model."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO models (
                    id, created_at, training_signals, adversarial_signals,
                    synthetic_signals, f1_score, precision_score, recall_score,
                    status, model_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.id,
                model.created_at.isoformat(),
                model.training_signals,
                model.adversarial_signals,
                model.synthetic_signals,
                model.f1_score,
                model.precision,
                model.recall,
                model.status,
                model.model_path,
            ))
        
        logger.info(f"Registered model {model.id} (F1={model.f1_score:.3f})")
        return model.id
    
    def get_active_model(self) -> Optional[ModelVersion]:
        """Get the currently active model."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM models WHERE status = 'active'
                ORDER BY promoted_at DESC LIMIT 1
            """).fetchone()
            
            if row:
                return self._row_to_model(row)
        return None
    
    def get_model_history(self, limit: int = 10) -> List[ModelVersion]:
        """Get recent model versions."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM models
                ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
            
            return [self._row_to_model(row) for row in rows]
    
    def promote_model(self, model_id: str) -> bool:
        """Promote a staged model to active.
        
        Demotes current active model to rolled_back.
        """
        now = datetime.utcnow()
        
        with self._connect() as conn:
            # Demote current active
            conn.execute("""
                UPDATE models SET status = 'rolled_back', rolled_back_at = ?
                WHERE status = 'active'
            """, (now.isoformat(),))
            
            # Promote new model
            result = conn.execute("""
                UPDATE models SET status = 'active', promoted_at = ?
                WHERE id = ?
            """, (now.isoformat(), model_id))
            
            return result.rowcount > 0
    
    def rollback_model(self, reason: str = "manual") -> Optional[str]:
        """Rollback to previous model.
        
        Returns the ID of the newly active model, or None if no rollback available.
        """
        now = datetime.utcnow()
        
        with self._connect() as conn:
            # Get current and previous models
            rows = conn.execute("""
                SELECT id, status FROM models
                WHERE status IN ('active', 'rolled_back')
                ORDER BY 
                    CASE status WHEN 'active' THEN 0 ELSE 1 END,
                    promoted_at DESC
                LIMIT 2
            """).fetchall()
            
            if len(rows) < 2:
                logger.warning("No previous model to rollback to")
                return None
            
            current_id = rows[0]['id']
            previous_id = rows[1]['id']
            
            # Demote current
            conn.execute("""
                UPDATE models SET 
                    status = 'rolled_back',
                    rolled_back_at = ?,
                    rollback_reason = ?
                WHERE id = ?
            """, (now.isoformat(), reason, current_id))
            
            # Promote previous
            conn.execute("""
                UPDATE models SET status = 'active', promoted_at = ?
                WHERE id = ?
            """, (now.isoformat(), previous_id))
            
            logger.info(f"Rolled back from {current_id} to {previous_id}: {reason}")
            return previous_id
    
    def _row_to_model(self, row) -> ModelVersion:
        """Convert database row to ModelVersion."""
        return ModelVersion(
            id=row['id'],
            created_at=datetime.fromisoformat(row['created_at']),
            training_signals=row['training_signals'],
            adversarial_signals=row['adversarial_signals'],
            synthetic_signals=row['synthetic_signals'],
            f1_score=row['f1_score'],
            precision=row['precision_score'],
            recall=row['recall_score'],
            status=row['status'],
            promoted_at=datetime.fromisoformat(row['promoted_at']) if row['promoted_at'] else None,
            rolled_back_at=datetime.fromisoformat(row['rolled_back_at']) if row['rolled_back_at'] else None,
            rollback_reason=row['rollback_reason'],
            model_path=row['model_path'],
        )
    
    # =========================================================================
    # ARCHIVAL
    # =========================================================================
    
    def archive_old_signals(self) -> int:
        """Move signals older than ARCHIVE_AFTER_DAYS to archive.
        
        Returns count of archived signals.
        """
        cutoff = datetime.utcnow() - timedelta(days=self.ARCHIVE_AFTER_DAYS)
        cutoff_str = cutoff.isoformat()
        
        with self._connect() as conn:
            # Get old signals
            rows = conn.execute("""
                SELECT * FROM signals
                WHERE created_at < ? AND status != ?
            """, (cutoff_str, SignalStatus.TRAINING.value)).fetchall()
            
            if not rows:
                return 0
            
            # Insert into archive
            with self._connect_archive() as archive_conn:
                for row in rows:
                    # Update status to expired if still pending
                    status = row['status']
                    if status == SignalStatus.PENDING.value:
                        status = SignalStatus.EXPIRED.value
                    
                    archive_conn.execute("""
                        INSERT OR REPLACE INTO signals (
                            id, created_at, document_id, doc_type, conversation_id,
                            span_start, span_end, span_text, features,
                            merged_type, merged_conf, model_version, status,
                            feedback_type, feedback_correct_type, feedback_at,
                            feedback_by, ground_truth_type, ground_truth_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['id'], row['created_at'], row['document_id'],
                        row['doc_type'], row['conversation_id'],
                        row['span_start'], row['span_end'], row['span_text'],
                        row['features'], row['merged_type'], row['merged_conf'],
                        row['model_version'], status,
                        row['feedback_type'], row['feedback_correct_type'],
                        row['feedback_at'], row['feedback_by'],
                        row['ground_truth_type'], row['ground_truth_source'],
                    ))
            
            # Delete from active
            signal_ids = [row['id'] for row in rows]
            placeholders = ",".join("?" * len(signal_ids))
            conn.execute(f"DELETE FROM signals WHERE id IN ({placeholders})", signal_ids)
            
            logger.info(f"Archived {len(rows)} signals older than {self.ARCHIVE_AFTER_DAYS} days")
            return len(rows)
    
    # =========================================================================
    # STATS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get signal store statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE status = ?",
                (SignalStatus.PENDING.value,)
            ).fetchone()[0]
            reviewed = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE status = ?",
                (SignalStatus.REVIEWED.value,)
            ).fetchone()[0]
            
            fn_pending = conn.execute(
                "SELECT COUNT(*) FROM false_negatives WHERE status = 'pending'"
            ).fetchone()[0]
            
            active_model = self.get_active_model()
        
        with self._connect_archive() as conn:
            archived = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        
        return {
            "total_active": total,
            "pending_review": pending,
            "reviewed": reviewed,
            "false_negatives_pending": fn_pending,
            "archived": archived,
            "active_model": active_model.id if active_model else None,
            "active_model_f1": active_model.f1_score if active_model else None,
        }


# =============================================================================
# CONVENIENCE
# =============================================================================

_signal_store: Optional[SignalStore] = None


def get_signal_store(db_path: Optional[str] = None) -> SignalStore:
    """Get or create global signal store instance."""
    global _signal_store
    if _signal_store is None:
        if db_path is None:
            from ..config import get_config
            db_path = str(get_config().data_dir / "signals.db")
        _signal_store = SignalStore(db_path)
    return _signal_store


def reset_signal_store():
    """Reset global signal store (for testing)."""
    global _signal_store
    _signal_store = None
