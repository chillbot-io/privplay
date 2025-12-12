"""SQLite database layer for Privplay."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, List
import json
import uuid

from .types import (
    Document, Entity, Correction, EntityType, 
    DecisionType, SourceType, ReviewStats
)
from .config import get_config


SCHEMA = """
-- Documents to scan/review
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    scanned_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detected entities
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    text TEXT NOT NULL,
    start_idx INTEGER NOT NULL,
    end_idx INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    llm_confidence REAL,
    llm_reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Human review decisions
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    document_id TEXT NOT NULL REFERENCES documents(id),
    entity_text TEXT NOT NULL,
    entity_start INTEGER NOT NULL,
    entity_end INTEGER NOT NULL,
    detected_type TEXT NOT NULL,
    decision TEXT NOT NULL,
    correct_type TEXT,
    context_before TEXT,
    context_after TEXT,
    ner_confidence REAL,
    llm_confidence REAL,
    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learned patterns (promoted from corrections)
CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    pattern TEXT NOT NULL,
    entity_type TEXT,
    confidence_adjustment REAL,
    source_corrections TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entities_doc ON entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_corrections_entity ON corrections(entity_id);
CREATE INDEX IF NOT EXISTS idx_corrections_decision ON corrections(decision);
"""


class Database:
    """SQLite database wrapper."""
    
    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = get_config().db_path
        self.path = path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
    
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connection."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    # Document operations
    
    def add_document(self, doc: Document) -> str:
        """Add a document."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents (id, content, source, scanned_at, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc.id, doc.content, doc.source, doc.scanned_at, doc.created_at)
            )
        return doc.id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            
            if row:
                return Document(
                    id=row["id"],
                    content=row["content"],
                    source=row["source"],
                    scanned_at=row["scanned_at"],
                    created_at=row["created_at"],
                )
        return None
    
    def get_unscanned_documents(self) -> List[Document]:
        """Get documents that haven't been scanned."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE scanned_at IS NULL"
            ).fetchall()
            
            return [
                Document(
                    id=row["id"],
                    content=row["content"],
                    source=row["source"],
                    scanned_at=row["scanned_at"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
    
    def mark_document_scanned(self, doc_id: str):
        """Mark a document as scanned."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET scanned_at = ? WHERE id = ?",
                (datetime.utcnow(), doc_id)
            )
    
    def count_documents(self) -> int:
        """Count total documents."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    
    # Entity operations
    
    def add_entity(self, entity: Entity, document_id: str) -> str:
        """Add a detected entity."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO entities 
                   (id, document_id, text, start_idx, end_idx, entity_type, 
                    confidence, source, llm_confidence, llm_reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity.id, document_id, entity.text, entity.start, entity.end,
                    entity.entity_type.value, entity.confidence, entity.source.value,
                    entity.llm_confidence, entity.llm_reasoning
                )
            )
        return entity.id
    
    def add_entities(self, entities: List[Entity], document_id: str):
        """Add multiple entities."""
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO entities 
                   (id, document_id, text, start_idx, end_idx, entity_type, 
                    confidence, source, llm_confidence, llm_reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        e.id, document_id, e.text, e.start, e.end,
                        e.entity_type.value, e.confidence, e.source.value,
                        e.llm_confidence, e.llm_reasoning
                    )
                    for e in entities
                ]
            )
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            
            if row:
                return self._row_to_entity(row)
        return None
    
    def get_entities_for_document(self, doc_id: str) -> List[Entity]:
        """Get all entities for a document."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entities WHERE document_id = ? ORDER BY start_idx",
                (doc_id,)
            ).fetchall()
            
            return [self._row_to_entity(row) for row in rows]
    
    def get_entities_for_review(
        self, 
        threshold: float = 0.95,
        entity_type: Optional[EntityType] = None,
        limit: int = 100
    ) -> List[tuple[Entity, Document]]:
        """Get entities needing review (below threshold, not yet reviewed)."""
        with self._connect() as conn:
            query = """
                SELECT e.*, d.content, d.source as doc_source, d.id as doc_id
                FROM entities e
                JOIN documents d ON e.document_id = d.id
                LEFT JOIN corrections c ON e.id = c.entity_id
                WHERE e.confidence < ?
                AND c.id IS NULL
            """
            params = [threshold]
            
            if entity_type:
                query += " AND e.entity_type = ?"
                params.append(entity_type.value)
            
            query += " ORDER BY e.confidence ASC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            results = []
            for row in rows:
                entity = self._row_to_entity(row)
                doc = Document(
                    id=row["doc_id"],
                    content=row["content"],
                    source=row["doc_source"],
                )
                results.append((entity, doc))
            
            return results
    
    def update_entity_llm(
        self, 
        entity_id: str, 
        llm_confidence: float,
        llm_reasoning: Optional[str] = None
    ):
        """Update entity with LLM verification results."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE entities 
                   SET llm_confidence = ?, llm_reasoning = ?
                   WHERE id = ?""",
                (llm_confidence, llm_reasoning, entity_id)
            )
    
    def count_entities(self) -> int:
        """Count total entities."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    
    def count_entities_above_threshold(self, threshold: float) -> int:
        """Count entities at or above threshold."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM entities WHERE confidence >= ?",
                (threshold,)
            ).fetchone()[0]
    
    def count_entities_needing_review(self, threshold: float) -> int:
        """Count entities below threshold that haven't been reviewed."""
        with self._connect() as conn:
            return conn.execute(
                """SELECT COUNT(*) FROM entities e
                   LEFT JOIN corrections c ON e.id = c.entity_id
                   WHERE e.confidence < ? AND c.id IS NULL""",
                (threshold,)
            ).fetchone()[0]
    
    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert database row to Entity."""
        return Entity(
            id=row["id"],
            text=row["text"],
            start=row["start_idx"],
            end=row["end_idx"],
            entity_type=EntityType(row["entity_type"]),
            confidence=row["confidence"],
            source=SourceType(row["source"]),
            llm_confidence=row["llm_confidence"],
            llm_reasoning=row["llm_reasoning"],
        )
    
    # Correction operations
    
    def add_correction(self, correction: Correction) -> str:
        """Add a human review decision."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO corrections 
                   (id, entity_id, document_id, entity_text, entity_start, entity_end,
                    detected_type, decision, correct_type, context_before, context_after,
                    ner_confidence, llm_confidence, reviewed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    correction.id, correction.entity_id, correction.document_id,
                    correction.entity_text, correction.entity_start, correction.entity_end,
                    correction.detected_type.value, correction.decision.value,
                    correction.correct_type.value if correction.correct_type else None,
                    correction.context_before, correction.context_after,
                    correction.ner_confidence, correction.llm_confidence,
                    correction.reviewed_at
                )
            )
        return correction.id
    
    def get_corrections(self) -> List[Correction]:
        """Get all corrections."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM corrections ORDER BY reviewed_at DESC"
            ).fetchall()
            
            return [self._row_to_correction(row) for row in rows]
    
    def count_corrections(self) -> int:
        """Count total corrections."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    
    def count_corrections_by_decision(self) -> dict[DecisionType, int]:
        """Count corrections grouped by decision."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT decision, COUNT(*) as cnt 
                   FROM corrections GROUP BY decision"""
            ).fetchall()
            
            return {DecisionType(row["decision"]): row["cnt"] for row in rows}
    
    def _row_to_correction(self, row: sqlite3.Row) -> Correction:
        """Convert database row to Correction."""
        return Correction(
            id=row["id"],
            entity_id=row["entity_id"],
            document_id=row["document_id"],
            entity_text=row["entity_text"],
            entity_start=row["entity_start"],
            entity_end=row["entity_end"],
            detected_type=EntityType(row["detected_type"]),
            decision=DecisionType(row["decision"]),
            correct_type=EntityType(row["correct_type"]) if row["correct_type"] else None,
            context_before=row["context_before"],
            context_after=row["context_after"],
            ner_confidence=row["ner_confidence"],
            llm_confidence=row["llm_confidence"],
            reviewed_at=row["reviewed_at"],
        )
    
    # Pattern operations
    
    def add_pattern(
        self, 
        pattern_type: str, 
        pattern: str, 
        entity_type: Optional[str] = None,
        confidence_adjustment: float = 0.0,
        source_corrections: Optional[List[str]] = None
    ) -> str:
        """Add a learned pattern."""
        pattern_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO patterns 
                   (id, pattern_type, pattern, entity_type, confidence_adjustment, source_corrections)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    pattern_id, pattern_type, pattern, entity_type,
                    confidence_adjustment, 
                    json.dumps(source_corrections) if source_corrections else None
                )
            )
        return pattern_id
    
    def get_patterns(self, pattern_type: Optional[str] = None) -> List[dict]:
        """Get patterns, optionally filtered by type."""
        with self._connect() as conn:
            if pattern_type:
                rows = conn.execute(
                    "SELECT * FROM patterns WHERE pattern_type = ?",
                    (pattern_type,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM patterns").fetchall()
            
            return [dict(row) for row in rows]
    
    # Stats
    
    def get_review_stats(self, threshold: float = 0.95) -> ReviewStats:
        """Get training progress statistics."""
        decision_counts = self.count_corrections_by_decision()
        
        return ReviewStats(
            total_documents=self.count_documents(),
            total_entities=self.count_entities(),
            reviewed=self.count_corrections(),
            pending=self.count_entities_needing_review(threshold),
            auto_approved=self.count_entities_above_threshold(threshold),
            confirmed=decision_counts.get(DecisionType.CONFIRMED, 0),
            rejected=decision_counts.get(DecisionType.REJECTED, 0),
            changed=decision_counts.get(DecisionType.CHANGED, 0),
        )
    
    def get_top_fp_patterns(self, limit: int = 10) -> List[tuple[str, str, int]]:
        """Get most common false positive patterns."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT entity_text, detected_type, COUNT(*) as cnt
                   FROM corrections 
                   WHERE decision = 'rejected'
                   GROUP BY entity_text, detected_type
                   ORDER BY cnt DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            
            return [(row["entity_text"], row["detected_type"], row["cnt"]) for row in rows]
    
    def reset(self):
        """Reset database (delete all data)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM corrections")
            conn.execute("DELETE FROM entities")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM patterns")


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def set_db(db: Database):
    """Set global database instance."""
    global _db
    _db = db
