"""Storage for span signals - training data for meta-classifier."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, List, Dict
import logging

from ..config import get_config
from ..engine.classifier import SpanSignals

logger = logging.getLogger(__name__)


SPAN_SIGNALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS span_signals (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    span_text TEXT NOT NULL,
    
    phi_bert_detected INTEGER NOT NULL DEFAULT 0,
    phi_bert_conf REAL NOT NULL DEFAULT 0.0,
    phi_bert_type TEXT DEFAULT '',
    
    pii_bert_detected INTEGER NOT NULL DEFAULT 0,
    pii_bert_conf REAL NOT NULL DEFAULT 0.0,
    pii_bert_type TEXT DEFAULT '',
    
    presidio_detected INTEGER NOT NULL DEFAULT 0,
    presidio_conf REAL NOT NULL DEFAULT 0.0,
    presidio_type TEXT DEFAULT '',
    
    rule_detected INTEGER NOT NULL DEFAULT 0,
    rule_conf REAL NOT NULL DEFAULT 0.0,
    rule_type TEXT DEFAULT '',
    rule_has_checksum INTEGER NOT NULL DEFAULT 0,
    
    llm_verified INTEGER NOT NULL DEFAULT 0,
    llm_decision TEXT DEFAULT '',
    llm_conf REAL NOT NULL DEFAULT 0.0,
    
    sources_agree_count INTEGER NOT NULL DEFAULT 0,
    span_length INTEGER NOT NULL DEFAULT 0,
    has_digits INTEGER NOT NULL DEFAULT 0,
    has_letters INTEGER NOT NULL DEFAULT 0,
    all_caps INTEGER NOT NULL DEFAULT 0,
    all_digits INTEGER NOT NULL DEFAULT 0,
    mixed_case INTEGER NOT NULL DEFAULT 0,
    
    merged_type TEXT DEFAULT '',
    merged_conf REAL NOT NULL DEFAULT 0.0,
    merged_source TEXT DEFAULT '',
    
    ground_truth_type TEXT,
    ground_truth_source TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_span_signals_doc ON span_signals(document_id);
CREATE INDEX IF NOT EXISTS idx_span_signals_ground_truth ON span_signals(ground_truth_type);
"""


class SignalsStorage:
    """Storage for span signals."""
    
    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = get_config().data_dir / "signals.db"
        self.path = path
        self._init_db()
    
    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SPAN_SIGNALS_SCHEMA)
    
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def add_signal(self, signal: SpanSignals, document_id: Optional[str] = None) -> str:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO span_signals (
                    id, document_id, span_start, span_end, span_text,
                    phi_bert_detected, phi_bert_conf, phi_bert_type,
                    pii_bert_detected, pii_bert_conf, pii_bert_type,
                    presidio_detected, presidio_conf, presidio_type,
                    rule_detected, rule_conf, rule_type, rule_has_checksum,
                    llm_verified, llm_decision, llm_conf,
                    sources_agree_count, span_length, has_digits, has_letters,
                    all_caps, all_digits, mixed_case,
                    merged_type, merged_conf, merged_source,
                    ground_truth_type, ground_truth_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.id, document_id, signal.span_start, signal.span_end, signal.span_text,
                    int(signal.phi_bert_detected), signal.phi_bert_conf, signal.phi_bert_type,
                    int(signal.pii_bert_detected), signal.pii_bert_conf, signal.pii_bert_type,
                    int(signal.presidio_detected), signal.presidio_conf, signal.presidio_type,
                    int(signal.rule_detected), signal.rule_conf, signal.rule_type, int(signal.rule_has_checksum),
                    int(signal.llm_verified), signal.llm_decision, signal.llm_conf,
                    signal.sources_agree_count, signal.span_length, int(signal.has_digits), int(signal.has_letters),
                    int(signal.all_caps), int(signal.all_digits), int(signal.mixed_case),
                    signal.merged_type, signal.merged_conf, signal.merged_source,
                    signal.ground_truth_type, signal.ground_truth_source,
                )
            )
        return signal.id
    
    def add_signals(self, signals: List[SpanSignals], document_id: Optional[str] = None):
        for signal in signals:
            self.add_signal(signal, document_id)
    
    def get_labeled_signals(self) -> List[SpanSignals]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM span_signals WHERE ground_truth_type IS NOT NULL"
            ).fetchall()
            return [self._row_to_signal(row) for row in rows]
    
    def get_all_signals(self) -> List[SpanSignals]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM span_signals").fetchall()
            return [self._row_to_signal(row) for row in rows]
    
    def count_signals(self) -> Dict[str, int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM span_signals").fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM span_signals WHERE ground_truth_type IS NOT NULL"
            ).fetchone()[0]
            positive = conn.execute(
                "SELECT COUNT(*) FROM span_signals WHERE ground_truth_type IS NOT NULL AND ground_truth_type != 'NONE'"
            ).fetchone()[0]
            negative = conn.execute(
                "SELECT COUNT(*) FROM span_signals WHERE ground_truth_type = 'NONE'"
            ).fetchone()[0]
            
            return {
                "total": total,
                "labeled": labeled,
                "positive": positive,
                "negative": negative,
                "unlabeled": total - labeled,
            }
    
    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM span_signals")
    
    def _row_to_signal(self, row: sqlite3.Row) -> SpanSignals:
        return SpanSignals(
            id=row["id"],
            span_start=row["span_start"],
            span_end=row["span_end"],
            span_text=row["span_text"],
            phi_bert_detected=bool(row["phi_bert_detected"]),
            phi_bert_conf=row["phi_bert_conf"],
            phi_bert_type=row["phi_bert_type"] or "",
            pii_bert_detected=bool(row["pii_bert_detected"]),
            pii_bert_conf=row["pii_bert_conf"],
            pii_bert_type=row["pii_bert_type"] or "",
            presidio_detected=bool(row["presidio_detected"]),
            presidio_conf=row["presidio_conf"],
            presidio_type=row["presidio_type"] or "",
            rule_detected=bool(row["rule_detected"]),
            rule_conf=row["rule_conf"],
            rule_type=row["rule_type"] or "",
            rule_has_checksum=bool(row["rule_has_checksum"]),
            llm_verified=bool(row["llm_verified"]),
            llm_decision=row["llm_decision"] or "",
            llm_conf=row["llm_conf"],
            sources_agree_count=row["sources_agree_count"],
            span_length=row["span_length"],
            has_digits=bool(row["has_digits"]),
            has_letters=bool(row["has_letters"]),
            all_caps=bool(row["all_caps"]),
            all_digits=bool(row["all_digits"]),
            mixed_case=bool(row["mixed_case"]),
            merged_type=row["merged_type"] or "",
            merged_conf=row["merged_conf"],
            merged_source=row["merged_source"] or "",
            ground_truth_type=row["ground_truth_type"],
            ground_truth_source=row["ground_truth_source"],
        )


_storage: Optional[SignalsStorage] = None


def get_signals_storage() -> SignalsStorage:
    global _storage
    if _storage is None:
        _storage = SignalsStorage()
    return _storage
