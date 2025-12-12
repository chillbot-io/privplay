"""Benchmark result storage."""

import sqlite3
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """A single benchmark run result."""
    id: Optional[int] = None
    run_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dataset_name: str = ""
    num_samples: int = 0
    
    # Overall metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Per-entity-type breakdown
    by_entity_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-stack-component breakdown (model, presidio, rules, etc.)
    by_component: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkRun":
        """Create from dictionary."""
        d = d.copy()
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


class BenchmarkStorage:
    """SQLite storage for benchmark results."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize storage with database path.
        
        Args:
            db_path: Path to SQLite database. Defaults to ~/.privplay/benchmarks.db
        """
        if db_path is None:
            from ..config import get_config
            db_path = get_config().data_dir / "benchmarks.db"
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    num_samples INTEGER NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1 REAL NOT NULL,
                    true_positives INTEGER NOT NULL,
                    false_positives INTEGER NOT NULL,
                    false_negatives INTEGER NOT NULL,
                    by_entity_type TEXT NOT NULL,
                    by_component TEXT NOT NULL,
                    config TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_timestamp 
                ON benchmark_runs(timestamp DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_dataset 
                ON benchmark_runs(dataset_name)
            """)
            
            conn.commit()
    
    def save_run(self, run: BenchmarkRun) -> int:
        """
        Save a benchmark run to the database.
        
        Args:
            run: BenchmarkRun to save
            
        Returns:
            Database ID of saved run
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO benchmark_runs (
                    run_id, timestamp, dataset_name, num_samples,
                    precision, recall, f1,
                    true_positives, false_positives, false_negatives,
                    by_entity_type, by_component, config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                run.timestamp.isoformat(),
                run.dataset_name,
                run.num_samples,
                run.precision,
                run.recall,
                run.f1,
                run.true_positives,
                run.false_positives,
                run.false_negatives,
                json.dumps(run.by_entity_type),
                json.dumps(run.by_component),
                json.dumps(run.config),
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_runs(
        self,
        dataset_name: Optional[str] = None,
        limit: int = 5,
    ) -> List[BenchmarkRun]:
        """
        Get recent benchmark runs.
        
        Args:
            dataset_name: Filter by dataset name (optional)
            limit: Maximum number of runs to return
            
        Returns:
            List of BenchmarkRun objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if dataset_name:
                cursor = conn.execute("""
                    SELECT * FROM benchmark_runs 
                    WHERE dataset_name = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (dataset_name, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM benchmark_runs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            runs = []
            for row in cursor.fetchall():
                runs.append(BenchmarkRun(
                    id=row["id"],
                    run_id=row["run_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    dataset_name=row["dataset_name"],
                    num_samples=row["num_samples"],
                    precision=row["precision"],
                    recall=row["recall"],
                    f1=row["f1"],
                    true_positives=row["true_positives"],
                    false_positives=row["false_positives"],
                    false_negatives=row["false_negatives"],
                    by_entity_type=json.loads(row["by_entity_type"]),
                    by_component=json.loads(row["by_component"]),
                    config=json.loads(row["config"]),
                ))
            
            return runs
    
    def get_run_by_id(self, run_id: str) -> Optional[BenchmarkRun]:
        """Get a specific run by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return BenchmarkRun(
                id=row["id"],
                run_id=row["run_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                dataset_name=row["dataset_name"],
                num_samples=row["num_samples"],
                precision=row["precision"],
                recall=row["recall"],
                f1=row["f1"],
                true_positives=row["true_positives"],
                false_positives=row["false_positives"],
                false_negatives=row["false_negatives"],
                by_entity_type=json.loads(row["by_entity_type"]),
                by_component=json.loads(row["by_component"]),
                config=json.loads(row["config"]),
            )
    
    def get_all_runs(
        self,
        dataset_name: Optional[str] = None,
    ) -> List[BenchmarkRun]:
        """Get all benchmark runs."""
        return self.get_recent_runs(dataset_name=dataset_name, limit=10000)
    
    def get_stats_summary(
        self,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get summary statistics across runs."""
        runs = self.get_recent_runs(dataset_name=dataset_name, limit=100)
        
        if not runs:
            return {
                "num_runs": 0,
                "avg_f1": 0.0,
                "best_f1": 0.0,
                "worst_f1": 0.0,
                "trend": "no_data",
            }
        
        f1_scores = [r.f1 for r in runs]
        recent_5 = f1_scores[:5] if len(f1_scores) >= 5 else f1_scores
        
        # Calculate trend
        if len(recent_5) >= 2:
            if recent_5[0] > recent_5[-1]:
                trend = "improving"
            elif recent_5[0] < recent_5[-1]:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "num_runs": len(runs),
            "avg_f1": sum(f1_scores) / len(f1_scores),
            "best_f1": max(f1_scores),
            "worst_f1": min(f1_scores),
            "recent_avg_f1": sum(recent_5) / len(recent_5),
            "trend": trend,
        }
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a benchmark run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM benchmark_runs WHERE run_id = ?",
                (run_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_all(self) -> int:
        """Clear all benchmark data. Returns number of deleted runs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM benchmark_runs")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM benchmark_runs")
            conn.commit()
            return count
