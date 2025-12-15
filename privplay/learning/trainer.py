"""Continuous Training Loop for Meta-Classifier.

Monitors labeled signal count and triggers retraining when threshold reached.
Runs overnight (off-hours) to avoid impacting production.

Training flow:
1. Check if labeled signals >= 1000
2. Load labeled signals from SignalStore
3. Load adversarial cases (100% - always included)
4. Sample 20% of original synthetic corpus
5. Train new model
6. Evaluate on holdout set
7. If F1 drop > 2%, reject and keep current model
8. Otherwise, stage new model for shadow scoring
9. After 24h shadow period, promote to active

Usage:
    # Check if training needed
    phi-train learn status
    
    # Trigger training manually
    phi-train learn retrain
    
    # Run as daemon (checks hourly, trains overnight)
    phi-train learn daemon --start-hour 2 --end-hour 6
"""

import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

from .signal_store import (
    SignalStore,
    ModelVersion,
    CapturedSignal,
    get_signal_store,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for continuous training."""
    
    # Threshold to trigger training
    signal_threshold: int = 1000
    
    # Training data mix
    adversarial_ratio: float = 1.0  # 100% of adversarial cases
    synthetic_ratio: float = 0.2    # 20% of synthetic corpus
    
    # Model evaluation
    holdout_ratio: float = 0.2      # 20% holdout for evaluation
    f1_regression_threshold: float = 0.02  # 2% max F1 drop
    
    # Shadow scoring period
    shadow_hours: int = 24
    
    # Model retention
    keep_models: int = 3  # current + 2 rollback
    
    # Paths
    model_dir: Optional[Path] = None
    synthetic_corpus_path: Optional[Path] = None
    adversarial_corpus_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.model_dir is None:
            from ..config import get_config
            self.model_dir = get_config().data_dir / "models"
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAINING LOOP
# =============================================================================

class ContinuousTrainer:
    """Manages continuous training of the meta-classifier."""
    
    FEATURE_NAMES = [
        # Detector signals
        "phi_bert_detected", "phi_bert_conf",
        "pii_bert_detected", "pii_bert_conf",
        "presidio_detected", "presidio_conf",
        "rule_detected", "rule_conf", "rule_has_checksum",
        
        # Coreference signals
        "in_coref_cluster", "coref_cluster_size", "coref_anchor_conf",
        "coref_is_anchor", "coref_is_pronoun", "coref_has_phi_anchor",
        
        # Document-level features
        "doc_has_ssn", "doc_has_dates", "doc_has_medical_terms",
        "doc_entity_count", "doc_pii_density",
        
        # Computed span features
        "sources_agree_count", "span_length",
        "has_digits", "has_letters", "all_caps", "all_digits", "mixed_case",
    ]
    
    def __init__(
        self,
        store: Optional[SignalStore] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.store = store or get_signal_store()
        self.config = config or TrainingConfig()
    
    def should_train(self) -> Tuple[bool, str]:
        """Check if training should be triggered.
        
        Returns:
            (should_train, reason)
        """
        labeled_count = self.store.get_labeled_signal_count()
        
        if labeled_count >= self.config.signal_threshold:
            return True, f"Threshold reached: {labeled_count} >= {self.config.signal_threshold}"
        
        return False, f"Below threshold: {labeled_count} < {self.config.signal_threshold}"
    
    def train(self, force: bool = False) -> Optional[ModelVersion]:
        """Run training if threshold met.
        
        Args:
            force: Train even if below threshold
            
        Returns:
            New ModelVersion if trained, None otherwise
        """
        should, reason = self.should_train()
        
        if not should and not force:
            logger.info(f"Skipping training: {reason}")
            return None
        
        logger.info(f"Starting training: {reason}")
        
        try:
            # 1. Gather training data
            X, y, metadata = self._prepare_training_data()
            
            if len(X) == 0:
                logger.error("No training data available")
                return None
            
            # 2. Split holdout
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            
            # 3. Train model
            model, train_metrics = self._train_model(X_train, y_train)
            
            # 4. Evaluate on holdout
            test_metrics = self._evaluate_model(model, X_test, y_test)
            
            # 5. Check for regression
            current_model = self.store.get_active_model()
            if current_model:
                f1_drop = current_model.f1_score - test_metrics['f1']
                if f1_drop > self.config.f1_regression_threshold:
                    logger.warning(
                        f"Model rejected: F1 dropped {f1_drop:.1%} "
                        f"(threshold: {self.config.f1_regression_threshold:.1%})"
                    )
                    return None
            
            # 6. Save model
            model_version = self._save_model(model, test_metrics, metadata)
            
            # 7. Register in store
            self.store.register_model(model_version)
            
            # 8. Clean up old models
            self._cleanup_old_models()
            
            logger.info(f"Training complete: {model_version.id} (F1={test_metrics['f1']:.3f})")
            return model_version
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare training data from all sources.
        
        Returns:
            (features, labels, metadata)
        """
        all_features = []
        all_labels = []
        
        metadata = {
            "production_signals": 0,
            "adversarial_signals": 0,
            "synthetic_signals": 0,
        }
        
        # 1. Production signals (from human feedback)
        labeled_signals = self.store.get_labeled_signals()
        for signal in labeled_signals:
            features = signal.to_feature_vector()
            label = 0 if signal.ground_truth_type == "NONE" else 1
            all_features.append(features)
            all_labels.append(label)
        metadata["production_signals"] = len(labeled_signals)
        
        # 2. Adversarial cases (100%)
        adversarial = self._load_adversarial_corpus()
        for signal_dict in adversarial:
            features = self._dict_to_feature_vector(signal_dict)
            label = 0 if signal_dict.get("ground_truth_type") == "NONE" else 1
            all_features.append(features)
            all_labels.append(label)
        metadata["adversarial_signals"] = len(adversarial)
        
        # 3. Synthetic corpus (20% sample)
        synthetic = self._load_synthetic_corpus(sample_ratio=self.config.synthetic_ratio)
        for signal_dict in synthetic:
            features = self._dict_to_feature_vector(signal_dict)
            label = 0 if signal_dict.get("ground_truth_type") == "NONE" else 1
            all_features.append(features)
            all_labels.append(label)
        metadata["synthetic_signals"] = len(synthetic)
        
        logger.info(
            f"Training data: {metadata['production_signals']} production, "
            f"{metadata['adversarial_signals']} adversarial, "
            f"{metadata['synthetic_signals']} synthetic"
        )
        
        return np.array(all_features), np.array(all_labels), metadata
    
    def _load_adversarial_corpus(self) -> List[Dict]:
        """Load adversarial training cases."""
        if self.config.adversarial_corpus_path:
            path = self.config.adversarial_corpus_path
        else:
            from ..config import get_config
            path = get_config().data_dir / "meta_classifier" / "adversarial_signals.json"
        
        if not path.exists():
            logger.warning(f"Adversarial corpus not found: {path}")
            return []
        
        with open(path) as f:
            return json.load(f)
    
    def _load_synthetic_corpus(self, sample_ratio: float = 0.2) -> List[Dict]:
        """Load synthetic training corpus with sampling."""
        if self.config.synthetic_corpus_path:
            path = self.config.synthetic_corpus_path
        else:
            from ..config import get_config
            path = get_config().data_dir / "meta_classifier" / "training_signals.json"
        
        if not path.exists():
            logger.warning(f"Synthetic corpus not found: {path}")
            return []
        
        with open(path) as f:
            all_signals = json.load(f)
        
        # Filter to only synthetic (not adversarial or production)
        synthetic = [
            s for s in all_signals
            if s.get("doc_type") in ("admission_note", "discharge_summary", "progress_note", 
                                      "lab_report", "consultation_note", "referral_letter",
                                      "insurance_form", "brief_note")
        ]
        
        # Sample
        n_sample = int(len(synthetic) * sample_ratio)
        if n_sample < len(synthetic):
            indices = np.random.choice(len(synthetic), n_sample, replace=False)
            synthetic = [synthetic[i] for i in indices]
        
        return synthetic
    
    def _dict_to_feature_vector(self, signal_dict: Dict) -> List[float]:
        """Convert signal dict to feature vector."""
        return [
            int(signal_dict.get("phi_bert_detected", False)),
            signal_dict.get("phi_bert_conf", 0.0),
            int(signal_dict.get("pii_bert_detected", False)),
            signal_dict.get("pii_bert_conf", 0.0),
            int(signal_dict.get("presidio_detected", False)),
            signal_dict.get("presidio_conf", 0.0),
            int(signal_dict.get("rule_detected", False)),
            signal_dict.get("rule_conf", 0.0),
            int(signal_dict.get("rule_has_checksum", False)),
            
            int(signal_dict.get("in_coref_cluster", False)),
            signal_dict.get("coref_cluster_size", 0),
            signal_dict.get("coref_anchor_conf", 0.0),
            int(signal_dict.get("coref_is_anchor", False)),
            int(signal_dict.get("coref_is_pronoun", False)),
            int(signal_dict.get("coref_anchor_type") is not None),
            
            int(signal_dict.get("doc_has_ssn", False)),
            int(signal_dict.get("doc_has_dates", False)),
            int(signal_dict.get("doc_has_medical_terms", False)),
            signal_dict.get("doc_entity_count", 0),
            signal_dict.get("doc_pii_density", 0.0),
            
            signal_dict.get("sources_agree_count", 0),
            signal_dict.get("span_length", 0),
            int(signal_dict.get("has_digits", False)),
            int(signal_dict.get("has_letters", False)),
            int(signal_dict.get("all_caps", False)),
            int(signal_dict.get("all_digits", False)),
            int(signal_dict.get("mixed_case", False)),
        ]
    
    def _split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and holdout sets."""
        from sklearn.model_selection import train_test_split
        
        return train_test_split(
            X, y,
            test_size=self.config.holdout_ratio,
            random_state=42,
            stratify=y,
        )
    
    def _train_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
    ) -> Tuple[Any, Dict]:
        """Train the classifier."""
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
        
        model.fit(X_train, y_train)
        
        # Training metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        y_pred = model.predict(X_train)
        
        metrics = {
            "f1": f1_score(y_train, y_pred),
            "precision": precision_score(y_train, y_pred),
            "recall": recall_score(y_train, y_pred),
        }
        
        return model, metrics
    
    def _evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
    ) -> Dict:
        """Evaluate model on holdout set."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        y_pred = model.predict(X_test)
        
        return {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
    
    def _save_model(
        self, 
        model: Any, 
        metrics: Dict,
        metadata: Dict,
    ) -> ModelVersion:
        """Save trained model and create ModelVersion."""
        import joblib
        
        model_id = secrets.token_hex(8)
        model_path = self.config.model_dir / f"model_{model_id}.pkl"
        
        joblib.dump(model, model_path)
        
        # Save metadata
        meta_path = self.config.model_dir / f"model_{model_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "id": model_id,
                "created_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "training_data": metadata,
                "feature_names": self.FEATURE_NAMES,
            }, f, indent=2)
        
        return ModelVersion(
            id=model_id,
            created_at=datetime.utcnow(),
            training_signals=metadata["production_signals"],
            adversarial_signals=metadata["adversarial_signals"],
            synthetic_signals=metadata["synthetic_signals"],
            f1_score=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            status="staged",
            model_path=str(model_path),
        )
    
    def _cleanup_old_models(self):
        """Remove old models beyond retention limit."""
        models = self.store.get_model_history(limit=100)
        
        # Keep active and last N rolled_back
        active = [m for m in models if m.status == "active"]
        rolled_back = [m for m in models if m.status == "rolled_back"]
        
        # Keep only config.keep_models - 1 rolled back (one slot for active)
        to_delete = rolled_back[self.config.keep_models - 1:]
        
        for model in to_delete:
            # Delete model file
            model_path = Path(model.model_path)
            if model_path.exists():
                model_path.unlink()
            
            # Delete metadata
            meta_path = model_path.with_suffix("").with_name(
                model_path.stem + "_meta.json"
            )
            if meta_path.exists():
                meta_path.unlink()
            
            logger.info(f"Cleaned up old model: {model.id}")
    
    def promote_if_ready(self) -> bool:
        """Check if staged model is ready for promotion.
        
        Promotes after shadow period (24h) if no issues detected.
        
        Returns:
            True if a model was promoted
        """
        # Find staged models
        models = self.store.get_model_history(limit=10)
        staged = [m for m in models if m.status == "staged"]
        
        if not staged:
            return False
        
        # Check if oldest staged has completed shadow period
        oldest = min(staged, key=lambda m: m.created_at)
        shadow_end = oldest.created_at + timedelta(hours=self.config.shadow_hours)
        
        if datetime.utcnow() < shadow_end:
            remaining = shadow_end - datetime.utcnow()
            logger.info(f"Model {oldest.id} still in shadow period ({remaining} remaining)")
            return False
        
        # Promote
        self.store.promote_model(oldest.id)
        logger.info(f"Promoted model {oldest.id} to active")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get training status."""
        should_train, reason = self.should_train()
        
        labeled_count = self.store.get_labeled_signal_count()
        pending_count = self.store.get_pending_count()
        
        active_model = self.store.get_active_model()
        models = self.store.get_model_history(limit=5)
        staged = [m for m in models if m.status == "staged"]
        
        return {
            "should_train": should_train,
            "reason": reason,
            "labeled_signals": labeled_count,
            "pending_review": pending_count,
            "threshold": self.config.signal_threshold,
            "active_model": {
                "id": active_model.id,
                "f1": active_model.f1_score,
                "created_at": active_model.created_at.isoformat(),
            } if active_model else None,
            "staged_models": [
                {
                    "id": m.id,
                    "f1": m.f1_score,
                    "created_at": m.created_at.isoformat(),
                    "shadow_remaining": str(
                        m.created_at + timedelta(hours=self.config.shadow_hours) - datetime.utcnow()
                    ),
                }
                for m in staged
            ],
        }


# =============================================================================
# DAEMON MODE
# =============================================================================

class TrainingDaemon:
    """Background daemon that monitors and triggers training."""
    
    def __init__(
        self,
        trainer: Optional[ContinuousTrainer] = None,
        check_interval_minutes: int = 60,
        training_start_hour: int = 2,  # 2 AM
        training_end_hour: int = 6,    # 6 AM
    ):
        self.trainer = trainer or ContinuousTrainer()
        self.check_interval = check_interval_minutes * 60
        self.training_start_hour = training_start_hour
        self.training_end_hour = training_end_hour
        
        self.running = False
    
    def is_training_window(self) -> bool:
        """Check if current time is within training window."""
        hour = datetime.now().hour
        return self.training_start_hour <= hour < self.training_end_hour
    
    def run(self):
        """Run the daemon loop."""
        self.running = True
        logger.info(
            f"Training daemon started (window: {self.training_start_hour}:00-{self.training_end_hour}:00)"
        )
        
        while self.running:
            try:
                # Check for promotion
                self.trainer.promote_if_ready()
                
                # Check for training
                if self.is_training_window():
                    should_train, reason = self.trainer.should_train()
                    if should_train:
                        logger.info(f"Training triggered: {reason}")
                        self.trainer.train()
                
                # Archive old signals periodically
                self.trainer.store.archive_old_signals()
                
            except Exception as e:
                logger.error(f"Daemon error: {e}")
            
            # Wait for next check
            time.sleep(self.check_interval)
    
    def stop(self):
        """Stop the daemon."""
        self.running = False


# =============================================================================
# CONVENIENCE
# =============================================================================

def get_trainer(
    store: Optional[SignalStore] = None,
    config: Optional[TrainingConfig] = None,
) -> ContinuousTrainer:
    """Get a ContinuousTrainer instance."""
    return ContinuousTrainer(store=store, config=config)


def check_training_status() -> Dict[str, Any]:
    """Quick check of training status."""
    trainer = get_trainer()
    return trainer.get_status()


def trigger_training(force: bool = False) -> Optional[ModelVersion]:
    """Trigger a training run."""
    trainer = get_trainer()
    return trainer.train(force=force)
