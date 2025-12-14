"""
Meta-classifier for learned entity merging.

Takes signals from all detectors and learns optimal merge strategy.
Supports online learning from human corrections.
"""

import json
import logging
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency
sklearn = None


def _ensure_sklearn():
    """Lazy load sklearn."""
    global sklearn
    if sklearn is None:
        try:
            import sklearn as sk
            sklearn = sk
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")


class MetaClassifier:
    """
    Learned meta-classifier for entity merge decisions.
    
    Takes signal vectors from all detectors and predicts:
    1. is_entity: bool (is this span actually PHI/PII?)
    2. entity_type: str (what type is it?)
    3. confidence: float (how sure are we?)
    
    Supports online learning via incremental updates.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_samples_for_training: int = 50,
    ):
        self.model_path = model_path or Path.home() / ".privplay" / "meta_classifier"
        self.min_samples_for_training = min_samples_for_training
        
        self._is_entity_model = None
        self._entity_type_model = None
        self._label_encoder = None
        self._feature_names = None
        self._is_trained = False
        
        # Try to load existing model
        self._load()
    
    def _load(self):
        """Load trained model from disk."""
        model_file = self.model_path / "model.pkl"
        if model_file.exists():
            try:
                with open(model_file, "rb") as f:
                    data = pickle.load(f)
                self._is_entity_model = data["is_entity_model"]
                self._entity_type_model = data["entity_type_model"]
                self._label_encoder = data["label_encoder"]
                self._feature_names = data["feature_names"]
                self._is_trained = True
                logger.info(f"Loaded meta-classifier from {model_file}")
            except Exception as e:
                logger.warning(f"Failed to load meta-classifier: {e}")
    
    def _save(self):
        """Save trained model to disk."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        model_file = self.model_path / "model.pkl"
        
        with open(model_file, "wb") as f:
            pickle.dump({
                "is_entity_model": self._is_entity_model,
                "entity_type_model": self._entity_type_model,
                "label_encoder": self._label_encoder,
                "feature_names": self._feature_names,
                "saved_at": datetime.utcnow().isoformat(),
            }, f)
        
        logger.info(f"Saved meta-classifier to {model_file}")
    
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
    
    def predict(self, signals: "SpanSignals") -> Tuple[bool, str, float]:
        """
        Predict whether span is entity and what type.
        
        Args:
            signals: SpanSignals object with all detector outputs
            
        Returns:
            (is_entity, entity_type, confidence)
        """
        if not self._is_trained:
            # Fall back to rule-based merge
            return self._fallback_predict(signals)
        
        _ensure_sklearn()
        
        # Convert to feature vector
        features = signals.to_feature_dict()
        X = np.array([[features[name] for name in self._feature_names]])
        
        # Predict is_entity
        is_entity_proba = self._is_entity_model.predict_proba(X)[0]
        is_entity = is_entity_proba[1] > 0.5
        confidence = float(is_entity_proba[1])
        
        # Predict entity type
        if is_entity:
            type_pred = self._entity_type_model.predict(X)[0]
            entity_type = self._label_encoder.inverse_transform([type_pred])[0]
        else:
            entity_type = "NONE"
        
        return is_entity, entity_type, confidence
    
    def _fallback_predict(self, signals: "SpanSignals") -> Tuple[bool, str, float]:
        """Fallback when model not trained - use current merge logic."""
        # If any detector found something with high confidence, trust it
        if signals.rule_detected and signals.rule_conf >= 0.9:
            return True, signals.rule_type, signals.rule_conf
        
        if signals.presidio_detected and signals.presidio_conf >= 0.7:
            return True, signals.presidio_type, signals.presidio_conf
        
        if signals.pii_bert_detected and signals.pii_bert_conf >= 0.7:
            return True, signals.pii_bert_type, signals.pii_bert_conf
        
        if signals.phi_bert_detected and signals.phi_bert_conf >= 0.7:
            return True, signals.phi_bert_type, signals.phi_bert_conf
        
        # Multiple weak signals = probably entity
        if signals.sources_agree_count >= 2:
            # Pick the most confident type
            best_type = signals.merged_type
            best_conf = max(
                signals.phi_bert_conf, 
                signals.pii_bert_conf,
                signals.presidio_conf,
                signals.rule_conf,
            )
            return True, best_type, min(0.8, best_conf + 0.1)
        
        # Single weak signal - uncertain
        if signals.sources_agree_count == 1:
            return True, signals.merged_type, signals.merged_conf
        
        return False, "NONE", 0.0
    
    def train(
        self,
        signals_list: List["SpanSignals"],
        use_xgboost: bool = False,
    ) -> Dict[str, float]:
        """
        Train meta-classifier on labeled signals.
        
        Args:
            signals_list: List of SpanSignals with ground_truth_type set
            use_xgboost: Use XGBoost instead of RandomForest
            
        Returns:
            Training metrics
        """
        _ensure_sklearn()
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Filter to labeled samples
        labeled = [s for s in signals_list if s.ground_truth_type is not None]
        
        if len(labeled) < self.min_samples_for_training:
            raise ValueError(
                f"Need at least {self.min_samples_for_training} labeled samples, "
                f"got {len(labeled)}"
            )
        
        logger.info(f"Training on {len(labeled)} labeled samples")
        
        # Build feature matrix
        feature_dicts = [s.to_feature_dict() for s in labeled]
        self._feature_names = list(feature_dicts[0].keys())
        X = np.array([[d[name] for name in self._feature_names] for d in feature_dicts])
        
        # Build labels
        # is_entity: True if ground_truth_type != "NONE"
        y_is_entity = np.array([s.ground_truth_type != "NONE" for s in labeled])
        
        # entity_type: actual type (for positives only)
        y_types = [s.ground_truth_type for s in labeled]
        self._label_encoder = LabelEncoder()
        y_type_encoded = self._label_encoder.fit_transform(y_types)
        
        # Split data
        X_train, X_test, y_ent_train, y_ent_test, y_type_train, y_type_test = train_test_split(
            X, y_is_entity, y_type_encoded, test_size=0.2, random_state=42
        )
        
        # Train is_entity classifier
        if use_xgboost:
            try:
                from xgboost import XGBClassifier
                self._is_entity_model = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )
            except ImportError:
                logger.warning("XGBoost not available, using RandomForest")
                self._is_entity_model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
        else:
            self._is_entity_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        
        self._is_entity_model.fit(X_train, y_ent_train)
        
        # Train entity_type classifier (on all samples for now)
        self._entity_type_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self._entity_type_model.fit(X_train, y_type_train)
        
        self._is_trained = True
        
        # Evaluate
        y_ent_pred = self._is_entity_model.predict(X_test)
        
        metrics = {
            "is_entity_precision": precision_score(y_ent_test, y_ent_pred),
            "is_entity_recall": recall_score(y_ent_test, y_ent_pred),
            "is_entity_f1": f1_score(y_ent_test, y_ent_pred),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        
        logger.info(f"Training complete: {metrics}")
        
        # Save model
        self._save()
        
        # Save feature importances
        self._save_feature_importance()
        
        return metrics
    
    def _save_feature_importance(self):
        """Save feature importance for interpretability."""
        if not self._is_trained:
            return
        
        importances = dict(zip(
            self._feature_names,
            self._is_entity_model.feature_importances_
        ))
        
        # Sort by importance
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
        
        imp_file = self.model_path / "feature_importance.json"
        with open(imp_file, "w") as f:
            json.dump({
                "importances": sorted_imp,
                "saved_at": datetime.utcnow().isoformat(),
            }, f, indent=2)
        
        logger.info(f"Top features: {sorted_imp[:5]}")
    
    def update(
        self,
        new_signals: List["SpanSignals"],
        epochs: int = 1,
    ) -> Dict[str, float]:
        """
        Incrementally update model with new labeled data.
        
        For online learning loop.
        """
        if not self._is_trained:
            return self.train(new_signals)
        
        _ensure_sklearn()
        
        # Filter to labeled
        labeled = [s for s in new_signals if s.ground_truth_type is not None]
        if not labeled:
            return {"updated": 0}
        
        # Build features
        X_new = np.array([
            [s.to_feature_dict()[name] for name in self._feature_names]
            for s in labeled
        ])
        y_ent_new = np.array([s.ground_truth_type != "NONE" for s in labeled])
        
        # Handle new entity types
        y_types_new = [s.ground_truth_type for s in labeled]
        for t in y_types_new:
            if t not in self._label_encoder.classes_:
                # Add new class
                self._label_encoder.classes_ = np.append(
                    self._label_encoder.classes_, t
                )
        y_type_new = self._label_encoder.transform(y_types_new)
        
        # Warm start training (partial fit not available for RF, so retrain)
        # In production, would use incremental learner or store all data
        
        # For now, just retrain with new data
        # This is a simplified approach - full implementation would
        # maintain a sliding window of training data
        
        for _ in range(epochs):
            self._is_entity_model.fit(X_new, y_ent_new)
            self._entity_type_model.fit(X_new, y_type_new)
        
        self._save()
        
        return {"updated": len(labeled)}
    
    def get_confidence_threshold_recommendation(
        self,
        signals_list: List["SpanSignals"],
        target_recall: float = 0.95,
    ) -> float:
        """
        Recommend confidence threshold for target recall.
        
        Useful for tuning the "needs human review" threshold.
        """
        if not self._is_trained:
            return 0.5  # Default
        
        _ensure_sklearn()
        
        labeled = [s for s in signals_list if s.ground_truth_type is not None]
        if len(labeled) < 10:
            return 0.5
        
        # Get predictions
        X = np.array([
            [s.to_feature_dict()[name] for name in self._feature_names]
            for s in labeled
        ])
        y_true = np.array([s.ground_truth_type != "NONE" for s in labeled])
        y_proba = self._is_entity_model.predict_proba(X)[:, 1]
        
        # Find threshold that achieves target recall
        thresholds = np.arange(0.1, 0.9, 0.05)
        for thresh in thresholds:
            y_pred = y_proba >= thresh
            recall = (y_pred & y_true).sum() / y_true.sum() if y_true.sum() > 0 else 0
            if recall >= target_recall:
                return float(thresh)
        
        return 0.3  # Low threshold if can't achieve target


class OnlineLearningLoop:
    """
    Orchestrates the online learning cycle:
    
    1. Detection → SpanSignals captured
    2. Low confidence → Human review queue  
    3. Human decision → Update ground_truth
    4. Periodic retrain on new data
    5. Deploy updated model
    """
    
    def __init__(
        self,
        meta_classifier: MetaClassifier,
        db,  # Your SQLite DB instance
        retrain_threshold: int = 100,  # Retrain after N new corrections
    ):
        self.meta_classifier = meta_classifier
        self.db = db
        self.retrain_threshold = retrain_threshold
        self._corrections_since_retrain = 0
    
    def record_correction(
        self,
        signals: "SpanSignals",
        is_phi: bool,
        correct_type: Optional[str] = None,
    ):
        """
        Record a human correction for a span.
        
        Args:
            signals: The signal vector for this span
            is_phi: Human says this IS PHI/PII
            correct_type: The correct entity type (if is_phi=True)
        """
        # Update ground truth
        signals.ground_truth_type = correct_type if is_phi else "NONE"
        signals.ground_truth_source = "human"
        
        # Store in DB
        self._store_signals(signals)
        
        self._corrections_since_retrain += 1
        
        # Check if time to retrain
        if self._corrections_since_retrain >= self.retrain_threshold:
            self.retrain()
    
    def _store_signals(self, signals: "SpanSignals"):
        """Store signals in database."""
        # Would call self.db.add_span_signals(signals)
        # For now, just log
        logger.info(f"Stored signals: {signals.id} → {signals.ground_truth_type}")
    
    def retrain(self) -> Dict[str, float]:
        """Retrain meta-classifier on all accumulated data."""
        # Would load all signals from DB
        # all_signals = self.db.get_all_span_signals()
        
        # For now, placeholder
        logger.info("Retraining meta-classifier...")
        self._corrections_since_retrain = 0
        
        # Return metrics
        return {"retrained": True}
    
    def get_review_queue(self, limit: int = 20) -> List["SpanSignals"]:
        """
        Get spans that need human review.
        
        These are low-confidence predictions from the meta-classifier.
        """
        # Would query DB for unreviewed low-confidence spans
        # return self.db.get_pending_reviews(limit=limit)
        return []
