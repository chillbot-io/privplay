"""Continuous Learning Module for Meta-Classifier.

Provides:
- SignalStore: Persistent storage for captured signals and feedback
- HIL TUI: Human-in-the-loop interface for reviewing uncertain detections
- ContinuousTrainer: Training loop that learns from feedback

Architecture:
    Detection (<0.95 conf) → SignalStore (review queue)
                                   ↓
                            HIL TUI (feedback)
                                   ↓
                            Labeled signals
                                   ↓
                         ContinuousTrainer (nightly, 1000 threshold)
                                   ↓
                            New model → Shadow → Promote

Usage:
    from privplay.learning import (
        SignalStore,
        CapturedSignal,
        FeedbackType,
        run_review_tui,
        trigger_training,
        check_training_status,
    )
    
    # Capture signals during detection
    store = SignalStore("~/.privplay/signals.db")
    store.capture_signal(signal)
    
    # Review signals (TUI)
    run_review_tui()
    
    # Check if training needed
    status = check_training_status()
    
    # Trigger training
    new_model = trigger_training()
"""

from .signal_store import (
    # Store
    SignalStore,
    get_signal_store,
    reset_signal_store,
    
    # Types
    CapturedSignal,
    FalseNegativeReport,
    ModelVersion,
    FeedbackType,
    SignalStatus,
)

from .hil_tui import (
    run_review_tui,
    show_stats_tui,
    ReviewSession,
)

from .trainer import (
    ContinuousTrainer,
    TrainingConfig,
    TrainingDaemon,
    get_trainer,
    check_training_status,
    trigger_training,
)

from .integration import (
    LearningClassifier,
    wrap_for_learning,
    span_signals_to_captured,
)

__all__ = [
    # Store
    "SignalStore",
    "get_signal_store",
    "reset_signal_store",
    
    # Types
    "CapturedSignal",
    "FalseNegativeReport",
    "ModelVersion",
    "FeedbackType",
    "SignalStatus",
    
    # HIL
    "run_review_tui",
    "show_stats_tui",
    "ReviewSession",
    
    # Training
    "ContinuousTrainer",
    "TrainingConfig",
    "TrainingDaemon",
    "get_trainer",
    "check_training_status",
    "trigger_training",
    
    # Integration
    "LearningClassifier",
    "wrap_for_learning",
    "span_signals_to_captured",
]
