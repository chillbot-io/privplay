"""Configuration for Privplay."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class PresidioConfig:
    """Presidio PII detection settings."""
    enabled: bool = True
    score_threshold: float = 0.7
    languages: list = field(default_factory=lambda: ["en"])


@dataclass
class OllamaConfig:
    """Ollama verification settings."""
    url: str = "http://localhost:11434"
    model: str = "phi3:mini"
    timeout: int = 30


@dataclass
class RemoteConfig:
    """Remote LLM verification settings (enterprise)."""
    url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class VerificationConfig:
    """LLM verification settings."""
    provider: str = "ollama"  # "ollama" | "remote" | "none"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)


@dataclass
class Config:
    """Main configuration."""
    # Paths
    data_dir: Path = field(default_factory=lambda: Path.home() / ".privplay")
    db_path: Path = field(default=None)
    
    # Detection settings
    confidence_threshold: float = 0.95  # Below this → review
    context_window: int = 50  # Characters before/after for context
    
    # Presidio PII detection (defense in depth)
    presidio: PresidioConfig = field(default_factory=PresidioConfig)
    
    # Verification
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    
    def __post_init__(self):
        if self.db_path is None:
            self.db_path = self.data_dir / "privplay.db"
        
        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load config from file or use defaults."""
        if path is None:
            path = Path.home() / ".privplay" / "config.yaml"
        
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                return cls._from_dict(data)
        
        return cls()
    
    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create config from dict."""
        config = cls()
        
        if "data_dir" in data:
            config.data_dir = Path(data["data_dir"])
            config.db_path = config.data_dir / "privplay.db"
        
        if "confidence_threshold" in data:
            config.confidence_threshold = data["confidence_threshold"]
        
        if "context_window" in data:
            config.context_window = data["context_window"]
        
        if "verification" in data:
            v = data["verification"]
            config.verification.provider = v.get("provider", "ollama")
            
            if "ollama" in v:
                config.verification.ollama.url = v["ollama"].get("url", "http://localhost:11434")
                config.verification.ollama.model = v["ollama"].get("model", "phi3:mini")
            
            if "remote" in v:
                config.verification.remote.url = v["remote"].get("url", "")
                config.verification.remote.api_key = os.environ.get(
                    "LLM_API_KEY", 
                    v["remote"].get("api_key", "")
                )
                config.verification.remote.model = v["remote"].get("model", "")
        
        return config
    
    def save(self, path: Optional[Path] = None):
        """Save config to file."""
        if path is None:
            path = self.data_dir / "config.yaml"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "data_dir": str(self.data_dir),
            "confidence_threshold": self.confidence_threshold,
            "context_window": self.context_window,
            "verification": {
                "provider": self.verification.provider,
                "ollama": {
                    "url": self.verification.ollama.url,
                    "model": self.verification.ollama.model,
                },
                "remote": {
                    "url": self.verification.remote.url,
                    "model": self.verification.remote.model,
                }
            }
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config):
    """Set global config instance."""
    global _config
    _config = config
