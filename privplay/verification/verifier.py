"""LLM verification for uncertain PHI/PII detections."""

from abc import ABC, abstractmethod
from typing import Optional
import logging
import httpx

from ..types import VerificationResponse, VerificationResult, EntityType
from ..config import get_config, VerificationConfig

logger = logging.getLogger(__name__)


VERIFICATION_PROMPT = """You are a PHI/PII detection assistant. Your task is to determine if a specific term is Protected Health Information (PHI) or Personally Identifiable Information (PII) that could identify a specific individual.

Given this clinical/medical text:
"{context}"

The term "{entity}" was flagged as potential {entity_type}.

Is "{entity}" PHI/PII that could identify a specific patient or individual?

Consider:
- Is this a specific person's name, or a generic term?
- Is this a specific identifier, or a general category?
- Could this be used to identify or contact a specific person?

For example:
- "MICU" (medical intensive care unit) is NOT PHI - it's a generic unit name
- "John Smith" IS PHI - it's a specific person's name
- "Building 4" is likely NOT PHI - it's a generic location
- "123-45-6789" IS PHI - it's a specific SSN

Answer with ONLY one word: YES or NO"""


class Verifier(ABC):
    """Base interface for LLM verification."""
    
    @abstractmethod
    def verify(
        self, 
        entity_text: str, 
        entity_type: EntityType,
        context: str
    ) -> VerificationResponse:
        """Ask LLM if entity is PHI in context."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if verifier is ready."""
        pass


class OllamaVerifier(Verifier):
    """Verify using local Ollama server."""
    
    def __init__(self, url: str = "http://localhost:11434", model: str = "phi3:mini"):
        self.url = url.rstrip("/")
        self.model = model
        self._available: Optional[bool] = None
    
    def verify(
        self, 
        entity_text: str, 
        entity_type: EntityType,
        context: str
    ) -> VerificationResponse:
        """Ask Ollama if entity is PHI."""
        if not self.is_available():
            return VerificationResponse(
                decision=VerificationResult.UNCERTAIN,
                confidence=0.5,
                reasoning="Ollama not available"
            )
        
        prompt = VERIFICATION_PROMPT.format(
            context=context,
            entity=entity_text,
            entity_type=entity_type.value
        )
        
        try:
            response = httpx.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistent answers
                        "num_predict": 10,   # We only need YES/NO
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "").strip().upper()
            
            # Parse response
            if "YES" in answer:
                return VerificationResponse(
                    decision=VerificationResult.YES,
                    confidence=0.95,
                    reasoning=f"LLM confirmed as PHI"
                )
            elif "NO" in answer:
                return VerificationResponse(
                    decision=VerificationResult.NO,
                    confidence=0.95,
                    reasoning=f"LLM rejected as not PHI"
                )
            else:
                return VerificationResponse(
                    decision=VerificationResult.UNCERTAIN,
                    confidence=0.5,
                    reasoning=f"LLM unclear: {answer[:50]}"
                )
                
        except Exception as e:
            logger.error(f"Ollama verification failed: {e}")
            return VerificationResponse(
                decision=VerificationResult.UNCERTAIN,
                confidence=0.5,
                reasoning=f"Error: {str(e)}"
            )
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is not None:
            return self._available
        
        try:
            # Check if Ollama is running
            response = httpx.get(f"{self.url}/api/tags", timeout=5.0)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check for exact match or partial match
            if any(self.model in name or name in self.model for name in model_names):
                self._available = True
            else:
                logger.warning(f"Model {self.model} not found. Available: {model_names}")
                self._available = False
            
            return self._available
            
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not available."""
        try:
            logger.info(f"Pulling model {self.model}...")
            response = httpx.post(
                f"{self.url}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=600.0  # 10 min timeout for download
            )
            response.raise_for_status()
            self._available = True
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False


class RemoteVerifier(Verifier):
    """Verify using remote OpenAI-compatible API."""
    
    def __init__(self, url: str, api_key: str, model: str):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model = model
    
    def verify(
        self, 
        entity_text: str, 
        entity_type: EntityType,
        context: str
    ) -> VerificationResponse:
        """Ask remote LLM if entity is PHI."""
        prompt = VERIFICATION_PROMPT.format(
            context=context,
            entity=entity_text,
            entity_type=entity_type.value
        )
        
        try:
            response = httpx.post(
                f"{self.url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 10,
                },
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip().upper()
            
            if "YES" in answer:
                return VerificationResponse(
                    decision=VerificationResult.YES,
                    confidence=0.95,
                    reasoning="LLM confirmed as PHI"
                )
            elif "NO" in answer:
                return VerificationResponse(
                    decision=VerificationResult.NO,
                    confidence=0.95,
                    reasoning="LLM rejected as not PHI"
                )
            else:
                return VerificationResponse(
                    decision=VerificationResult.UNCERTAIN,
                    confidence=0.5,
                    reasoning=f"LLM unclear: {answer[:50]}"
                )
                
        except Exception as e:
            logger.error(f"Remote verification failed: {e}")
            return VerificationResponse(
                decision=VerificationResult.UNCERTAIN,
                confidence=0.5,
                reasoning=f"Error: {str(e)}"
            )
    
    def is_available(self) -> bool:
        """Check if remote API is configured."""
        return bool(self.url and self.api_key and self.model)


class NoopVerifier(Verifier):
    """Skip LLM verification - everything goes to human review."""
    
    def verify(
        self, 
        entity_text: str, 
        entity_type: EntityType,
        context: str
    ) -> VerificationResponse:
        return VerificationResponse(
            decision=VerificationResult.UNCERTAIN,
            confidence=0.5,
            reasoning="LLM verification disabled"
        )
    
    def is_available(self) -> bool:
        return True


def get_verifier(config: Optional[VerificationConfig] = None) -> Verifier:
    """Factory function to get appropriate verifier."""
    if config is None:
        config = get_config().verification
    
    if config.provider == "none":
        return NoopVerifier()
    
    if config.provider == "remote":
        return RemoteVerifier(
            url=config.remote.url,
            api_key=config.remote.api_key,
            model=config.remote.model
        )
    
    # Default to Ollama
    return OllamaVerifier(
        url=config.ollama.url,
        model=config.ollama.model
    )
