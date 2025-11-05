from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BrainRegion(ABC):
    """Base interface for all brain regions"""

    @abstractmethod
    async def initialize(self):
        """Initialize the brain region"""
        pass

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return output"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current state for monitoring"""
        pass


class MemorySystem(BrainRegion):
    """Interface for memory systems"""

    @abstractmethod
    async def store(self, data: Any, metadata: Optional[Dict] = None):
        """Store information"""
        pass

    @abstractmethod
    async def retrieve(self, query: Any, k: int = 5) -> list:
        """Retrieve relevant information"""
        pass

    @abstractmethod
    async def consolidate(self):
        """Consolidate/compress memories"""
        pass


class ReasoningModule(BrainRegion):
    """Interface for reasoning modules"""

    @abstractmethod
    async def reason(self, problem: str, context: Dict) -> Dict:
        """Perform reasoning on the problem"""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """Get confidence in last reasoning"""
        pass