# Import all brain regions for easy access
# NOTE: Commented out to avoid loading heavy dependencies (sentence-transformers, spacy, torch)
# If you have these installed, you can uncomment these imports
# For lightweight version (Gemini-only), use main_lite.py which imports modules directly

# from .language import LanguageComprehension, LanguageProduction, SemanticDecoder
# from .memory import WorkingMemory, Hippocampus, SemanticCortex, MemoryConsolidation
# from .executive import PrefrontalCortex, AttentionController, MetaCognition
# from .reasoning import (
#     LogicalReasoning,
#     AnalogicalReasoning,
#     CausalReasoning,
#     CreativeReasoning
# )
# from .integration import GlobalWorkspace, Thalamus, CorpusCallosum
# from .gemini import GeminiService, KnowledgeBase, ReasoningEngine

__all__ = [
    # Language
    'LanguageComprehension',
    'LanguageProduction',
    'SemanticDecoder',

    # Memory
    'WorkingMemory',
    'Hippocampus',
    'SemanticCortex',
    'MemoryConsolidation',

    # Executive
    'PrefrontalCortex',
    'AttentionController',
    'MetaCognition',

    # Reasoning
    'LogicalReasoning',
    'AnalogicalReasoning',
    'CausalReasoning',
    'CreativeReasoning',

    # Integration
    'GlobalWorkspace',
    'Thalamus',
    'CorpusCallosum',

    # Gemini
    'GeminiService',
    'KnowledgeBase',
    'ReasoningEngine'
]