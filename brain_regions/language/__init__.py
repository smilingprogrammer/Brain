# Conditional imports to support both full and lite versions
try:
    from .comprehension import LanguageComprehension
except ImportError:
    LanguageComprehension = None

try:
    from .production import LanguageProduction
except ImportError:
    LanguageProduction = None

try:
    from .semantic_decoder import SemanticDecoder
except ImportError:
    SemanticDecoder = None

# Always available lite versions
from .comprehension_lite import LanguageComprehensionLite
from .semantic_decoder_lite import SemanticDecoderLite

__all__ = ['LanguageComprehension', 'LanguageProduction', 'SemanticDecoder',
           'LanguageComprehensionLite', 'SemanticDecoderLite']