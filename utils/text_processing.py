import re
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextProcessor:
    """Utilities for text processing"""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text: str) -> str:
        """Basic text preprocessing"""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Fix common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""

        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """Extract keywords from text"""

        # Tokenize
        words = word_tokenize(text.lower())

        # Filter
        keywords = [
            w for w in words
            if w.isalnum() and
               w not in self.stop_words and
               len(w) > 2
        ]

        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)

        # Return top n
        return [word for word, _ in word_freq.most_common(n)]

    def extract_questions(self, text: str) -> List[str]:
        """Extract questions from text"""

        sentences = self.extract_sentences(text)
        questions = [s for s in sentences if s.strip().endswith('?')]

        return questions

    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Simple extractive summarization"""

        sentences = self.extract_sentences(text)

        if not sentences:
            return ""

        # Score sentences by keyword density
        keywords = set(self.extract_keywords(text, n=20))

        scored_sentences = []
        for sent in sentences:
            words = word_tokenize(sent.lower())
            score = sum(1 for w in words if w in keywords) / len(words)
            scored_sentences.append((score, sent))

        # Sort by score
        scored_sentences.sort(reverse=True)

        # Build summary
        summary = []
        current_length = 0

        for score, sent in scored_sentences:
            if current_length + len(sent) <= max_length:
                summary.append(sent)
                current_length += len(sent)
            else:
                break

        return ' '.join(summary)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using regex patterns"""

        entities = {
            "numbers": [],
            "percentages": [],
            "dates": [],
            "urls": [],
            "emails": []
        }

        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        entities["numbers"] = numbers

        # Percentages
        percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
        entities["percentages"] = percentages

        # Simple date patterns
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        entities["dates"] = dates

        # URLs
        urls = re.findall(r'https?://[^\s]+', text)
        entities["urls"] = urls

        # Emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities["emails"] = emails

        return entities


class TextCleaner:
    """Advanced text cleaning utilities"""

    def __init__(self):
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }

    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""

        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def remove_special_chars(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters"""

        if keep_punctuation:
            # Keep basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'"()]', '', text)
        else:
            # Remove all non-alphanumeric
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def clean_text(self, text: str,
                   expand_contractions: bool = True,
                   remove_special: bool = True,
                   lowercase: bool = False) -> str:
        """Complete text cleaning pipeline"""

        if expand_contractions:
            text = self.expand_contractions(text)

        if remove_special:
            text = self.remove_special_chars(text)

        text = self.normalize_whitespace(text)

        if lowercase:
            text = text.lower()

        return text