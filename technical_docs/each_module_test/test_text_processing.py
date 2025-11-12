# test_text_processing.py
from utils.text_processing import TextProcessor, TextCleaner


def test_text_processing():
    print("=== Text Processing Test ===\n")

    # Initialize
    processor = TextProcessor()
    cleaner = TextCleaner()

    # Test text
    test_text = """
    The quick brown fox jumps over the lazy dog! 
    This is a TEST of text processing... 
    Email: test@example.com, 
    URL: https://example.com, 
    Date: 12/25/2023,
    Percentage: 85.5%
    """

    # Test 1: Basic preprocessing
    print("1. Basic preprocessing:")
    processed = processor.preprocess(test_text)
    print(f"Original length: {len(test_text)}")
    print(f"Processed length: {len(processed)}")
    print(f"Preview: {processed[:100]}...")

    # Test 2: Sentence extraction
    print("\n2. Sentence extraction:")
    sentences = processor.extract_sentences(processed)
    print(f"Found {len(sentences)} sentences:")

    for i, sent in enumerate(sentences):
        print(f"  {i + 1}. {sent}")

    # Test 3: Keyword extraction
    print("\n3. Keyword extraction:")
    keywords = processor.extract_keywords(processed, n=10)
    print(f"Top keywords: {keywords}")

    # Test 4: Entity extraction
    print("\n4. Entity extraction:")
    entities = processor.extract_entities(processed)
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {values}")

    # Test 5: Text cleaning
    print("\n5. Text cleaning:")

    dirty_text = "I can't believe it's already 2024!!! What a year... #amazing"

    print(f"Original: {dirty_text}")

    # Different cleaning options
    cleaned1 = cleaner.clean_text(dirty_text, expand_contractions=True, lowercase=False)
    print(f"Expanded contractions: {cleaned1}")

    cleaned2 = cleaner.clean_text(dirty_text, remove_special=True, lowercase=True)
    print(f"Removed special + lowercase: {cleaned2}")

    # Test 6: Summarization
    print("\n6. Text summarization:")
    long_text = """
    Artificial intelligence has made remarkable progress in recent years. 
    Machine learning algorithms can now perform tasks that were once thought 
    to be exclusively human. Natural language processing has enabled computers 
    to understand and generate human-like text. Computer vision systems can 
    identify objects and faces with high accuracy. These advances have led to 
    practical applications in healthcare, finance, transportation, and many 
    other fields. However, challenges remain in areas like explainability, 
    bias, and ethical considerations.
    """

    summary = processor.summarize_text(long_text, max_length=100)
    print(f"Original: {len(long_text)} chars")
    print(f"Summary: {summary}")


test_text_processing()
