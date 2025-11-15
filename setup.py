from setuptools import setup, find_packages

setup(
    name="cognitive-text-brain",
    version="0.1.0",
    description="A neuromorphic cognitive architecture for text reasoning",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "google-generativeai>=0.3.0",
        "nengo>=3.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "spacy>=3.5.0",
        "structlog>=23.0.0",
        "pydantic>=2.0",
        "python-dotenv",
        "pyyaml>=6.0",
        "asyncio",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "neo4j>=5.0.0",
        "prometheus-client>=0.16.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black",
            "flake8",
            "mypy"
        ]
    },
    entry_points={
        "console_scripts": [
            "cognitive-brain=main:main",
        ],
    },
)