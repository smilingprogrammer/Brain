cognitive-text-brain/
├── README.md
├── requirements.txt
├── setup.py
├── docker-compose.yml
├── .env.example
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── prompts.yaml
├── core/
│   ├── __init__.py
│   ├── event_bus.py
│   ├── interfaces.py
│   ├── logging.py
│   └── metrics.py
├── brain_regions/
│   ├── __init__.py
│   ├── language/
│   │   ├── __init__.py
│   │   ├── comprehension.py      # Wernicke's area
│   │   ├── production.py         # Broca's area
│   │   └── semantic_decoder.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working_memory.py     # Prefrontal working memory
│   │   ├── hippocampus.py        # Episodic encoding/retrieval
│   │   ├── semantic_cortex.py    # Long-term semantic storage
│   │   └── memory_consolidation.py
│   ├── executive/
│   │   ├── __init__.py
│   │   ├── prefrontal_cortex.py  # Executive control
│   │   ├── attention.py          # Attention mechanisms
│   │   └── meta_cognition.py     # Self-monitoring
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── logical_reasoning.py
│   │   ├── analogical_reasoning.py
│   │   ├── causal_reasoning.py
│   │   └── creative_reasoning.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── global_workspace.py   # Consciousness-like integration
│   │   ├── thalamus.py          # Information routing
│   │   └── corpus_callosum.py   # Hemisphere integration
│   └── gemini/
│       ├── __init__.py
│       ├── gemini_service.py     # Core Gemini integration
│       ├── knowledge_base.py     # Gemini as semantic memory
│       └── reasoning_engine.py   # Gemini reasoning paths
├── utils/
│   ├── __init__.py
│   ├── text_processing.py
│   ├── vector_operations.py
│   └── neural_converters.py
├── tests/
│   ├── __init__.py
│   ├── test_integration.py
│   ├── test_reasoning.py
│   └── benchmarks/
│       ├── __init__.py
│       ├── latency_tests.py
│       └── accuracy_tests.py
├── examples/
│   ├── __init__.py
│   ├── simple_reasoning.py
│   ├── complex_problem.py
│   └── creative_task.py
└── main.py