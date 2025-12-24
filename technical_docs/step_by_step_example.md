# 🧠 Step-by-Step Brain Processing Flow with Examples

Here's how data flows through each brain region, with concrete input/output examples:

## Step 1: Text Input → Language Comprehension

**Input:**
```python
text_input = "If water suddenly became twice as dense, what would happen to marine life?"
```

**Process:**
```python
language = LanguageComprehension(event_bus)
result = await language.process({"text": text_input})
```

**Output:**
```python
{
    "original_text": "If water suddenly became twice as dense, what would happen to marine life?",
    "tokens": ["If", "water", "suddenly", "became", "twice", "as", "dense", ",", "what", "would", "happen", "to", "marine", "life", "?"],
    "entities": [("water", "SUBSTANCE"), ("marine life", "NOUN")],
    "complexity_score": 0.7,
    "embedding": [0.234, -0.567, 0.891, ...],  # 768-dim vector
    "dependencies": [("became", "ROOT", "root"), ("water", "nsubj", "became"), ("dense", "acomp", "became")]
}
```

## Step 2: Language Output → Working Memory

**Input:** (from Step 1)
```python
language_output = {
    "original_text": "If water suddenly became twice as dense...",
    "entities": [("water", "SUBSTANCE"), ("marine life", "NOUN")],
    "embedding": [0.234, -0.567, 0.891, ...],
    "complexity_score": 0.7
}
```

**Process:**
```python
working_memory = WorkingMemory(event_bus, gemini)
await working_memory.store({
    "type": "language_comprehension",
    "content": language_output,
    "salience": 0.85  # High due to novelty and complexity
})
```

**Output:**
```python
{
    "current_contents": [
        {
            "type": "language_comprehension",
            "content": {...language_output...},
            "timestamp": 1234567890.123,
            "salience": 0.85,
            "access_count": 0
        }
    ],
    "attention_focus": {
        "focused_item": {...},
        "attention_weight": 0.85
    },
    "capacity_used": 0.14  # 1/7 slots used
}
```

## Step 3: Working Memory → Prefrontal Cortex (Executive Planning)

**Input:** (from Step 2)
```python
working_memory_state = {
    "current_contents": [{...}],
    "attention_focus": {"focused_item": {...}}
}

task_input = {
    "task": "If water suddenly became twice as dense, what would happen to marine life?",
    "context": {
        "working_memory_summary": "Question about water density change and marine life impact",
        "complexity": 0.7
    }
}
```

**Process:**
```python
prefrontal = PrefrontalCortex(event_bus, gemini)
plan = await prefrontal._generate_plan(task_input["task"], task_input["context"])
```

**Output:**
```python
{
    "main_goal": "Analyze impact of water density change on marine life",
    "sub_goals": [
        {
            "goal": "Understand physics of density change",
            "strategy": "logical",
            "required_regions": ["logical_reasoning", "semantic_memory"],
            "priority": 1
        },
        {
            "goal": "Analyze buoyancy effects on marine organisms",
            "strategy": "causal",
            "required_regions": ["causal_reasoning", "semantic_memory"],
            "priority": 2
        },
        {
            "goal": "Predict ecosystem-wide impacts",
            "strategy": "creative",
            "required_regions": ["creative_reasoning", "analogical_reasoning"],
            "priority": 3
        }
    ],
    "success_criteria": ["Physics explained", "Biological impacts identified"],
    "contingency_plans": ["Use analogies from high-pressure environments"]
}
```

## Step 4: Executive Plan → Multiple Reasoning Paths (Parallel)

### Step 4a: Logical Reasoning Path

**Input:** (from Step 3, sub_goal 1)
```python
logical_input = {
    "problem": "Understand physics of water density doubling",
    "context": {"working_memory": {...}}
}
```

**Process:**
```python
logical = LogicalReasoning(event_bus, gemini)
logical_result = await logical.reason(logical_input["problem"], logical_input["context"])
```

**Output:**
```python
{
    "success": True,
    "conclusion": "Water pressure doubles at all depths (P = ρgh, where ρ doubles)",
    "proof_steps": [
        {"statement": "Given: ρ_new = 2 × ρ_original"},
        {"statement": "P = ρgh, therefore P_new = 2ρgh = 2P_original"},
        {"statement": "Buoyant force F_b = ρ × V × g also doubles"}
    ],
    "confidence": 0.95
}
```

### Step 4b: Causal Reasoning Path (Parallel)

**Input:** (from Step 3, sub_goal 2)
```python
causal_input = {
    "scenario": "Water density doubles",
    "query": "Effects on marine organisms"
}
```

**Process:**
```python
causal = CausalReasoning(event_bus, gemini)
causal_result = await causal.process(causal_input)
```

**Output:**
```python
{
    "success": True,
    "causal_chains": [
        {
            "path": ["density doubles", "pressure doubles", "swim bladders compress", "fish sink"],
            "strength": 0.9,
            "type": "direct"
        }
    ],
    "predictions": [
        {
            "outcome": "Mass extinction of surface fish",
            "timeframe": "immediate",
            "probability": 0.95
        }
    ],
    "confidence": 0.85
}
```

## Step 5: Reasoning Results → Global Workspace Integration

**Input:** (from Steps 4a and 4b)
```python
competing_inputs = {
    "logical_reasoning": {
        "data": {"conclusion": "Pressure doubles at all depths"},
        "confidence": 0.95,
        "salience": 0.85
    },
    "causal_reasoning": {
        "data": {"impact": "Catastrophic for marine life"},
        "confidence": 0.85,
        "salience": 0.92
    },
    "working_memory": {
        "data": {"context": "Sudden change scenario"},
        "confidence": 0.8,
        "salience": 0.75
    }
}
```

**Process:**
```python
global_workspace = GlobalWorkspace(event_bus)
# Add each to competition
for region, data in competing_inputs.items():
    await global_workspace._add_to_competition(region, data["data"])

# Run competition and integrate
winners = await global_workspace._run_attention_competition()
integrated = await global_workspace._integrate_representations(winners)
```

**Output:**
```python
{
    "primary_focus": {
        "region": "causal_reasoning",
        "content": "Catastrophic cascade: buoyancy failure → sinking → pressure death"
    },
    "secondary_elements": [
        {
            "region": "logical_reasoning",
            "content": "Pressure doubles: 10m = old 20m",
            "relevance": 0.31
        }
    ],
    "integrated_meaning": "Immediate catastrophic impact on marine life due to physics of doubled pressure",
    "confidence": 0.88
}
```

## Step 6: Integrated State → Semantic Memory Storage

**Input:** (from Step 5)
```python
semantic_input = {
    "content": {
        "concept": "water_density_impact",
        "key_findings": "Doubling water density causes immediate marine extinction",
        "causal_model": {...integrated_state...}
    },
    "metadata": {"source": "reasoning_synthesis", "confidence": 0.88}
}
```

**Process:**
```python
semantic_cortex = SemanticCortex(event_bus, gemini)
await semantic_cortex.store(semantic_input["content"], semantic_input["metadata"])
```

**Output:**
```python
{
    "concept_id": "water_density_marine_impact_001",
    "stored": True,
    "relationships": [
        ("causes", "pressure_increase"),
        ("affects", "marine_life"),
        ("results_in", "extinction_event")
    ]
}
```

## Step 7: All Results → Language Production

**Input:** (from Steps 5 & 6)
```python
production_input = {
    "intent": "explain",
    "content": {
        "integrated_state": {...from_global_workspace...},
        "semantic_knowledge": {...from_semantic_cortex...}
    },
    "style": "clear"
}
```

**Process:**
```python
language_production = LanguageProduction(event_bus, gemini)
response = await language_production.process(production_input)
```

**Output:**
```python
{
    "success": True,
    "text": "If water suddenly became twice as dense, it would be catastrophic for marine life:\n\n**Immediate Physical Effects:**\n- Water pressure would double at every depth\n- All buoyancy forces would double\n\n**Impact on Marine Life:**\n- Fish swim bladders would compress to half size\n- Most fish would sink uncontrollably\n- Fatal pressure increases would occur\n\n**Ecosystem Collapse:**\n- Mass extinction of surface species\n- Only deep-sea adapted creatures might survive",
    "intent": "explain",
    "style": "clear"
}
```

## Step 8: Final Response → Motor Output

**Input:** (from Step 7)
```python
motor_input = {
    "action": "generate_text_response",
    "content": response["text"]
}
```

**Process:**
```python
motor = MotorModule(event_bus)
await motor.execute(motor_input)
```

**Output:**
```python
{
    "success": True,
    "action_executed": "text_output",
    "output": "If water suddenly became twice as dense, it would be catastrophic for marine life...",
    "execution_time": 3.2  # seconds
}
```

## Complete Flow Summary

```
1. Text → Language Comprehension → {tokens, entities, embedding}
2. Language Output → Working Memory → {stored context, salience}
3. Working Memory → Executive Planning → {goals, strategies}
4. Plan → Parallel Reasoning → {logical + causal + creative results}
5. All Results → Global Workspace → {integrated understanding}
6. Integrated State → Semantic Memory → {stored knowledge}
7. All Knowledge → Language Production → {natural language response}
8. Response → Motor Output → {final text to user}
```

Each step transforms the data, adding layers of understanding until a complete, reasoned response emerges. The parallel processing in Step 4 and the integration in Step 5 are key to the system's advanced reasoning capabilities.