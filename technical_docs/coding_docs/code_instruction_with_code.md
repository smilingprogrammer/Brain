# 🧠 Code Update Scenario: Instruction + Existing Code

## How the Brain Would Process Code Updates:

### Input Example:
```python
"""
Please optimize this function to handle edge cases and improve performance.
Also add proper error handling.

def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
"""
```

## Step-by-Step Cognitive Processing:

### 1. **Language Comprehension** → Dual Analysis
```python
Output: {
    "instruction_analysis": {
        "tasks": ["optimize", "handle edge cases", "add error handling"],
        "priority": ["performance", "robustness", "error_handling"]
    },
    "code_analysis": {
        "language": "python",
        "function_name": "find_duplicates",
        "complexity": "O(n²)",
        "issues_detected": ["nested loops", "inefficient duplicate check", "no error handling"]
    }
}
```

### 2. **Working Memory** → Maintains Both Contexts
```python
{
    "original_code": {...},
    "requirements": ["optimize", "edge cases", "error handling"],
    "identified_problems": [
        "O(n²) complexity",
        "No input validation",
        "Inefficient 'not in' check"
    ],
    "improvement_targets": ["algorithm", "validation", "error_handling"]
}
```

### 3. **Executive Planning** → Structured Update Plan
```python
Plan: {
    "analysis_phase": [
        {"task": "Understand current algorithm", "strategy": "logical"},
        {"task": "Identify inefficiencies", "strategy": "causal"}
    ],
    "improvement_phase": [
        {"task": "Find better algorithm", "strategy": "analogical"},
        {"task": "Add error handling", "strategy": "logical"},
        {"task": "Handle edge cases", "strategy": "creative"}
    ],
    "validation_phase": [
        {"task": "Verify correctness", "strategy": "logical"},
        {"task": "Check performance", "strategy": "causal"}
    ]
}
```

### 4. **Parallel Reasoning** → Multiple Improvement Strategies

**Logical Reasoning**: 
- "Current algorithm has O(n²) time complexity"
- "Using a set/dict would give O(n) complexity"
- "Need to handle None, empty array, non-hashable types"

**Causal Reasoning**:
- "Nested loops cause quadratic time"
- "Repeated 'not in' checks on list cause additional O(n) operations"
- "No type checking could cause runtime errors"

**Analogical Reasoning**:
- "Similar to 'two-sum' problem - use hash table"
- "Like database DISTINCT - use set operations"

**Creative Reasoning**:
- "What if we use Counter from collections?"
- "Could we handle unhashable types separately?"

### 5. **Global Workspace** → Integrated Solution
```python
Integration: {
    "best_approach": "Use dictionary for O(n) complexity",
    "edge_cases_identified": [
        "Empty input",
        "None input", 
        "Single element",
        "All duplicates",
        "Unhashable types (lists, dicts)"
    ],
    "error_handling_strategy": "Type hints + try-except for unhashable"
}
```

### 6. **Code Production** → Updated Code
```python
from typing import List, Any, Union
from collections import Counter

def find_duplicates(arr: Union[List[Any], None]) -> List[Any]:
    """
    Find all duplicate elements in an array.
    
    Args:
        arr: List of elements (hashable types)
        
    Returns:
        List of duplicate elements (each appears only once)
        
    Raises:
        TypeError: If array contains unhashable types
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Handle edge cases
    if arr is None:
        raise ValueError("Input array cannot be None")
    
    if len(arr) <= 1:
        return []
    
    try:
        # Use Counter for O(n) complexity
        element_counts = Counter(arr)
        
        # Return elements that appear more than once
        return [element for element, count in element_counts.items() if count > 1]
        
    except TypeError as e:
        # Handle unhashable types
        # Fallback to O(n²) for unhashable types with proper handling
        duplicates = []
        seen = []
        
        for item in arr:
            if item in seen and item not in duplicates:
                duplicates.append(item)
            elif item not in seen:
                seen.append(item)
                
        return duplicates

# Additional test cases suggested by creative reasoning
def test_find_duplicates():
    # Normal cases
    assert find_duplicates([1, 2, 3, 2, 4, 3]) == [2, 3]
    assert find_duplicates(['a', 'b', 'a', 'c']) == ['a']
    
    # Edge cases
    assert find_duplicates([]) == []
    assert find_duplicates([1]) == []
    assert find_duplicates([1, 1, 1, 1]) == [1]
    
    # Error cases
    try:
        find_duplicates(None)
    except ValueError:
        pass
    
    # Unhashable types
    result = find_duplicates([[1], [2], [1], [3]])
    assert result == [[1]]
```

## Key Cognitive Features for Code Updates:

### 1. **Dual-Context Processing**
- Comprehends both natural language instructions AND code
- Maintains relationship between requirements and implementation

### 2. **Code-Aware Reasoning**
```python
# The brain understands:
- Algorithm complexity (recognizes O(n²) pattern)
- Language idioms (Python's "in" operator cost)
- Best practices (type hints, docstrings)
- Common pitfalls (unhashable types in sets)
```

### 3. **Incremental Improvement Strategy**
```python
Working Memory tracks:
{
    "original_functionality": "find duplicates",
    "must_preserve": ["return format", "basic behavior"],
    "can_change": ["algorithm", "data structures"],
    "must_add": ["error handling", "edge cases"]
}
```

### 4. **Meta-Cognitive Code Review**
The meta-cognition module evaluates:
- Is the new code actually better?
- Did we introduce new bugs?
- Is it more maintainable?
- Are the trade-offs worth it?

### 5. **Semantic Memory Integration**
Recalls from knowledge base:
- Common optimization patterns
- Python-specific performance tips
- Error handling best practices
- Testing strategies

## Advanced Code Update Example:

```python
Input: """
Add caching to this recursive function and make it handle negative inputs:

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Cognitive Processing identifies:
1. Performance issue (exponential time)
2. Missing negative number handling
3. No memoization

# Produces:
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number with memoization.
    
    Handles negative indices using the formula: F(-n) = (-1)^(n+1) * F(n)
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n < 0:
        # Handle negative indices
        return ((-1) ** ((-n) + 1)) * fibonacci(-n)
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

The cognitive architecture excels at code updates because it:
- **Understands intent** from natural language
- **Analyzes existing code** structurally
- **Reasons about improvements** from multiple angles
- **Preserves functionality** while enhancing
- **Explains changes** it makes

This is much more sophisticated than simple find-and-replace or template-based code modification!