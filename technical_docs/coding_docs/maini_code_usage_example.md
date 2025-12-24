
## 4. Usage Examples

### Example 1: Syntax Validation
```python
# Input code with syntax error
code = """
def calculate_average(numbers)
    if not numbers:
        return 0
    total = sum(numbers)
    return total / len(numbers)
"""

result = await brain.process_code(code)

# Output:
{
    "type": "syntax_fix",
    "original_errors": [{
        "line": 1,
        "column": 30,
        "message": "expected ':'",
        "type": "SyntaxError"
    }],
    "fixes": [{
        "error": {...},
        "suggestion": "Add ':' at the end of the line",
        "fix_type": "missing_colon",
        "confidence": 0.95
    }],
    "suggestion": "Please fix syntax errors before proceeding"
}

### Example 2: Code Review
# Input code for review
code = """
def process_data(d):
    result = []
    for i in range(len(d)):
        for j in range(len(d[i])):
            if d[i][j] > 0:
                result.append(d[i][j])
    return result
"""

result = await brain.process_code(code)

# Output:
{
    "type": "code_review",
    "review": {
        "issues": [
            {
                "type": "readability",
                "severity": "medium",
                "message": "Single letter variables found: {'d', 'i', 'j'}",
                "suggestion": "Use descriptive variable names"
            },
            {
                "type": "performance",
                "severity": "medium",
                "message": "Nested loops with append operations detected",
                "suggestion": "Consider using list comprehension or numpy"
            },
            {
                "type": "documentation",
                "severity": "medium",
                "message": "Function 'process_data' lacks a docstring"
            }
        ],
        "metrics": {
            "readability": 60,
            "maintainability": 70,
            "performance": 50,
            "security": 100,
            "best_practices": 60,
            "documentation": 20
        },
        "code_smells": [{
            "type": "too_many_nested_loops",
            "severity": "medium"
        }]
    },
    "overall_score": 60.0,
    "improvements": [
        {
            "type": "refactoring",
            "priority": "medium",
            "suggestion": "Use list comprehension: [val for row in data for val in row if val > 0]",
            "benefits": ["Better performance", "More Pythonic", "Clearer intent"]
        }
    ]
}

# (Example 3: Code Update with Instruction)
# Input
code = """
def find_max(arr):
    max_val = arr[0]
    for i in range(len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
"""

instruction = "Add error handling and optimize this function"

result = await brain.process_code(code, instruction)

# Output:
{
    "type": "code_update",
    "instruction": "Add error handling and optimize this function",
    "updated_code": """
def find_max(arr):
    '''Find the maximum value in an array.
    
    Args:
        arr: List of comparable values
        
    Returns:
        Maximum value in the array
        
    Raises:
        ValueError: If array is empty
        TypeError: If array is None
    '''
    if arr is None:
        raise TypeError("Input array cannot be None")
    
    if not arr:
        raise ValueError("Cannot find max of empty array")
    
    # Use built-in max() for better performance
    return max(arr)
""",
    "validation": {
        "valid": True,
        "errors": [],
        "warnings": []
    },
    "explanation": "I've improved the function by:\n1. Adding comprehensive error handling for None and empty inputs\n2. Using Python's built-in max() function which is optimized in C\n3. Adding a detailed docstring\n4. The original O(n) algorithm is replaced with the built-in which is also O(n) but faster"
}
