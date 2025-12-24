Great question! Currently, the system requires the `code:` prefix to explicitly trigger code processing. However, we can definitely make it automatically detect code. Let me show you both approaches:

## Current Implementation (Explicit):
```python
# User must type:
You: code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)

# Or for code with instruction:
You: code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)
Please add error handling and optimize this function
```

## Enhanced Implementation (Automatic Detection):

Here's how we can modify `main_complete.py` to automatically detect code:

```python
def detect_code_in_text(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Automatically detect if input contains code"""
    
    # Code indicators
    code_patterns = [
        r'def\s+\w+\s*\(',           # Python function
        r'class\s+\w+\s*[\(:]',      # Python class
        r'function\s+\w+\s*\(',      # JavaScript function
        r'const\s+\w+\s*=',          # JavaScript const
        r'public\s+class\s+\w+',     # Java class
        r'#include\s*<',             # C++ include
        r'import\s+\w+',             # Python import
        r'from\s+\w+\s+import',      # Python from import
        r'if\s*\(.*\)\s*{',         # C-style if
        r'for\s*\(.*\)\s*{',        # C-style for
        r'while\s*\(.*\)\s*{',      # C-style while
        r'```[\w]*\n',              # Markdown code block
        r'return\s+\w+',            # Return statement
        r'print\s*\(',              # Print function
        r'console\.log\s*\(',       # JavaScript console
    ]
    
    # Check if text contains code
    for pattern in code_patterns:
        if re.search(pattern, text, re.MULTILINE):
            # Extract code and instruction
            code, instruction = self._extract_code_and_instruction(text)
            return True, code, instruction
    
    # Check for multi-line structure that looks like code
    lines = text.strip().split('\n')
    if len(lines) > 2:
        # Check indentation patterns
        indented_lines = sum(1 for line in lines if line.startswith((' ', '\t')))
        if indented_lines / len(lines) > 0.3:  # 30% indented lines
            code, instruction = self._extract_code_and_instruction(text)
            return True, code, instruction
    
    return False, None, None

def _extract_code_and_instruction(self, text: str) -> Tuple[str, Optional[str]]:
    """Extract code and optional instruction from mixed text"""
    
    # Check for markdown code blocks first
    code_block_pattern = r'```(?:\w+)?\n(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        code = matches[0].strip()
        # Remove code block from text to get instruction
        instruction = re.sub(code_block_pattern, '', text, flags=re.DOTALL).strip()
        return code, instruction if instruction else None
    
    # Check for natural language before/after code
    lines = text.strip().split('\n')
    
    # Common instruction patterns
    instruction_keywords = [
        'please', 'can you', 'could you', 'help', 'fix', 'improve', 
        'optimize', 'refactor', 'add', 'modify', 'explain', 'review'
    ]
    
    # Find where code likely starts and ends
    code_start = 0
    code_end = len(lines)
    instruction_lines = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in instruction_keywords):
            if i < len(lines) / 2:  # Instruction at beginning
                code_start = i + 1
                instruction_lines.extend(lines[:i+1])
            else:  # Instruction at end
                code_end = i
                instruction_lines.extend(lines[i:])
    
    # Extract code and instruction
    code_lines = lines[code_start:code_end]
    code = '\n'.join(code_lines).strip()
    instruction = '\n'.join(instruction_lines).strip() if instruction_lines else None
    
    return code, instruction

# Modified process_text method in CompleteCognitiveTextBrain
async def process_text(self, text: str) -> str:
    """Process text through complete cognitive pipeline"""
    
    if not self.initialized:
        await self.initialize()
    
    # Auto-detect code
    is_code, code, instruction = self.detect_code_in_text(text)
    
    if is_code:
        # Process as code
        logger.info("auto_detected_code", has_instruction=bool(instruction))
        result = await self.process_code(code, instruction)
        
        # Format code result as text response
        if result["type"] == "syntax_fix":
            return f"I detected syntax errors in your code:\n\n{format_code_result(result)}"
        elif result["type"] == "code_review":
            return f"Here's my analysis of your code:\n\n{format_code_result(result)}"
        elif result["type"] == "code_update":
            return f"I've updated your code based on your request:\n\n{result['explanation']}"
        else:
            return format_code_result(result)
    
    # Continue with normal text processing
    start_time = time.time()
    logger.info("processing_text_input", text_length=len(text))
    
    # ... rest of the original process_text implementation ...
```

## Usage Examples with Auto-Detection:

### 1. Plain Code (Auto-Detected)
```python
You: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)

Brain: Here's my analysis of your code:

Code Review Score: 75.0/100

Issues Found:
  [medium] Function 'factorial' lacks a docstring
  [low] Consider adding type hints for better code clarity
  [medium] No error handling for negative numbers

Top Improvements:
  - Add docstring explaining the function
  - Add type hints: def factorial(n: int) -> int
  - Handle edge case when n is negative
```

### 2. Code with Natural Language (Auto-Detected)
```python
You: Can you help me optimize this function?
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates

Brain: I've updated your code based on your request:

The original function has O(n²) time complexity due to nested loops. Here's an optimized version using a set for O(n) complexity:

```python
def find_duplicates(arr):
    """Find all duplicate elements in an array.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = set()
    duplicates = set()
    
    for item in arr:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)
```

### 3. Markdown Code Blocks (Auto-Detected)
```python
You: Please review this code:
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
```

Brain: Here's my analysis of your code:
[Review details...]
```

### 4. Mixed Content (Auto-Detected)
```python
You: I'm learning Python and wrote this:
x = 10
if x > 5:
print("x is greater than 5")

Brain: I detected syntax errors in your code:

Syntax Errors Found:
  Line 3: expected an indented block
  
Suggested Fixes:
  - Fix indentation to match surrounding code
```

## Benefits of Auto-Detection:

1. **Natural Interaction**: Users can paste code naturally without remembering special syntax
2. **Context Preservation**: Instructions and code are automatically separated
3. **Flexible Input**: Supports various code formats (plain, markdown, mixed with text)
4. **Fallback**: If code detection fails, it processes as normal text (no harm done)

Would you like me to provide the complete updated `main_complete.py` with this auto-detection feature integrated?