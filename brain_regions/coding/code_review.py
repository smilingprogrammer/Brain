from typing import Dict, Any, List, Optional
import ast
import re
from core.interfaces import BrainRegion
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
from brain_regions.executive.meta_cognition import MetaCognition
import structlog

logger = structlog.get_logger()


class CodeReviewModule(BrainRegion):
    """Comprehensive code review with quality evaluation"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService, meta_cognition: MetaCognition):
        self.event_bus = event_bus
        self.gemini = gemini
        self.meta_cognition = meta_cognition

        # Review criteria
        self.review_criteria = {
            "readability": self._check_readability,
            "maintainability": self._check_maintainability,
            "performance": self._check_performance,
            "security": self._check_security,
            "best_practices": self._check_best_practices,
            "documentation": self._check_documentation
        }

        # Code smell patterns
        self.code_smells = {
            "long_method": {"threshold": 50, "severity": "medium"},
            "large_class": {"threshold": 300, "severity": "medium"},
            "too_many_parameters": {"threshold": 5, "severity": "low"},
            "duplicate_code": {"threshold": 10, "severity": "high"},
            "complex_conditionals": {"threshold": 5, "severity": "medium"}
        }

    async def initialize(self):
        """Initialize code review module"""
        logger.info("initializing_code_review_module")

        # Subscribe to review requests
        self.event_bus.subscribe("code_review_request", self._on_review_request)
        self.event_bus.subscribe("improvement_suggestion_request", self._on_improvement_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process code review request"""

        code = input_data.get("code", "")
        language = input_data.get("language", "python")
        review_type = input_data.get("review_type", "comprehensive")
        context = input_data.get("context", {})

        # Perform review
        review_result = await self._perform_review(code, language, review_type, context)

        # Meta-cognitive evaluation
        quality_assessment = await self._meta_evaluate_code(code, review_result)

        # Generate improvement suggestions
        improvements = await self._generate_improvements(code, review_result, quality_assessment)

        return {
            "success": True,
            "review": review_result,
            "quality_assessment": quality_assessment,
            "improvements": improvements,
            "overall_score": self._calculate_overall_score(review_result)
        }

    async def _perform_review(self, code: str, language: str, review_type: str, context: Dict) -> Dict:
        """Perform comprehensive code review"""

        review_results = {
            "issues": [],
            "suggestions": [],
            "metrics": {},
            "positive_aspects": []
        }

        # Run all review criteria
        for criterion, checker in self.review_criteria.items():
            if review_type == "comprehensive" or criterion in review_type:
                result = await checker(code, language, context)

                review_results["issues"].extend(result.get("issues", []))
                review_results["suggestions"].extend(result.get("suggestions", []))
                review_results["metrics"][criterion] = result.get("score", 0)
                review_results["positive_aspects"].extend(result.get("positive", []))

        # Check for code smells
        smells = await self._detect_code_smells(code, language)
        review_results["code_smells"] = smells

        return review_results

    async def _check_readability(self, code: str, language: str, context: Dict) -> Dict:
        """Check code readability"""

        issues = []
        suggestions = []
        positive = []

        # Variable naming
        if language == "python":
            # Check for single letter variables (except common ones)
            single_letter_vars = re.findall(r'\b[a-z]\b(?![_\w])', code)
            allowed = ['i', 'j', 'k', 'x', 'y', 'z', 'n', 'm']
            bad_vars = [v for v in single_letter_vars if v not in allowed]

            if bad_vars:
                issues.append({
                    "type": "readability",
                    "severity": "medium",
                    "message": f"Single letter variables found: {set(bad_vars)}",
                    "suggestion": "Use descriptive variable names"
                })

            # Check for clear function names
            function_names = re.findall(r'def\s+(\w+)', code)
            for func_name in function_names:
                if len(func_name) < 3:
                    issues.append({
                        "type": "readability",
                        "severity": "medium",
                        "message": f"Function name '{func_name}' is too short",
                        "line": self._find_line_number(code, f"def {func_name}")
                    })
                elif func_name.islower() and '_' in func_name:
                    positive.append(f"Good function naming: {func_name}")

        # Line length
        lines = code.split('\n')
        long_lines = [(i + 1, len(line)) for i, line in enumerate(lines) if len(line) > 100]

        if long_lines:
            for line_num, length in long_lines[:5]:  # Report first 5
                issues.append({
                    "type": "readability",
                    "severity": "low",
                    "message": f"Line {line_num} is too long ({length} chars)",
                    "suggestion": "Break into multiple lines"
                })

        # Calculate readability score
        score = 100
        score -= len(issues) * 10
        score = max(0, min(100, score))

        return {
            "issues": issues,
            "suggestions": suggestions,
            "positive": positive,
            "score": score
        }

    async def _check_maintainability(self, code: str, language: str, context: Dict) -> Dict:
        """Check code maintainability"""

        issues = []
        suggestions = []

        # Use Gemini for deeper analysis
        prompt = f"""Analyze this {language} code for maintainability:

            {code[:1000]}...

            Check for:
            1. Modularity and separation of concerns
            2. Code coupling and cohesion
            3. Ease of testing
            4. Extensibility

            Provide specific issues and suggestions."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        if response["success"]:
            # Parse Gemini's analysis
            analysis = self._parse_gemini_analysis(response["text"])
            issues.extend(analysis.get("issues", []))
            suggestions.extend(analysis.get("suggestions", []))

        # Check cyclomatic complexity
        if language == "python":
            complexity = self._calculate_cyclomatic_complexity(code)
            if complexity > 10:
                issues.append({
                    "type": "maintainability",
                    "severity": "high",
                    "message": f"High cyclomatic complexity: {complexity}",
                    "suggestion": "Refactor complex logic into smaller functions"
                })

        score = 100 - (len(issues) * 15)
        return {
            "issues": issues,
            "suggestions": suggestions,
            "score": max(0, min(100, score))
        }

    async def _check_performance(self, code: str, language: str, context: Dict) -> Dict:
        """Check for performance issues"""

        issues = []
        suggestions = []

        if language == "python":
            # Check for common performance anti-patterns

            # Nested loops with list operations
            if re.search(r'for .+ in .+:\s*for .+ in .+:.*\.appendKATEX_INLINE_OPEN', code, re.MULTILINE):
                issues.append({
                    "type": "performance",
                    "severity": "medium",
                    "message": "Nested loops with append operations detected",
                    "suggestion": "Consider using list comprehension or numpy for better performance"
                })

            # String concatenation in loops
            if re.search(r'for .+ in .+:.*\+=.*["\']', code):
                issues.append({
                    "type": "performance",
                    "severity": "medium",
                    "message": "String concatenation in loop detected",
                    "suggestion": "Use join() or list append for better performance"
                })

            # Repeated regex compilation
            if code.count('re.search') + code.count('re.match') > 3:
                if not re.search(r're\.compile', code):
                    suggestions.append({
                        "type": "performance",
                        "message": "Consider pre-compiling regex patterns used multiple times"
                    })

        score = 100 - (len(issues) * 20)
        return {
            "issues": issues,
            "suggestions": suggestions,
            "score": max(0, min(100, score))
        }

    async def _check_security(self, code: str, language: str, context: Dict) -> Dict:
        """Check for security vulnerabilities"""

        issues = []

        # Common security checks
        security_patterns = {
            "eval": {
                "pattern": r'\beval\s*KATEX_INLINE_OPEN',
                "message": "Use of eval() is a security risk",
                "severity": "high"
            },
            "exec": {
                "pattern": r'\bexec\s*KATEX_INLINE_OPEN',
                "message": "Use of exec() is a security risk",
                "severity": "high"
            },
            "pickle": {
                "pattern": r'pickle\.loads?\s*KATEX_INLINE_OPEN',
                "message": "Unpickling untrusted data is dangerous",
                "severity": "high"
            },
            "sql_injection": {
                "pattern": r'\".*SELECT.*WHERE.*\%s.*\"',
                "message": "Potential SQL injection vulnerability",
                "severity": "critical"
            }
        }

        for vuln_type, config in security_patterns.items():
            if re.search(config["pattern"], code):
                issues.append({
                    "type": "security",
                    "severity": config["severity"],
                    "message": config["message"],
                    "vulnerability": vuln_type
                })

        score = 100 if not issues else max(0, 100 - (len(issues) * 30))
        return {
            "issues": issues,
            "suggestions": [],
            "score": score
        }

    async def _check_best_practices(self, code: str, language: str, context: Dict) -> Dict:
        """Check adherence to best practices"""

        issues = []
        positive = []

        if language == "python":
            # Check for type hints
            has_type_hints = bool(re.search(r'def \w+KATEX_INLINE_OPEN.*:.*KATEX_INLINE_CLOSE', code))
            if has_type_hints:
                positive.append("Good use of type hints")
            else:
                issues.append({
                    "type": "best_practice",
                    "severity": "low",
                    "message": "Consider adding type hints for better code clarity"
                })

            # Check for docstrings
            functions = re.findall(r'def (\w+)KATEX_INLINE_OPEN.*?KATEX_INLINE_CLOSE:', code)
            for func in functions:
                func_pattern = f'def {func}\KATEX_INLINE_OPEN.*?\KATEX_INLINE_CLOSE:.*?"""'
                if not re.search(func_pattern, code, re.DOTALL):
                    issues.append({
                        "type": "best_practice",
                        "severity": "medium",
                        "message": f"Function '{func}' lacks a docstring"
                    })

            # Check for proper exception handling
            if 'except:' in code or 'except Exception:' in code:
                issues.append({
                    "type": "best_practice",
                    "severity": "medium",
                    "message": "Avoid bare except or catching Exception",
                    "suggestion": "Catch specific exceptions"
                })

        score = 100 - (len(issues) * 10)
        return {
            "issues": issues,
            "suggestions": [],
            "positive": positive,
            "score": max(0, min(100, score))
        }

    async def _check_documentation(self, code: str, language: str, context: Dict) -> Dict:
        """Check documentation quality"""

        issues = []
        positive = []

        # Count docstrings
        docstring_count = code.count('"""')
        function_count = code.count('def ')
        class_count = code.count('class ')

        expected_docstrings = function_count + class_count

        if docstring_count < expected_docstrings:
            issues.append({
                "type": "documentation",
                "severity": "medium",
                "message": f"Missing docstrings: found {docstring_count // 2}, expected at least {expected_docstrings}"
            })
        elif docstring_count >= expected_docstrings * 2:
            positive.append("Excellent documentation coverage")

        # Check for inline comments
        comment_lines = [line for line in code.split('\n') if '#' in line and not line.strip().startswith('#')]
        if len(comment_lines) > 0:
            positive.append("Good use of inline comments")

        score = 100
        if expected_docstrings > 0:
            score = int((docstring_count / (expected_docstrings * 2)) * 100)

        return {
            "issues": issues,
            "suggestions": [],
            "positive": positive,
            "score": min(100, score)
        }

    async def _detect_code_smells(self, code: str, language: str) -> List[Dict]:
        """Detect common code smells"""

        smells = []

        if language == "python":
            # Long methods
            methods = re.findall(r'def \w+KATEX_INLINE_OPEN.*?KATEX_INLINE_CLOSE:.*?(?=def|\Z)', code, re.DOTALL)
            for method in methods:
                line_count = method.count('\n')
                if line_count > self.code_smells["long_method"]["threshold"]:
                    smells.append({
                        "type": "long_method",
                        "severity": self.code_smells["long_method"]["severity"],
                        "lines": line_count
                    })

            # Too many parameters
            param_patterns = re.findall(r'def \w+KATEX_INLINE_OPEN(.*?)KATEX_INLINE_CLOSE:', code)
            for params in param_patterns:
                param_count = len([p.strip() for p in params.split(',') if p.strip()])
                if param_count > self.code_smells["too_many_parameters"]["threshold"]:
                    smells.append({
                        "type": "too_many_parameters",
                        "severity": self.code_smells["too_many_parameters"]["severity"],
                        "count": param_count
                    })

        return smells

    async def _meta_evaluate_code(self, code: str, review_result: Dict) -> Dict:
        """Meta-cognitive evaluation of code quality"""

        # Use meta-cognition to evaluate overall code quality
        evaluation_context = {
            "code_length": len(code.split('\n')),
            "issue_count": len(review_result.get("issues", [])),
            "metrics": review_result.get("metrics", {}),
            "code_smells": len(review_result.get("code_smells", []))
        }

        meta_assessment = await self.meta_cognition.process({
            "operation": "evaluate_code_quality",
            "context": evaluation_context
        })

        # Generate quality insights
        quality_insights = {
            "overall_health": self._determine_health_status(review_result),
            "maintainability_risk": self._assess_maintainability_risk(review_result),
            "technical_debt": self._estimate_technical_debt(review_result),
            "refactoring_priority": self._calculate_refactoring_priority(review_result)
        }

        return {
            "meta_assessment": meta_assessment,
            "quality_insights": quality_insights,
            "confidence": 0.85
        }

    async def _generate_improvements(self, code: str, review_result: Dict, quality_assessment: Dict) -> List[Dict]:
        """Generate specific improvement suggestions"""

        improvements = []

        # Prioritize based on severity
        high_priority_issues = [
            issue for issue in review_result.get("issues", [])
            if issue.get("severity") in ["high", "critical"]
        ]

        for issue in high_priority_issues[:5]:  # Top 5 high priority
            improvement = await self._create_improvement_suggestion(code, issue)
            if improvement:
                improvements.append(improvement)

        # Add refactoring suggestions for code smells
        for smell in review_result.get("code_smells", [])[:3]:
            if smell["type"] == "long_method":
                improvements.append({
                    "type": "refactoring",
                    "priority": "medium",
                    "suggestion": "Extract method to break down long function",
                    "pattern": "Extract Method",
                    "benefits": ["Improved readability", "Better testability", "Easier maintenance"]
                })
            elif smell["type"] == "too_many_parameters":
                improvements.append({
                    "type": "refactoring",
                    "priority": "medium",
                    "suggestion": "Introduce parameter object or use builder pattern",
                    "pattern": "Parameter Object",
                    "benefits": ["Cleaner interface", "Easier to extend", "Better encapsulation"]
                })

        return improvements

    async def _create_improvement_suggestion(self, code: str, issue: Dict) -> Optional[Dict]:
        """Create specific improvement suggestion for an issue"""

        suggestion_prompt = f"""Given this code issue:
            Type: {issue.get('type')}
            Message: {issue.get('message')}
            Severity: {issue.get('severity')}

            Suggest a specific improvement with:
            1. What to change
            2. How to change it
            3. Example of the improved code
            4. Benefits of the change"""

        response = await self.gemini.generate(suggestion_prompt, config_name="balanced")

        if response["success"]:
            return {
                "issue": issue,
                "improvement": response["text"],
                "priority": issue.get("severity", "medium")
            }

        return None

    def _calculate_overall_score(self, review_result: Dict) -> float:
        """Calculate overall code quality score"""

        metrics = review_result.get("metrics", {})
        if not metrics:
            return 50.0

        # Weighted average of all metrics
        weights = {
            "readability": 0.25,
            "maintainability": 0.25,
            "performance": 0.15,
            "security": 0.20,
            "best_practices": 0.10,
            "documentation": 0.05
        }

        total_score = 0
        total_weight = 0

        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 50.0

    def _determine_health_status(self, review_result: Dict) -> str:
        """Determine overall code health status"""

        issue_count = len(review_result.get("issues", []))
        critical_issues = len([i for i in review_result.get("issues", []) if i.get("severity") == "critical"])

        if critical_issues > 0:
            return "critical"
        elif issue_count > 20:
            return "poor"
        elif issue_count > 10:
            return "fair"
        elif issue_count > 5:
            return "good"
        else:
            return "excellent"

    def _assess_maintainability_risk(self, review_result: Dict) -> str:
        """Assess maintainability risk level"""

        maintainability_score = review_result.get("metrics", {}).get("maintainability", 50)
        smell_count = len(review_result.get("code_smells", []))

        if maintainability_score < 30 or smell_count > 5:
            return "high"
        elif maintainability_score < 60 or smell_count > 2:
            return "medium"
        else:
            return "low"

    def _estimate_technical_debt(self, review_result: Dict) -> Dict:
        """Estimate technical debt"""

        # Simple estimation based on issues
        issue_count = len(review_result.get("issues", []))
        smell_count = len(review_result.get("code_smells", []))

        # Rough time estimates
        time_per_issue = {
            "low": 0.5,  # hours
            "medium": 2,
            "high": 4,
            "critical": 8
        }

        total_hours = 0
        for issue in review_result.get("issues", []):
            severity = issue.get("severity", "medium")
            total_hours += time_per_issue.get(severity, 2)

        # Add time for code smells
        total_hours += smell_count * 3

        return {
            "estimated_hours": total_hours,
            "debt_level": "high" if total_hours > 40 else "medium" if total_hours > 20 else "low",
            "priority_items": issue_count + smell_count
        }

    def _calculate_refactoring_priority(self, review_result: Dict) -> float:
        """Calculate refactoring priority score (0-100)"""

        # Factors affecting priority
        metrics = review_result.get("metrics", {})
        maintainability = metrics.get("maintainability", 50)
        readability = metrics.get("readability", 50)
        security = metrics.get("security", 100)

        # Critical issues boost priority
        critical_count = len([i for i in review_result.get("issues", []) if i.get("severity") == "critical"])

        # Calculate priority
        base_priority = (100 - maintainability) * 0.4 + (100 - readability) * 0.3 + (100 - security) * 0.3
        priority = min(100, base_priority + (critical_count * 10))

        return priority

    def _find_line_number(self, code: str, pattern: str) -> int:
        """Find line number for a pattern in code"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern in line:
                return i + 1
        return 0

    def _parse_gemini_analysis(self, text: str) -> Dict:
        """Parse Gemini's analysis into structured format"""

        issues = []
        suggestions = []

        lines = text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "issue" in line.lower() or "problem" in line.lower():
                current_section = "issues"
            elif "suggestion" in line.lower() or "recommend" in line.lower():
                current_section = "suggestions"
            elif current_section == "issues" and line.startswith('-'):
                issues.append({
                    "type": "maintainability",
                    "severity": "medium",
                    "message": line.lstrip('- ')
                })
            elif current_section == "suggestions" and line.startswith('-'):
                suggestions.append({
                    "type": "improvement",
                    "message": line.lstrip('- ')
                })

        return {"issues": issues, "suggestions": suggestions}

    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity for Python code"""

        # Simple approximation
        complexity = 1  # Base complexity

        # Add complexity for control structures
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('except ')
        complexity += code.count(' and ')
        complexity += code.count(' or ')

        return complexity

    async def _on_review_request(self, data: Dict):
        """Handle code review request"""
        result = await self.process(data)
        await self.event_bus.emit("code_review_complete", result)

    async def _on_improvement_request(self, data: Dict):
        """Handle improvement suggestion request"""
        code = data.get("code", "")
        language = data.get("language", "python")

        # Quick review for improvements
        review = await self._perform_review(code, language, "quick", {})
        improvements = await self._generate_improvements(code, review, {})

        await self.event_bus.emit("improvements_generated", {
            "improvements": improvements,
            "code": code
        })

    def get_state(self) -> Dict[str, Any]:
        return {
            "review_criteria": list(self.review_criteria.keys()),
            "code_smell_types": list(self.code_smells.keys()),
            "active": True
        }