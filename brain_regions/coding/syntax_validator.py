import ast
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from pylsp import lsp
from core.interfaces import BrainRegion
from core.event_bus import EventBus
import structlog

logger = structlog.get_logger()


class SyntaxValidator(BrainRegion):
    """Syntax validation and language server integration"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.language_servers = {}
        self.syntax_cache = {}

        # Language-specific validators
        self.validators = {
            "python": self._validate_python,
            "javascript": self._validate_javascript,
            "java": self._validate_java,
            "cpp": self._validate_cpp
        }

        # LSP configurations
        self.lsp_configs = {
            "python": {"cmd": ["pylsp"], "port": 2087},
            "javascript": {"cmd": ["typescript-language-server", "--stdio"]},
            "java": {"cmd": ["jdtls"]},
            "cpp": {"cmd": ["clangd"]}
        }

    async def initialize(self):
        """Initialize syntax validator and language servers"""
        logger.info("initializing_syntax_validator")

        # Subscribe to code validation requests
        self.event_bus.subscribe("validate_syntax", self._on_validation_request)
        self.event_bus.subscribe("code_analysis_request", self._on_analysis_request)

        # Start language servers for common languages
        await self._initialize_language_servers(["python", "javascript"])

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process syntax validation request"""

        code = input_data.get("code", "")
        language = input_data.get("language", "python")
        mode = input_data.get("mode", "validate")  # validate, analyze, fix

        if mode == "validate":
            return await self._validate_syntax(code, language)
        elif mode == "analyze":
            return await self._analyze_code(code, language)
        elif mode == "fix":
            return await self._suggest_fixes(code, language)

        return {"success": False, "error": "Unknown mode"}

    async def _validate_syntax(self, code: str, language: str) -> Dict:
        """Validate syntax for given code"""

        # Check cache
        cache_key = f"{language}:{hash(code)}"
        if cache_key in self.syntax_cache:
            return self.syntax_cache[cache_key]

        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "language": language
        }

        # Use language-specific validator
        if language in self.validators:
            validation = await self.validators[language](code)
            result.update(validation)
        else:
            result["error"] = f"Unsupported language: {language}"
            result["valid"] = False

        # Cache result
        self.syntax_cache[cache_key] = result

        # Emit validation complete
        await self.event_bus.emit("syntax_validation_complete", result)

        return result

    async def _validate_python(self, code: str) -> Dict:
        """Validate Python syntax"""

        errors = []
        warnings = []

        try:
            # First pass: AST parsing
            tree = ast.parse(code)

            # Analyze AST for common issues
            analyzer = PythonCodeAnalyzer()
            issues = analyzer.analyze(tree)
            warnings.extend(issues.get("warnings", []))

            # Second pass: pyflakes/pylint style checks
            if self.language_servers.get("python"):
                lsp_diagnostics = await self._get_lsp_diagnostics("python", code)
                for diag in lsp_diagnostics:
                    if diag["severity"] == 1:  # Error
                        errors.append(self._format_diagnostic(diag))
                    else:  # Warning
                        warnings.append(self._format_diagnostic(diag))

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "ast": True
            }

        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [{
                    "line": e.lineno,
                    "column": e.offset,
                    "message": str(e.msg),
                    "type": "SyntaxError"
                }],
                "warnings": warnings,
                "ast": False
            }

    async def _analyze_code(self, code: str, language: str) -> Dict:
        """Deep code analysis using language server"""

        if language not in self.language_servers:
            await self._start_language_server(language)

        analysis = {
            "complexity": await self._calculate_complexity(code, language),
            "symbols": await self._extract_symbols(code, language),
            "dependencies": await self._analyze_dependencies(code, language),
            "suggestions": await self._get_suggestions(code, language)
        }

        return {
            "success": True,
            "language": language,
            "analysis": analysis
        }

    async def _suggest_fixes(self, code: str, language: str) -> Dict:
        """Suggest fixes for syntax errors"""

        # First validate to find errors
        validation = await self._validate_syntax(code, language)

        if validation["valid"]:
            return {
                "success": True,
                "message": "No syntax errors found",
                "fixes": []
            }

        fixes = []

        for error in validation["errors"]:
            fix = await self._generate_fix(code, error, language)
            if fix:
                fixes.append(fix)

        return {
            "success": True,
            "errors": validation["errors"],
            "fixes": fixes
        }

    async def _generate_fix(self, code: str, error: Dict, language: str) -> Optional[Dict]:
        """Generate fix suggestion for specific error"""

        if language == "python":
            return self._generate_python_fix(code, error)

        # Generic fix suggestion
        return {
            "error": error,
            "suggestion": "Check syntax near the error location",
            "confidence": 0.3
        }

    def _generate_python_fix(self, code: str, error: Dict) -> Dict:
        """Generate Python-specific fixes"""

        error_type = error.get("type", "")
        line_num = error.get("line", 0)

        if "IndentationError" in error_type:
            return {
                "error": error,
                "suggestion": "Fix indentation to match surrounding code",
                "fix_type": "indentation",
                "confidence": 0.9
            }
        elif "missing ':'" in error.get("message", ""):
            return {
                "error": error,
                "suggestion": "Add ':' at the end of the line",
                "fix_type": "missing_colon",
                "confidence": 0.95
            }
        elif "invalid syntax" in error.get("message", ""):
            # Analyze context for common patterns
            lines = code.split('\n')
            if line_num > 0 and line_num <= len(lines):
                line = lines[line_num - 1]

                # Check for common issues
                if "=" in line and "==" not in line:
                    return {
                        "error": error,
                        "suggestion": "Check if you meant '==' instead of '='",
                        "fix_type": "comparison",
                        "confidence": 0.7
                    }

        return None

    async def _initialize_language_servers(self, languages: List[str]):
        """Initialize language servers for given languages"""

        for lang in languages:
            if lang in self.lsp_configs:
                await self._start_language_server(lang)

    async def _start_language_server(self, language: str):
        """Start a language server process"""

        # Simplified - in real implementation would start actual LSP
        logger.info("starting_language_server", language=language)
        self.language_servers[language] = {
            "status": "running",
            "capabilities": ["validation", "completion", "hover"]
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "active_servers": list(self.language_servers.keys()),
            "cache_size": len(self.syntax_cache),
            "supported_languages": list(self.validators.keys())
        }


class PythonCodeAnalyzer(ast.NodeVisitor):
    """AST analyzer for Python code quality checks"""

    def __init__(self):
        self.issues = {"warnings": [], "info": []}
        self.complexity = 0

    def analyze(self, tree: ast.AST) -> Dict:
        """Analyze AST and return issues"""
        self.visit(tree)
        return self.issues

    def visit_FunctionDef(self, node):
        """Check function definitions"""

        # Check function length
        if len(node.body) > 50:
            self.issues["warnings"].append({
                "line": node.lineno,
                "message": f"Function '{node.name}' is too long ({len(node.body)} lines)",
                "type": "complexity"
            })

        # Check parameter count
        if len(node.args.args) > 5:
            self.issues["warnings"].append({
                "line": node.lineno,
                "message": f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                "type": "complexity"
            })

        self.generic_visit(node)