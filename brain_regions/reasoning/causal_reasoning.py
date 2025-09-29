from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
from core.interfaces import ReasoningModule
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class CausalReasoning(ReasoningModule):
    """Causal reasoning and cause-effect analysis"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini
        self.confidence = 0.0
        self.causal_graphs = {}  # Store causal models

    async def initialize(self):
        """Initialize causal reasoning"""
        logger.info("initializing_causal_reasoning")

        # Subscribe to reasoning requests
        self.event_bus.subscribe("causal_reasoning_request", self._on_reasoning_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process causal reasoning request"""

        scenario = input_data.get("scenario", "")
        query = input_data.get("query", "")
        context = input_data.get("context", {})

        result = await self.reason(f"{scenario}. {query}", context)

        # Emit completion
        await self.event_bus.emit("causal_reasoning_complete", result)

        return result

    async def reason(self, problem: str, context: Dict) -> Dict:
        """Perform causal reasoning"""

        # Extract causal scenario
        scenario = await self._extract_scenario(problem, context)

        if not scenario["success"]:
            return {
                "success": False,
                "error": "Could not extract causal scenario"
            }

        # Build causal graph
        causal_graph = await self._build_causal_graph(scenario)

        # Trace causal chains
        causal_chains = self._trace_causal_chains(
            causal_graph,
            scenario.get("initial_cause"),
            scenario.get("query_effect")
        )

        # Analyze interventions
        interventions = await self._analyze_interventions(causal_graph, scenario)

        # Generate predictions
        predictions = await self._generate_predictions(causal_chains, context)

        self.confidence = self._evaluate_causal_model(causal_graph, causal_chains)

        return {
            "success": True,
            "scenario": scenario,
            "causal_graph": self._graph_to_dict(causal_graph),
            "causal_chains": causal_chains,
            "interventions": interventions,
            "predictions": predictions,
            "confidence": self.confidence
        }

    async def _extract_scenario(self, problem: str, context: Dict) -> Dict:
        """Extract causal scenario from problem"""

        prompt = f"""Extract the causal scenario from this problem:

Problem: {problem}
Context: {context}

Identify:
1. initial_cause: The root cause or starting event
2. effects: List of effects or consequences
3. mediating_factors: Intermediate causes/effects
4. query_effect: What effect we're analyzing (if specified)
5. time_scale: immediate/short-term/long-term

Output as JSON."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "initial_cause": "string",
                "effects": ["string"],
                "mediating_factors": ["string"],
                "query_effect": "string",
                "time_scale": "string"
            }
        )

        if response["success"] and response["parsed"]:
            return {
                "success": True,
                **response["parsed"]
            }

        return {"success": False}

    async def _build_causal_graph(self, scenario: Dict) -> nx.DiGraph:
        """Build causal graph from scenario"""

        G = nx.DiGraph()

        # Add initial cause
        initial = scenario.get("initial_cause", "cause")
        G.add_node(initial, node_type="cause", level=0)

        # Add effects
        for effect in scenario.get("effects", []):
            G.add_node(effect, node_type="effect", level=2)
            G.add_edge(initial, effect, weight=0.8)

        # Add mediating factors
        for factor in scenario.get("mediating_factors", []):
            G.add_node(factor, node_type="mediator", level=1)

            # Connect to initial cause
            G.add_edge(initial, factor, weight=0.7)

            # Connect to effects (use Gemini to determine connections)
            connections = await self._determine_connections(factor, scenario.get("effects", []))

            for effect, strength in connections:
                G.add_edge(factor, effect, weight=strength)

        # Add feedback loops if detected
        feedback_loops = await self._detect_feedback_loops(scenario)
        for loop in feedback_loops:
            if len(loop) >= 2:
                G.add_edge(loop[-1], loop[0], weight=0.5, feedback=True)

        return G

    async def _determine_connections(self, factor: str, effects: List[str]) -> List[Tuple[str, float]]:
        """Determine causal connections between factor and effects"""

        if not effects:
            return []

        prompt = f"""Determine causal connections between this factor and effects:

Factor: {factor}
Possible Effects: {', '.join(effects)}

For each effect that the factor influences:
1. State the effect
2. Rate connection strength (0-1)
3. Explain the causal mechanism briefly

Only include actual causal relationships."""

        response = await self.gemini.generate(prompt, config_name="fast")

        connections = []

        if response["success"]:
            # Parse connections from response
            lines = response["text"].split('\n')

            for line in lines:
                for effect in effects:
                    if effect.lower() in line.lower():
                        # Extract strength
                        strength = 0.7  # default

                        if "strong" in line.lower():
                            strength = 0.9
                        elif "weak" in line.lower():
                            strength = 0.4
                        elif "moderate" in line.lower():
                            strength = 0.6

                        connections.append((effect, strength))
                        break

        return connections

    async def _detect_feedback_loops(self, scenario: Dict) -> List[List[str]]:
        """Detect potential feedback loops in causal scenario"""

        elements = [scenario.get("initial_cause")] + \
                   scenario.get("mediating_factors", []) + \
                   scenario.get("effects", [])

        prompt = f"""Identify feedback loops in this causal scenario:

Elements: {', '.join(elements)}

A feedback loop occurs when an effect influences its own cause.
List any feedback loops as sequences of elements.
Only include actual feedback relationships."""

        response = await self.gemini.generate(prompt, config_name="fast")

        loops = []

        if response["success"]:
            # Parse loops from response
            # Simple parsing - could be enhanced
            lines = response["text"].split('\n')

            for line in lines:
                if '→' in line or '->' in line:
                    # Extract sequence
                    parts = line.replace('→', '->').split('->')
                    loop = [p.strip() for p in parts if p.strip()]

                    if len(loop) >= 2:
                        loops.append(loop)

        return loops

    def _trace_causal_chains(self,
                             graph: nx.DiGraph,
                             start: Optional[str],
                             end: Optional[str]) -> List[Dict]:
        """Trace causal chains through the graph"""

        chains = []

        if not start:
            # Find root causes
            starts = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        else:
            starts = [start] if start in graph else []

        if not end:
            # Find final effects
            ends = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        else:
            ends = [end] if end in graph else []

        # Find all paths
        for s in starts:
            for e in ends:
                try:
                    paths = list(nx.all_simple_paths(graph, s, e, cutoff=5))

                    for path in paths:
                        # Calculate chain strength
                        strength = 1.0

                        for i in range(len(path) - 1):
                            edge_data = graph.get_edge_data(path[i], path[i + 1])
                            strength *= edge_data.get("weight", 1.0)
                        chains.append({
                            "path": path,
                            "strength": strength,
                            "length": len(path),
                            "type": "direct" if len(path) == 2 else "indirect"
                        })

                except nx.NetworkXNoPath:
                    continue

        # Sort by strength
        chains.sort(key=lambda x: x["strength"], reverse=True)

        return chains

    async def _analyze_interventions(self, graph: nx.DiGraph, scenario: Dict) -> List[Dict]:
        """Analyze potential interventions in causal chain"""

        nodes = list(graph.nodes())

        prompt = f"""Analyze potential interventions in this causal scenario:

Initial Cause: {scenario.get('initial_cause')}
Effects: {', '.join(scenario.get('effects', []))}
Mediating Factors: {', '.join(scenario.get('mediating_factors', []))}

For each possible intervention point:
1. Where to intervene
2. Type of intervention (block/enhance/redirect)
3. Expected outcome
4. Confidence level

Focus on practical, actionable interventions."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        interventions = []

        if response["success"]:
            # Parse interventions
            parsed = self._parse_interventions(response["text"], nodes)

            # Analyze each intervention's impact
            for intervention in parsed:
                impact = self._analyze_intervention_impact(graph, intervention)
                intervention["impact"] = impact
                interventions.append(intervention)

        return interventions

    async def _generate_predictions(self, causal_chains: List[Dict], context: Dict) -> List[Dict]:
        """Generate predictions based on causal analysis"""

        if not causal_chains:
            return []

        # Format top chains for analysis
        chain_text = "\n".join([
            f"Chain {i + 1} (strength: {c['strength']:.2f}): {' → '.join(c['path'])}"
            for i, c in enumerate(causal_chains[:5])
        ])

        prompt = f"""Based on these causal chains:

        {chain_text}
        
        Context: {context}
        
        Generate predictions:
        1. Most likely outcomes
        2. Time frame for each outcome
        3. Probability estimate
        4. Potential surprises or unintended consequences
        
        Be specific and consider chain strengths."""

        response = await self.gemini.generate(prompt, config_name="balanced")

        predictions = []

        if response["success"]:
            # Parse predictions
            parsed = self._parse_predictions(response["text"])

            for pred in parsed:
                # Add causal support
                supporting_chains = [
                    c for c in causal_chains
                    if any(outcome in ' '.join(c['path']) for outcome in [pred.get('outcome', '')])
                ]

                pred["causal_support"] = len(supporting_chains)
                pred["max_chain_strength"] = max([c['strength'] for c in supporting_chains], default=0)

                predictions.append(pred)

        return predictions

    def _parse_interventions(self, text: str, valid_nodes: List[str]) -> List[Dict]:
        """Parse interventions from text"""

        interventions = []
        lines = text.split('\n')

        current_intervention = {}

        for line in lines:
            line = line.strip()

            # Look for intervention points
            for node in valid_nodes:
                if node.lower() in line.lower():
                    if current_intervention:
                        interventions.append(current_intervention)

                    current_intervention = {
                        "target": node,
                        "type": "block",  # default
                        "description": line,
                        "confidence": 0.7
                    }

                    # Determine type
                    if "block" in line.lower() or "prevent" in line.lower():
                        current_intervention["type"] = "block"
                    elif "enhance" in line.lower() or "increase" in line.lower():
                        current_intervention["type"] = "enhance"
                    elif "redirect" in line.lower() or "change" in line.lower():
                        current_intervention["type"] = "redirect"

                    break

            # Look for confidence indicators
            if current_intervention:
                if "high confidence" in line.lower():
                    current_intervention["confidence"] = 0.9
                elif "low confidence" in line.lower():
                    current_intervention["confidence"] = 0.4

        # Add last intervention
        if current_intervention:
            interventions.append(current_intervention)

        return interventions

    def _analyze_intervention_impact(self, graph: nx.DiGraph, intervention: Dict) -> Dict:
        """Analyze the impact of an intervention"""

        target = intervention.get("target")
        itype = intervention.get("type")

        if target not in graph:
            return {"affected_nodes": 0, "blocked_paths": 0}

        impact = {
            "affected_nodes": 0,
            "blocked_paths": 0,
            "enhanced_paths": 0
        }

        if itype == "block":
            # Count downstream nodes that would be affected
            descendants = nx.descendants(graph, target)
            impact["affected_nodes"] = len(descendants)

            # Count paths that would be blocked
            for node in descendants:
                if graph.out_degree(node) == 0:  # End node
                    impact["blocked_paths"] += 1

        elif itype == "enhance":
            # Count paths that would be strengthened
            descendants = nx.descendants(graph, target)
            impact["affected_nodes"] = len(descendants)
            impact["enhanced_paths"] = len([n for n in descendants if graph.out_degree(n) == 0])

        return impact

    def _parse_predictions(self, text: str) -> List[Dict]:
        """Parse predictions from text"""

        predictions = []
        lines = text.split('\n')

        current_pred = {}

        for line in lines:
            line = line.strip()

            if not line:
                if current_pred:
                    predictions.append(current_pred)
                    current_pred = {}
                continue

            # Look for outcome descriptions
            if any(keyword in line.lower() for keyword in ["outcome", "result", "consequence"]):
                if current_pred:
                    predictions.append(current_pred)

                current_pred = {
                    "outcome": line,
                    "timeframe": "medium-term",
                    "probability": 0.7
                }

            # Look for timeframes
            if current_pred:
                if "immediate" in line.lower():
                    current_pred["timeframe"] = "immediate"
                elif "short" in line.lower():
                    current_pred["timeframe"] = "short-term"
                elif "long" in line.lower():
                    current_pred["timeframe"] = "long-term"

                # Look for probability
                if "likely" in line.lower():
                    current_pred["probability"] = 0.8
                elif "unlikely" in line.lower():
                    current_pred["probability"] = 0.3
                elif "certain" in line.lower():
                    current_pred["probability"] = 0.95

        # Add last prediction
        if current_pred:
            predictions.append(current_pred)

        return predictions

    def _evaluate_causal_model(self, graph: nx.DiGraph, chains: List[Dict]) -> float:
        """Evaluate confidence in causal model"""

        # Factors affecting confidence
        confidence = 0.5

        # More nodes and edges = more complete model
        if graph.number_of_nodes() >= 5:
            confidence += 0.1

        if graph.number_of_edges() >= 7:
            confidence += 0.1

        # Strong causal chains increase confidence
        if chains:
            max_strength = max(c["strength"] for c in chains)
            if max_strength > 0.7:
                confidence += 0.2

        # Multiple paths increase confidence
        if len(chains) >= 3:
            confidence += 0.1

        # Feedback loops add complexity but reduce certainty
        feedback_edges = [e for e in graph.edges(data=True) if e[2].get("feedback", False)]
        if feedback_edges:
            confidence -= 0.1

        return min(max(confidence, 0.0), 1.0)

    def _graph_to_dict(self, graph: nx.DiGraph) -> Dict:
        """Convert networkx graph to dictionary"""

        return {
            "nodes": [
                {
                    "id": node,
                    "type": data.get("node_type", "unknown"),
                    "level": data.get("level", 0)
                }
                for node, data in graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1.0),
                    "feedback": data.get("feedback", False)
                }
                for u, v, data in graph.edges(data=True)
            ]
        }

    async def _on_reasoning_request(self, data: Dict):
        """Handle causal reasoning requests"""
        result = await self.process(data)

    def get_confidence(self) -> float:
        return self.confidence

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_confidence": self.confidence,
            "cached_graphs": len(self.causal_graphs)
        }