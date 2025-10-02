from typing import Dict, Any, List, Optional
import numpy as np
from collections import defaultdict
from core.interfaces import MemorySystem
from core.event_bus import EventBus
from brain_regions.gemini.gemini_service import GeminiService
import structlog

logger = structlog.get_logger()


class SemanticCortex(MemorySystem):
    """Long-term semantic memory storage and retrieval"""

    def __init__(self, event_bus: EventBus, gemini: GeminiService):
        self.event_bus = event_bus
        self.gemini = gemini

        # Semantic network
        self.concepts = {}  # concept_id -> concept_data
        self.relationships = defaultdict(list)  # concept_id -> [(relation, target_id)]
        self.concept_embeddings = {}  # concept_id -> embedding

        # Category hierarchies
        self.categories = defaultdict(set)  # category -> set of concept_ids
        self.category_hierarchy = {}  # child -> parent category

        self.state = {
            "concept_count": 0,
            "relationship_count": 0,
            "category_count": 0
        }

    async def initialize(self):
        """Initialize semantic cortex with base knowledge"""
        logger.info("initializing_semantic_cortex")

        # Load base semantic knowledge
        await self._load_base_knowledge()

        # Subscribe to events
        self.event_bus.subscribe("store_semantic_memory", self._on_store_request)
        self.event_bus.subscribe("query_semantic_memory", self._on_query_request)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process semantic memory operations"""

        operation = input_data.get("operation", "query")

        if operation == "store":
            return await self._store_concept(input_data)
        elif operation == "query":
            return await self._query_concepts(input_data)
        elif operation == "relate":
            return await self._add_relationship(input_data)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def store(self, data: Any, metadata: Optional[Dict] = None):
        """Store semantic knowledge"""

        # Extract concept information
        concept_info = await self._extract_concept_info(data)

        if concept_info:
            concept_id = concept_info["id"]

            # Store concept
            self.concepts[concept_id] = {
                "name": concept_info["name"],
                "definition": concept_info["definition"],
                "properties": concept_info["properties"],
                "metadata": metadata or {}
            }

            # Store embedding
            if "embedding" in concept_info:
                self.concept_embeddings[concept_id] = concept_info["embedding"]

            # Update categories
            for category in concept_info.get("categories", []):
                self.categories[category].add(concept_id)

            # Update relationships
            for relation, target in concept_info.get("relationships", []):
                self.relationships[concept_id].append((relation, target))

            self.state["concept_count"] = len(self.concepts)
            self.state["relationship_count"] = sum(
                len(rels) for rels in self.relationships.values()
            )

            logger.info("semantic_concept_stored",
                        concept_id=concept_id,
                        categories=concept_info.get("categories", []))

    async def retrieve(self, query: Any, k: int = 5) -> List[Dict]:
        """Retrieve relevant semantic knowledge"""

        # Parse query
        query_type = "text" if isinstance(query, str) else "structured"

        if query_type == "text":
            # Natural language query
            results = await self._retrieve_by_text(query, k)
        else:
            # Structured query
            results = await self._retrieve_by_structure(query, k)

        return results

    async def consolidate(self):
        """Consolidate and organize semantic knowledge"""

        # Find and merge similar concepts
        await self._merge_similar_concepts()

        # Infer new relationships
        await self._infer_relationships()

        # Update category hierarchy
        await self._update_hierarchy()

    async def _load_base_knowledge(self):
        """Load foundational semantic knowledge"""

        base_concepts = [
            {
                "id": "water",
                "name": "water",
                "definition": "A transparent, tasteless, odorless liquid (H2O)",
                "properties": ["liquid", "essential for life", "density: 1g/cm³"],
                "categories": ["substance", "liquid"]
            },
            {
                "id": "density",
                "name": "density",
                "definition": "Mass per unit volume",
                "properties": ["physical property", "formula: ρ = m/V"],
                "categories": ["physics", "property"]
            },
            {
                "id": "marine_life",
                "name": "marine life",
                "definition": "Organisms living in ocean or sea water",
                "properties": ["aquatic", "adapted to salt water"],
                "categories": ["biology", "ecosystem"]
            }
        ]

        for concept in base_concepts:
            await self.store(concept)

    async def _extract_concept_info(self, data: Any) -> Optional[Dict]:
        """Extract concept information from data"""

        if isinstance(data, dict) and "id" in data:
            # Already structured
            return data

        # Use Gemini to extract structure
        prompt = f"""Extract semantic concept information from:

{str(data)[:500]}

Provide:
1. id: unique identifier (lowercase, underscore-separated)
2. name: human-readable name
3. definition: clear, concise definition
4. properties: list of key properties
5. categories: list of categories it belongs to
6. relationships: list of (relation_type, target_concept) pairs

Output as JSON."""

        response = await self.gemini.generate_structured(
            prompt,
            schema={
                "id": "string",
                "name": "string",
                "definition": "string",
                "properties": ["string"],
                "categories": ["string"],
                "relationships": [["string", "string"]]
            }
        )

        if response["success"] and response["parsed"]:
            concept_info = response["parsed"]

            # Generate embedding
            embedding_text = f"{concept_info['name']}: {concept_info['definition']}"
            embedding_response = await self.event_bus.emit("decode_semantics", {
                "text": embedding_text,
                "mode": "encode"
            })

            # Add embedding if available
            if embedding_response and "vector" in embedding_response:
                concept_info["embedding"] = embedding_response["vector"]

            return concept_info

        return None

    async def _retrieve_by_text(self, query: str, k: int) -> List[Dict]:
        """Retrieve concepts by text query"""

        # Get query embedding
        query_embedding = await self._get_text_embedding(query)

        if not query_embedding:
            # Fallback to keyword search
            return self._keyword_search(query, k)

        # Find similar concepts by embedding
        similarities = []

        for concept_id, embedding in self.concept_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((concept_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for concept_id, similarity in similarities[:k]:
            concept = self.concepts.get(concept_id, {})
            results.append({
                "concept": concept,
                "id": concept_id,
                "similarity": similarity,
                "relationships": self.relationships.get(concept_id, [])
            })

        return results

    async def _retrieve_by_structure(self, query: Dict, k: int) -> List[Dict]:
        """Retrieve concepts by structured query"""

        category = query.get("category")
        properties = query.get("properties", [])
        relationships = query.get("relationships", [])

        matching_concepts = set(self.concepts.keys())

        # Filter by category
        if category:
            if category in self.categories:
                matching_concepts &= self.categories[category]
            else:
                matching_concepts = set()

        # Filter by properties
        for prop in properties:
            matching_concepts = {
                cid for cid in matching_concepts
                if prop in self.concepts[cid].get("properties", [])
            }

        # Filter by relationships
        for rel_type, target in relationships:
            matching_concepts = {
                cid for cid in matching_concepts
                if (rel_type, target) in self.relationships.get(cid, [])
            }

        # Build results
        results = []
        for concept_id in list(matching_concepts)[:k]:
            concept = self.concepts[concept_id]
            results.append({
                "concept": concept,
                "id": concept_id,
                "relationships": self.relationships.get(concept_id, [])
            })

        return results

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """Fallback keyword search"""

        query_lower = query.lower()
        scores = {}

        for concept_id, concept in self.concepts.items():
            score = 0

            # Check name
            if query_lower in concept["name"].lower():
                score += 3

            # Check definition
            if query_lower in concept["definition"].lower():
                score += 2

            # Check properties
            for prop in concept.get("properties", []):
                if query_lower in prop.lower():
                    score += 1

            if score > 0:
                scores[concept_id] = score

        # Sort by score
        sorted_concepts = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for concept_id, score in sorted_concepts[:k]:
            concept = self.concepts[concept_id]
            results.append({
                "concept": concept,
                "id": concept_id,
                "score": score,
                "relationships": self.relationships.get(concept_id, [])
            })

        return results

    async def _add_relationship(self, data: Dict) -> Dict:
        """Add relationship between concepts"""

        source_id = data.get("source")
        relation = data.get("relation")
        target_id = data.get("target")

        if source_id in self.concepts and target_id in self.concepts:
            self.relationships[source_id].append((relation, target_id))

            # Add inverse relationship if applicable
            inverse_relations = {
                "is_a": "has_instance",
                "part_of": "has_part",
                "causes": "caused_by",
                "requires": "required_by"
            }

            if relation in inverse_relations:
                inverse = inverse_relations[relation]
                self.relationships[target_id].append((inverse, source_id))

            self.state["relationship_count"] = sum(
                len(rels) for rels in self.relationships.values()
            )

            return {"success": True, "relationship_added": True}

        return {"success": False, "error": "Concepts not found"}

    async def _merge_similar_concepts(self):
        """Find and merge similar concepts during consolidation"""

        # Find highly similar concept pairs
        merge_candidates = []

        concept_ids = list(self.concept_embeddings.keys())
        for i in range(len(concept_ids)):
            for j in range(i + 1, len(concept_ids)):
                id1, id2 = concept_ids[i], concept_ids[j]

                similarity = self._cosine_similarity(
                    self.concept_embeddings[id1],
                    self.concept_embeddings[id2]
                )

                if similarity > 0.95:  # Very similar
                    merge_candidates.append((id1, id2, similarity))

        # Merge concepts
        for id1, id2, sim in merge_candidates:
            logger.info("merging_similar_concepts",
                        concept1=id1, concept2=id2, similarity=sim)

            # Merge properties
            props1 = set(self.concepts[id1].get("properties", []))
            props2 = set(self.concepts[id2].get("properties", []))
            self.concepts[id1]["properties"] = list(props1 | props2)

            # Merge relationships
            rels2 = self.relationships.get(id2, [])
            for rel in rels2:
                if rel not in self.relationships[id1]:
                    self.relationships[id1].append(rel)

            # Remove duplicate
            del self.concepts[id2]
            del self.relationships[id2]
            if id2 in self.concept_embeddings:
                del self.concept_embeddings[id2]

    async def _infer_relationships(self):
        """Infer new relationships from existing ones"""

        # Transitive relationships
        transitive_relations = ["is_a", "part_of", "causes"]

        new_relationships = []

        for concept_id, relations in self.relationships.items():
            for rel_type, target_id in relations:
                if rel_type in transitive_relations:
                    # Check target's relationships
                    target_rels = self.relationships.get(target_id, [])
                    for target_rel_type, final_id in target_rels:
                        if target_rel_type == rel_type:
                            # Transitive inference
                            new_rel = (rel_type, final_id)
                            if new_rel not in relations:
                                new_relationships.append((concept_id, new_rel))

        # Add inferred relationships
        for concept_id, (rel_type, target_id) in new_relationships:
            self.relationships[concept_id].append((rel_type, target_id))
            logger.info("inferred_relationship",
                        source=concept_id, relation=rel_type, target=target_id)

    async def _update_hierarchy(self):
        """Update category hierarchy"""

        # Infer hierarchy from is_a relationships
        for concept_id, relations in self.relationships.items():
            for rel_type, target_id in relations:
                if rel_type == "is_a" and target_id in self.categories:
                    # concept_id is a subcategory of target_id's categories
                    for parent_cat in self.categories.get(target_id, []):
                        self.category_hierarchy[concept_id] = parent_cat

    async def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""

        # Request embedding from semantic decoder
        response = await self.event_bus.emit("decode_semantics", {
            "text": text,
            "mode": "encode"
        })

        if response and "vector" in response:
            return np.array(response["vector"])

        return None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""

        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))

    async def _on_store_request(self, data: Dict):
        """Handle semantic storage requests"""
        await self.store(data.get("content"), data.get("metadata"))

    async def _on_query_request(self, data: Dict):
        """Handle semantic query requests"""
        results = await self.retrieve(data.get("query"), data.get("k", 5))

        await self.event_bus.emit("semantic_query_complete", {
            "results": results,
            "query": data.get("query")
        })

    def get_state(self) -> Dict[str, Any]:
        return self.state