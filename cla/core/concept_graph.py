"""
Graf konceptów - substrat relacyjny dla CLA.
Implementuje spreading activation i dynamiczną ewolucję.
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import networkx as nx

from .concept import Concept, DualityPair


class ConceptGraph:
    """
    Dynamiczny graf konceptów z spreading activation.
    
    To jest "pamięć konceptualna" systemu - nie przechowuje surowego tekstu,
    ale strukturę znaczeń i relacji między nimi.
    """
    
    def __init__(self, decay_rate: float = 0.1):
        self.concepts: Dict[str, Concept] = {}  # {concept_id: Concept}
        self.decay_rate = decay_rate
        
        # NetworkX graph dla zaawansowanych operacji
        self.graph = nx.DiGraph()  # Skierowany dla relacji hierarchicznych
        self.current_era = "genesis"
    
    def add_concept(self, concept: Concept) -> str:
        """Dodaj koncept do grafu."""
        self.concepts[concept.concept_id] = concept
        self.graph.add_node(concept.concept_id, concept=concept)
        return concept.concept_id

    def remove_concept(self, concept_id: str):
        """Usuń koncept z grafu."""
        if concept_id in self.concepts:
            del self.concepts[concept_id]
        if self.graph.has_node(concept_id):
            self.graph.remove_node(concept_id)
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Pobierz koncept po ID."""
        return self.concepts.get(concept_id)
    
    def find_concept_by_name(self, name: str) -> Optional[Concept]:
        """Znajdź koncept po nazwie."""
        for concept in self.concepts.values():
            if concept.name == name:
                return concept
        return None
    
    def find_similar_concepts(self, target_embedding: np.ndarray, threshold: float = 0.7, limit: int = 3) -> List[Tuple[Concept, float]]:
        """Znajdź koncepty semantycznie zbliżone (Cosine Similarity)."""
        matches = []
        if target_embedding is None:
            return []
            
        norm_target = np.linalg.norm(target_embedding)
        if norm_target == 0: return []
        
        for concept in self.concepts.values():
            if concept.embedding is None: continue
            
            # Oblicz podobieństwo (tylko jeśli wymiary się zgadzają)
            if concept.embedding.shape != target_embedding.shape:
                continue
                
            norm_c = np.linalg.norm(concept.embedding)
            if norm_c == 0: continue
            
            sim = np.dot(target_embedding, concept.embedding) / (norm_target * norm_c)
            
            if sim >= threshold:
                matches.append((concept, sim))
        
        # Sortuj malejąco
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]
    
    def link_concepts(self, concept_a_id: str, concept_b_id: str, strength: float, rel_type: str = 'generic'):
        """Stwórz lub wzmocnij link między konceptami."""
        if concept_a_id in self.concepts and concept_b_id in self.concepts:
            self.concepts[concept_a_id].link_to(concept_b_id, strength, rel_type)
            
            # Aktualizuj NetworkX graph
            self.graph.add_edge(concept_a_id, concept_b_id, weight=strength, type=rel_type)
    
    def spreading_activation(
        self, 
        source_concept_ids: List[str], 
        initial_activation: float = 1.0,
        decay_factor: float = 0.7,
        max_hops: int = 3
    ) -> Dict[str, float]:
        """
        Spreading activation - energia rozprzestrzenia się po grafie.
        
        Args:
            source_concept_ids: Koncepty źródłowe (początkowa aktywacja)
            initial_activation: Początkowa siła aktywacji
            decay_factor: Jak bardzo aktywacja słabnie z odległością
            max_hops: Maksymalna odległość propagacji
        
        Returns:
            Dict {concept_id: activation_level}
        """
        activations = defaultdict(float)
        
        # Inicjalizacja źródeł
        for concept_id in source_concept_ids:
            activations[concept_id] = initial_activation
        
        # Propagacja przez max_hops kroków
        for hop in range(max_hops):
            new_activations = activations.copy()
            current_decay = decay_factor ** (hop + 1)
            
            for concept_id, activation in activations.items():
                if activation > 0.01:  # Threshold - ignoruj bardzo słabe
                    concept = self.concepts.get(concept_id)
                    if concept:
                        # Propaguj do sąsiadów
                        for neighbor_id, link_data in concept.links.items():
                            link_strength, rel_type = link_data
                            propagated = activation * link_strength * current_decay
                            new_activations[neighbor_id] = max(
                                new_activations[neighbor_id],
                                propagated
                            )
            
            activations = new_activations
        
        # Aktualizuj aktywacje w konceptach
        for concept_id, activation in activations.items():
            if concept_id in self.concepts:
                self.concepts[concept_id].activation = activation
        
        return dict(activations)
    
    def get_active_concepts(self, threshold: float = 0.3) -> List[Concept]:
        """Pobierz wszystkie aktywne koncepty powyżej threshold."""
        return [
            concept for concept in self.concepts.values()
            if concept.activation >= threshold
        ]
    
    def decay_all(self, global_decay_rate: float = 0.95) -> List[str]:
        """
        Zaktualizowany naturalny zanik (ADS v5.6: Tryb Ogrodnika).
        Zwraca listę ID konceptów do usunięcia.
        """
        to_remove = []
        for cid, concept in self.concepts.items():
            # 1. Chroń DNA, Wartości i Emocje
            ctype = (concept.properties or {}).get("type", "unknown")
            if ctype in ["dna", "core_value", "emotion"] or concept.weight > 0.8:
                continue

            # 2. Inspekt Inkubacji: Chroń nowe idee z potencjałem (2+ linki)
            if (len(concept.links) >= 2 or concept.is_incubating) and concept.weight < 0.3:
                concept.weight = min(0.3, concept.weight + 0.02) # "Podlewanie" rośliny
                concept.is_incubating = True
                continue
            
            # 3. Reguła Długiego Cienia: Chroń mosty do DNA/Ważnych pojęć
            is_bridge = False
            for target_id in concept.links.keys():
                target_c = self.get_concept(target_id)
                if target_c and target_c.weight > 0.8:
                    is_bridge = True
                    break
            if is_bridge and concept.weight > 0.15:
                continue

            # 4. Standardowy Decay (modyfikowany przez Depth)
            # depth 1.0 -> brak decay, depth 0.0 -> pełny decay
            effective_decay = global_decay_rate + (1.0 - global_decay_rate) * concept.depth
            
            concept.weight *= effective_decay
            concept.activation *= (effective_decay * 0.6)
            
            if concept.weight < 0.12: # Próg zapomnienia (ADS Gardener)
                to_remove.append(cid)
        
        # Wykonaj usuwanie
        for cid in to_remove:
            self.remove_concept(cid)
            
        return to_remove
    
    def find_dualities(
        self, 
        active_concepts: Optional[List[Concept]] = None,
        min_opposition: float = 0.6
    ) -> List[DualityPair]:
        """
        Znajdź pary konceptów w relacji dualności.
        
        Args:
            active_concepts: Lista aktywnych konceptów (jeśli None, użyj wszystkich)
            min_opposition: Minimalna przeciwstawność do uznania za dualność
        
        Returns:
            Lista par dualności
        """
        if active_concepts is None:
            active_concepts = self.get_active_concepts(threshold=0.3)
        
        dualities = []
        
        # Sprawdź wszystkie pary
        for i, concept_a in enumerate(active_concepts):
            for concept_b in active_concepts[i+1:]:
                # Czy są w tej samej erze (opcjonalnie) lub tej samej kategorii?
                if (concept_a.duality_category and 
                    concept_a.duality_category == concept_b.duality_category):
                    
                    pair = DualityPair(
                        pole_a=concept_a,
                        pole_b=concept_b,
                        category=concept_a.duality_category
                    )
                    
                    # Oblicz przeciwstawność
                    pair.calculate_opposition()
                    
                    if pair.opposition >= min_opposition:
                        # Oblicz tarcie
                        pair.calculate_friction()
                        dualities.append(pair)
        
        # Sortuj po sile tarcia (malejąco)
        dualities.sort(key=lambda p: p.friction, reverse=True)
        
        return dualities
    
    def get_subgraph(self, concept_ids: List[str], depth: int = 1) -> nx.Graph:
        """Pobierz podgraf wokół danych konceptów."""
        nodes = set(concept_ids)
        
        # Dodaj sąsiadów do głębokości depth
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:
                    new_nodes.update(self.graph.neighbors(node))
            nodes.update(new_nodes)
        
        return self.graph.subgraph(nodes)
    
    def evolve_era(self, new_era_name: str):
        """Przejdź do nowej ery poznawczej."""
        self.current_era = new_era_name
        # Przy przejściu ery można zrezygnować z bardzo słabych linków lub konceptów
        to_remove = []
        for cid, concept in self.concepts.items():
            if concept.activation < 0.05 and concept.weight < 0.2:
                to_remove.append(cid)
        
        for cid in to_remove:
            del self.concepts[cid]
            self.graph.remove_node(cid)

    def get_hierarchical_path(self, start_id: str) -> List[str]:
        """Pobierz ścieżkę hierarchiczną (is_a)."""
        path = [start_id]
        current = start_id
        while True:
            found = False
            concept = self.concepts.get(current)
            if concept:
                for target_id, (strength, rtype) in concept.links.items():
                    if rtype == 'is_a':
                        path.append(target_id)
                        current = target_id
                        found = True
                        break
            if not found:
                break
        return path

    def __len__(self):
        return len(self.concepts)
    
    def __repr__(self):
        active = len(self.get_active_concepts())
        return f"ConceptGraph({len(self)} concepts, {active} active)"

