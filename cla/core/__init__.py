"""
CLA Core - główne komponenty Cognitive Layer Architecture.
"""

from .concept import Concept, DualityPair, create_concept_from_dict
from .concept_graph import ConceptGraph
from .dual_processing import DualProcessingEngine, CognitiveSynthesis
from .awareness import CognitiveAwareness, AwarenessState
from .meta_controller import MetaController, CognitiveSensitivity, AttentionAllocation
from .safety_gate import SafetyGate, SafetyViolation
from .cognitive_layer import CognitiveLayer

__all__ = [
    'Concept',
    'DualityPair',
    'create_concept_from_dict',
    'ConceptGraph',
    'DualProcessingEngine',
    'CognitiveSynthesis',
    'CognitiveAwareness',
    'AwarenessState',
    'MetaController',
    'CognitiveSensitivity',
    'AttentionAllocation',
    'SafetyGate',
    'SafetyViolation',
    'CognitiveLayer'
]

