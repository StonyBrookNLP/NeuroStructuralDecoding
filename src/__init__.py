"""
NeuroStructural Decoding Implementation

A complete implementation of NeuroStructural Decoding for constrained neural text generation,
extending NeuroLogic Decoding with syntactic structural constraints derived from dependency parsing.

This implementation supports:
- Unary constraints (word-POS tag pairs)
- Binary constraints (head-dependent relationships) 
- Triplet constraints (head-dependent-label relationships)
- MST-based dependency parsing with joint head-label optimization
- Integration with trained BiLSTM dependency parsers
- CommonGen dataset constraint extraction

Core Modules:
- enhanced_neurostructural_decoding: Main NeuroStructural decoding algorithm
- parsers: Dependency parser adapters (DirectParserAdapter with MST decoding)
- commongen_constraint_extractor: Constraint extraction from concept sets

References:
Based on NeuroLogic Decoding (Lu et al., 2022) with structural constraints
as described in ACL 2023 NeuroStructural Decoding paper.
"""

# Core NeuroStructural Decoding components
from .enhanced_neurostructural_decoding import (
    EnhancedNeuroStructuralDecoder,
    EnhancedConstrainedHypothesis,
    UnaryConstraint,
    BinaryConstraint,
    TripletConstraint
)

# Parser adapters with graceful fallback
try:
    from .parsers import (
        DirectParserAdapter,
        AllenNLPParserAdapter, 
        ParserAdapter
    )
except ImportError as e:
    # Parser dependencies not available - set to None for conditional usage
    import warnings
    warnings.warn(f"Parser modules not available: {e}", ImportWarning)
    DirectParserAdapter = None
    AllenNLPParserAdapter = None
    ParserAdapter = None

# Constraint extraction utilities
from .commongen_constraint_extractor import (
    CommonGenConstraintExtractor
)

# Package metadata
__version__ = "1.0.0"
__author__ = "NeuroStructural Decoding Team"
__license__ = "MIT"
__description__ = "NeuroStructural Decoding for Constrained Neural Text Generation"

# Public API exports
__all__ = [
    # Core decoding algorithm
    "EnhancedNeuroStructuralDecoder",
    "EnhancedConstrainedHypothesis",
    
    # Structural constraints
    "UnaryConstraint", 
    "BinaryConstraint", 
    "TripletConstraint",
    
    # Parser adapters
    "DirectParserAdapter",
    "AllenNLPParserAdapter", 
    "ParserAdapter",
    
    # Constraint extraction
    "CommonGenConstraintExtractor",
    
    # Convenience functions
    "create_decoder",
    "setup_parser", 
    "extract_commongen_constraints",
]

# Convenience functions for common usage patterns
def create_decoder(model, tokenizer, **kwargs):
    """
    Create a NeuroStructural decoder with default configuration.
    
    Args:
        model: Hugging Face model (AutoModelForSeq2SeqLM)
        tokenizer: Hugging Face tokenizer (AutoTokenizer)
        **kwargs: Additional arguments for EnhancedNeuroStructuralDecoder
        
    Returns:
        EnhancedNeuroStructuralDecoder instance
        
    Note:
        The global parser is automatically initialized from best_parser.th
        when the decoder is created.
    """
    from .enhanced_neurostructural_decoding import EnhancedNeuroStructuralDecoder
    
    # Create decoder - parser will be automatically initialized
    return EnhancedNeuroStructuralDecoder(model=model, tokenizer=tokenizer, **kwargs)

def setup_parser(parser_path="best_parser.th"):
    """
    Setup dependency parser for NeuroStructural decoding.
    
    Args:
        parser_path: Path to trained dependency parser model
        
    Returns:
        DirectParserAdapter instance
        
    Note:
        The global parser is automatically initialized when needed.
        This function creates a standalone parser instance.
    """
    from .parsers import DirectParserAdapter
    
    parser = DirectParserAdapter(parser_path)
    return parser

def extract_commongen_constraints(concept_set, parser_path="best_parser.th"):
    """
    Extract constraints from CommonGen concept set.
    
    Args:
        concept_set: List of concept words
        parser_path: Path to trained dependency parser model
        
    Returns:
        Dictionary with unary, binary, and triplet constraints
    """
    from .commongen_constraint_extractor import CommonGenConstraintExtractor
    
    extractor = CommonGenConstraintExtractor(parser_path)
    sentence = " ".join(concept_set)  # Simple concatenation
    return extractor.extract_constraints_from_sentence(sentence, concept_set)