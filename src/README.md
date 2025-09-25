# NeuroStructural Decoding - Core Algorithm

This directory contains the core implementation of the NeuroStructural Decoding algorithm as described in the ACL 2023 paper.

## Core Files

- **`enhanced_neurostructural_decoding.py`** - Main NeuroStructural Decoding algorithm with NeuroLogic integration
- **`parsers.py`** - Parser adapter for your trained parser (DirectParserAdapter) 
- **`commongen_constraint_extractor.py`** - CommonGen-specific constraint extraction using trained parser
- **`__init__.py`** - Package initialization and exports

## Quick Start

```python

from enhanced_neurostructural_decoding import EnhancedNeuroStructuralDecoder
from parsers import DirectParserAdapter  # For trained parser
from commongen_constraint_extractor import CommonGenConstraintExtractor

# Initialize decoder
decoder = EnhancedNeuroStructuralDecoder(
    model_name="google/flan-t5-small",
    parser_adapter=DirectParserAdapter("path/to/best_parser.th")
)

# Generate with constraints
constraints = [UnaryConstraint("dog", "NOUN"), BinaryConstraint("run", "park", "nmod")]
results = decoder.generate_with_constraints("Generate:", concepts=["dog", "run", "park"], constraints=constraints)
```

## Algorithm Components

### Constraint Types
- **Unary Constraints**: Part-of-speech requirements for individual tokens
- **Binary Constraints**: Dependency relations between token pairs  
- **Triplet Constraints**: Complex structural patterns involving three tokens

### Parser Integration
- **DirectParserAdapter**: Uses your trained parser (best_parser.th)
- You can download the parser from <a href='https://drive.google.com/file/d/1crjCoUs9KK3S83QT0edpycv-g8V6NK4H'> here</a>

### Generation Process
1. Structural constraint extraction from concept sets
2. Enhanced beam search with constraint satisfaction
3. NeuroLogic-style lookahead integration
4. Constraint violation penalty system
