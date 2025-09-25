#!/usr/bin/env python3
"""
Constraint extraction for CommonGen dataset using trained parser.

This implements the automatic constraint extraction process as described in the
ACL 2023 NeuroStructural Decoding paper. The implementation uses our trained
dependency parser (DirectParserAdapter) instead of external parsers.
"""

import json
from typing import List, Dict, Tuple, Set

try:
    from .parsers import DirectParserAdapter
except ImportError:
    # Fallback for direct execution
    from parsers import DirectParserAdapter
    
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommonGenConstraintExtractor:
    """
    Extract structural constraints from CommonGen gold outputs.
    
    Based on the original paper approach:
    - Parse gold outputs with trained dependency parser
    - Extract unary, binary, and triplet constraints
    - Focus on dependency relations from actual model predictions
    """
    
    def __init__(self, parser_model_path: str = "best_parser.th"):
        """Initialize the constraint extractor with your trained parser."""
        self.parser = DirectParserAdapter(parser_model_path)
        logger.info("CommonGen constraint extractor initialized")
    
    def extract_constraints_from_sentence(self, sentence: str, concepts: List[str]) -> Dict:
        """
        Extract structural constraints from a single sentence.
        
        Args:
            sentence: Target sentence
            concepts: List of concept words that should appear
            
        Returns:
            Dictionary with unary, binary, and triplet constraints
        """
        # Parse the sentence
        parse_result = self.parser.parse_sentence(sentence)
        
        # Convert concepts to lowercase for matching
        concepts_lower = [c.lower() for c in concepts]
        
        constraints = {
            'lexical': concepts,  # Original lexical constraints
            'unary': [],         # Word + role constraints
            'binary': [],        # Word-relation-word constraints  
            'triplet': []        # Complex connected constraints
        }
        
        # Build token index for head lookups
        tokens = parse_result['tokens']
        token_by_id = {token['id']: token for token in tokens}
        
        # Extract unary constraints (word appearing in specific roles)
        for token in tokens:
            word = token['token'].lower()
            head_idx = token['head']
            relation = token['relation']
            
            # Since we use raw label indices, we'll work with any relation
            # that connects concept words (our parser uses label_X format)
            if word in concepts_lower:
                constraints['unary'].append({
                    'word': word,
                    'relation': relation,
                    'type': 'token'
                })
        
        # Extract binary constraints (direct relations between concept words)
        for token in tokens:
            word = token['token'].lower()
            head_idx = token['head']
            relation = token['relation']
            
            # Get head word if it exists
            if head_idx != 0 and head_idx in token_by_id:
                head_token = token_by_id[head_idx]
                head_word = head_token['token'].lower()
                
                # If both words are concept words
                if word in concepts_lower and head_word in concepts_lower:
                    constraints['binary'].append({
                        'word1': word,
                        'relation': relation,
                        'word2': head_word,
                        'direction': 'word1_to_word2'
                    })
        
        # Extract triplet constraints (complex patterns)
        # Following the paper's approach for connected relations
        triplets = self._extract_triplet_constraints(parse_result, concepts_lower)
        constraints['triplet'] = triplets
        
        return constraints
    
    def _extract_triplet_constraints(self, parse_result: Dict, concepts: List[str]) -> List[Dict]:
        """Extract triplet constraints (subject-verb-object patterns)."""
        triplets = []
        tokens = parse_result['tokens']
        token_by_id = {token['id']: token for token in tokens}
        
        # Find head tokens that connect to multiple concept words
        connections_by_head = {}
        
        for token in tokens:
            word = token['token'].lower()
            head_idx = token['head']
            relation = token['relation']
            
            # If this token is a concept word and has a head
            if word in concepts and head_idx != 0 and head_idx in token_by_id:
                head_token = token_by_id[head_idx]
                head_word = head_token['token'].lower()
                
                if head_word not in connections_by_head:
                    connections_by_head[head_word] = []
                connections_by_head[head_word].append({
                    'word': word,
                    'relation': relation
                })
        
        # Create triplet constraints for heads with multiple concept connections
        for head_word, connections in connections_by_head.items():
            if len(connections) >= 2:
                # Create triplets from all combinations of connected concepts
                for i, conn1 in enumerate(connections):
                    for conn2 in connections[i+1:]:
                        if conn1['word'] != conn2['word']:
                            triplets.append({
                                'word1': conn1['word'],
                                'center': head_word,
                                'word2': conn2['word'],
                                'relation1': conn1['relation'],
                                'relation2': conn2['relation']
                            })
        
        return triplets
    
    def process_commongen_dataset(self, data_file: str, output_file: str, max_examples: int = None):
        """
        Process CommonGen dataset and extract constraints from all examples.
        
        Args:
            data_file: Path to CommonGen JSON file
            output_file: Path to save extracted constraints
            max_examples: Maximum number of examples to process (for testing)
        """
        logger.info(f"Processing CommonGen dataset from {data_file}")
        
        # Load dataset
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_examples:
            data = data[:max_examples]
            logger.info(f"Processing first {max_examples} examples")
        
        results = []
        
        for i, example in enumerate(data):
            if i % 100 == 0:
                logger.info(f"Processing example {i+1}/{len(data)}")
            
            concepts = example['concepts']
            target = example['target']
            
            # Extract constraints
            constraints = self.extract_constraints_from_sentence(target, concepts)
            
            result = {
                'concept_set_idx': example.get('concept_set_idx', i),
                'concepts': concepts,
                'target': target,
                'constraints': constraints
            }
            
            results.append(result)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} examples with constraints to {output_file}")
        
        # Print statistics
        self._print_constraint_statistics(results)
        
        return results
    
    def _print_constraint_statistics(self, results: List[Dict]):
        """Print statistics about extracted constraints."""
        total_examples = len(results)
        unary_counts = []
        binary_counts = []
        triplet_counts = []
        
        for result in results:
            constraints = result['constraints']
            unary_counts.append(len(constraints['unary']))
            binary_counts.append(len(constraints['binary']))
            triplet_counts.append(len(constraints['triplet']))
        
        logger.info(f"Constraint Statistics for {total_examples} examples:")
        logger.info(f"  Unary constraints: avg={sum(unary_counts)/total_examples:.2f}, max={max(unary_counts)}")
        logger.info(f"  Binary constraints: avg={sum(binary_counts)/total_examples:.2f}, max={max(binary_counts)}")
        logger.info(f"  Triplet constraints: avg={sum(triplet_counts)/total_examples:.2f}, max={max(triplet_counts)}")
        
        # Show examples
        if results:
            logger.info("\nExample constraints:")
            example = results[0]
            logger.info(f"  Sentence: {example['target']}")
            logger.info(f"  Concepts: {example['concepts']}")
            logger.info(f"  Unary: {example['constraints']['unary']}")
            logger.info(f"  Binary: {example['constraints']['binary']}")
            logger.info(f"  Triplet: {example['constraints']['triplet']}")

def main(val_file, output_file):
    """Extract constraints from CommonGen validation set."""
    extractor = CommonGenConstraintExtractor()

    # Start with a small subset for testing
    extractor.process_commongen_dataset(val_file, output_file, max_examples=50)
    
    print("Constraint extraction complete!")

if __name__ == "__main__":
    main(val_file = "../all_other_files/commongen_constraints_validation.json", output_file = "../all_other_files/commongen_constraints_validation_output.json")