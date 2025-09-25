#!/usr/bin/env python3
"""
Enhanced NeuroStructural Decoding Implementation

This code includes all base NeuroLogic Decoding features plus our structural extensions:
1. Trie-based constraint representation from NeuroLogic
2. Advanced beam search with proper candidate ranking
3. Clause-based CNF constraint logic
4. Beta penalty parameter and satisfaction tolerance
5. Structural constraints (our extension)
6. Dependency parsing integration (our extension)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Union
from enum import Enum
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from collections import defaultdict
import copy
import math
from operator import attrgetter
from parsers import DirectParserAdapter

# Global parser instance to avoid re-initialization
_GLOBAL_PARSER = None

def get_global_parser():
    """Get or create global parser instance with proper error handling"""
    global _GLOBAL_PARSER
    if _GLOBAL_PARSER is None:
        try:
            _GLOBAL_PARSER = DirectParserAdapter("best_parser.th")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize parser: {e}")
    
    return _GLOBAL_PARSER

# Import original constraint classes from NeuroLogic
Phrase = List[int]
Literal = Tuple[Phrase, bool]
RawConstraintList = List[Phrase]
ClauseConstraintList = List[List[Literal]]

class Trie:
    """
    Trie data structure for efficient constraint tracking (from NeuroLogic)
    """
    def __init__(self, raw_phrases: Optional[RawConstraintList] = None, 
                 parent_arc: int = None, parent_trie: 'Trie' = None):
        self.final_ids = set()
        self.children = {}
        self.parent_arc = parent_arc
        self.parent_trie = parent_trie
        
        if raw_phrases:
            for phrase in raw_phrases:
                self.add_phrase(phrase)
    
    def add_phrase(self, phrase: List[int]):
        """Add a phrase to the trie"""
        if len(phrase) == 1:
            self.final_ids.add(phrase[0])
        else:
            next_word = phrase[0]
            if next_word not in self.children:
                self.children[next_word] = Trie(parent_arc=next_word, parent_trie=self)
            self.children[next_word].add_phrase(phrase[1:])
    
    def step(self, word_id: int) -> Optional['Trie']:
        """Move to next state in trie"""
        return self.children.get(word_id)
    
    def final(self) -> Set[int]:
        """Get final word IDs at this node"""
        return self.final_ids
    
    def __len__(self) -> int:
        return len(self.children) + len(self.final_ids)

class PositiveState:
    """
    Tracks positive constraints (must include)
    """
    def __init__(self, positive_trie: Trie, state: List[Trie] = None, 
                 met_phrases: RawConstraintList = None):
        self.root = positive_trie
        self.state = state if state else [self.root]
        self.met_phrases = met_phrases if met_phrases else []
    
    def allowed(self) -> Set[int]:
        """Get allowed next words"""
        allowed_words = set()
        for state in self.state:
            allowed_words.update(state.final())
            allowed_words.update(state.children.keys())
        return allowed_words
    
    def advance(self, word_id: int) -> 'PositiveState':
        """Advance state with new word"""
        new_state = []
        new_met = self.met_phrases.copy()
        
        for state in self.state:
            # Check if word completes a phrase
            if word_id in state.final():
                # Phrase completed - trace back to get full phrase
                phrase = self._trace_phrase(state, word_id)
                new_met.append(phrase)
            
            # Check if word continues a phrase
            if word_id in state.children:
                new_state.append(state.children[word_id])
        
        # Always allow starting new phrases
        if word_id in self.root.children:
            new_state.append(self.root.children[word_id])
        
        return PositiveState(self.root, new_state, new_met)
    
    def _trace_phrase(self, state: Trie, word_id: int) -> List[int]:
        """Trace back to get complete phrase"""
        phrase = []
        current = state
        while current.parent_trie is not None:
            phrase.insert(0, current.parent_arc)
            current = current.parent_trie
        phrase.append(word_id)
        return phrase

class NegativeState:
    """
    Tracks negative constraints (must avoid)
    """
    def __init__(self, negative_trie: Trie, state: Trie = None):
        self.root = negative_trie
        self.state = state if state else self.root
    
    def consume(self, word_id: int) -> 'NegativeState':
        """Consume word and update state"""
        if word_id in self.state.children:
            return NegativeState(self.root, self.state.children[word_id])
        else:
            return NegativeState(self.root, self.root)
    
    def avoid(self) -> Set[int]:
        """Get words to avoid at current state"""
        return self.state.final()

class StructuralConstraint:
    """
    Base class for structural constraints (our extension)
    """
    def __init__(self, words: List[str], constraint_type: str):
        self.words = words
        self.constraint_type = constraint_type
    
    def check_satisfaction(self, parse_tree: Dict, tokenizer=None) -> bool:
        """Check if constraint is satisfied in parse tree"""
        raise NotImplementedError
    
    def get_word_ids(self, tokenizer) -> List[int]:
        """Convert words to token IDs"""
        return [tokenizer.encode(word, add_special_tokens=False)[0] for word in self.words]

class UnaryConstraint(StructuralConstraint):
    """Constraint on single word's syntactic role"""
    def __init__(self, word: str, relation: str):
        super().__init__([word], 'unary')
        self.word = word.lower()
        self.relation = relation
    
    def check(self, word: str, relation: str = None) -> bool:
        """Check if word satisfies the unary constraint"""
        return self.word == word.lower() and (relation is None or self.relation == relation)
    
    def check_satisfaction(self, parse_tree: Dict, tokenizer=None) -> bool:
        """Check if constraint is satisfied in parse tree"""
        for token in parse_tree.get('tokens', []):
            if (token.get('text', '').lower() == self.word and 
                token.get('deprel', '') == self.relation):
                return True
        return False
    
    def evaluate(self, parse_tree: Dict) -> bool:
        """Alias for check_satisfaction"""
        return self.check_satisfaction(parse_tree)
    
    def __str__(self):
        return f"UnaryConstraint({self.word}, {self.relation})"

class BinaryConstraint(StructuralConstraint):
    """Binary structural constraint (our extension)"""
    def __init__(self, head: str, dependent: str, relation: str):
        super().__init__([head, dependent], "binary")
        self.head = head
        self.dependent = dependent
        self.relation = relation
    
    def check(self, head: str, dependent: str, relation: str) -> bool:
        """Check if words satisfy the binary constraint"""
        return (self.head.lower() == head.lower() and 
                self.dependent.lower() == dependent.lower() and 
                self.relation == relation)
    
    def check_satisfaction(self, parse_tree: Dict, tokenizer=None) -> bool:
        tokens = parse_tree.get('tokens', [])
        head_idx = dep_idx = None
        
        for i, token_data in enumerate(tokens):
            if token_data.get('text', '').lower() == self.head.lower():
                head_idx = i
            if token_data.get('text', '').lower() == self.dependent.lower():
                dep_idx = i
        
        if head_idx is None or dep_idx is None:
            return False
            
        dep_token = tokens[dep_idx]
        return (dep_token.get('head', -1) == head_idx + 1 and
                dep_token.get('deprel', '') == self.relation)
    
    def __str__(self):
        return f"BinaryConstraint({self.head} -[{self.relation}]-> {self.dependent})"

class TripletConstraint(StructuralConstraint):
    """Triplet structural constraint (our extension)"""
    def __init__(self, head: str, dependent1: str, dependent2: str, relation1: str = 'nsubj', relation2: str = 'obj'):
        super().__init__([head, dependent1, dependent2], "triplet")
        self.head = head
        self.dependent1 = dependent1
        self.dependent2 = dependent2
        self.relation1 = relation1
        self.relation2 = relation2
    
    def check(self, head: str, dep1: str, dep2: str, rel1: str, rel2: str) -> bool:
        """Check if triplet satisfies the constraint"""
        return (self.head.lower() == head.lower() and 
                self.dependent1.lower() == dep1.lower() and 
                self.dependent2.lower() == dep2.lower() and
                self.relation1 == rel1 and self.relation2 == rel2)
    
    def check_satisfaction(self, parse_tree: Dict, tokenizer=None) -> bool:
        tokens = parse_tree.get('tokens', [])
        head_idx = dep1_idx = dep2_idx = None
        
        for i, token_data in enumerate(tokens):
            text = token_data.get('text', '').lower()
            if text == self.head.lower():
                head_idx = i
            elif text == self.dependent1.lower():
                dep1_idx = i
            elif text == self.dependent2.lower():
                dep2_idx = i
        
        if any(idx is None for idx in [head_idx, dep1_idx, dep2_idx]):
            return False
        
        dep1_token = tokens[dep1_idx]
        dep2_token = tokens[dep2_idx]
        
        dep1_correct = (dep1_token.get('head', -1) == head_idx + 1 and
                       dep1_token.get('deprel', '') == self.relation1)
        dep2_correct = (dep2_token.get('head', -1) == head_idx + 1 and
                      dep2_token.get('deprel', '') == self.relation2)
        
        return dep1_correct and dep2_correct
    
    def __str__(self):
        return f"TripletConstraint({self.head} -[{self.relation1}]-> {self.dependent1}, -[{self.relation2}]-> {self.dependent2})"

class Clause:
    """
    Represents a logical clause with positive and negative literals
    """
    def __init__(self, idx: int, positive: List[Phrase], negative: List[Phrase], 
                 satisfy: float, structural_constraints: List[StructuralConstraint] = None):
        self.idx = idx
        self.positive = positive
        self.negative = negative
        self.satisfy = satisfy
        self.structural_constraints = structural_constraints or []
    
    def evaluate_lexical(self, met_phrases: List[Phrase]) -> bool:
        """Evaluate lexical constraints"""
        if not self.positive:
            return True
        return any(phrase in met_phrases for phrase in self.positive)
    
    def evaluate_structural(self, parse_tree: Dict, tokenizer=None) -> bool:
        """Evaluate structural constraints (our extension)"""
        if not self.structural_constraints:
            return True
        return any(constraint.check_satisfaction(parse_tree, tokenizer) 
                  for constraint in self.structural_constraints)
    
    def evaluate(self, met_phrases: List[Phrase], parse_tree: Dict = None, tokenizer=None) -> bool:
        """Evaluate both lexical and structural constraints"""
        lexical_satisfied = self.evaluate_lexical(met_phrases)
        structural_satisfied = self.evaluate_structural(parse_tree, tokenizer) if parse_tree else True
        return lexical_satisfied or structural_satisfied

class ConstrainedCandidate:
    """
    Represents a candidate hypothesis with score and constraints
    """
    def __init__(self, row: int, col: int, score: float, hypothesis: 'EnhancedConstrainedHypothesis'):
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis
        self.rank = score  # Will be updated with penalty

class EnhancedConstrainedHypothesis:
    """
    Enhanced hypothesis class combining NeuroLogic base with structural extensions
    """
    def __init__(self, constraint_list: ClauseConstraintList, eos_id: Union[int, List[int]],
                 structural_constraints: List[StructuralConstraint] = None,
                 tokenizer=None, parser=None):
        self.eos_id = eos_id if isinstance(eos_id, list) else [eos_id]
        self.clauses = []
        self.tokenizer = tokenizer
        self._parser = parser  # Store parser reference
        self.structural_constraints = structural_constraints or []
        self.seq_len = 0
        self.last_processed = 0
        self.max_process = 0
        
        # Process lexical constraints (from NeuroLogic)
        hard_neg_pool, soft_neg_pool, pos_pool = [], [], []
        for idx, clause in enumerate(constraint_list):
            if not clause:
                continue
            pos_phrases = [l[0] for l in clause if l[1]]
            neg_phrases = [l[0] for l in clause if not l[1]]
            
            if not pos_phrases and len(neg_phrases) == 1:
                hard_neg_pool.extend(neg_phrases)
            elif neg_phrases:
                soft_neg_pool.extend(neg_phrases)
                self.clauses.append(Clause(idx, pos_phrases, neg_phrases, True))
            elif pos_phrases:
                pos_pool.extend(pos_phrases)
                self.clauses.append(Clause(idx, pos_phrases, [], False))
        
        # Add structural constraints (our extension)
        if structural_constraints:
            for i, sc in enumerate(structural_constraints):
                structural_clause = Clause(
                    idx=len(self.clauses) + i,
                    positive=[],
                    negative=[],
                    satisfy=False,
                    structural_constraints=[sc]
                )
                self.clauses.append(structural_clause)
        
        # Initialize states
        self.hard_negative_state = NegativeState(Trie(hard_neg_pool)) if hard_neg_pool else None
        self.soft_negative_state = NegativeState(Trie(soft_neg_pool)) if soft_neg_pool else None
        self.positive_state = PositiveState(Trie(pos_pool)) if pos_pool else None
        
        self.orders = []
        self.in_process = []
        self.max_process = 0
    
    @property
    def parser(self):
        """Get parser, using global instance to avoid re-initialization"""
        if not hasattr(self, '_parser') or self._parser is None:
            self._parser = get_global_parser()
        return self._parser
    
    def size(self) -> int:
        """Number of constraints"""
        return len(self.clauses)
    
    def num_met(self) -> int:
        """Number of satisfied constraints"""
        return sum(int(c.satisfy) for c in self.clauses)
    
    def finished(self) -> bool:
        """Check if all constraints are satisfied"""
        return all(c.satisfy for c in self.clauses)
    
    def is_valid(self, word_id: int) -> bool:
        """Check if word is valid to generate"""
        return self.finished() or word_id not in self.eos_id
    
    def avoid(self) -> Set[int]:
        """Get words to avoid"""
        banned = set()
        if self.hard_negative_state:
            banned.update(self.hard_negative_state.avoid())
        return banned
    
    def advance(self, word_id: int, text: str = None) -> 'EnhancedConstrainedHypothesis':
        """
        Advance hypothesis with new word (enhanced with structural analysis)
        """
        # Create new hypothesis with empty constraint list
        obj = EnhancedConstrainedHypothesis(
            constraint_list=[],
            eos_id=self.eos_id,
            structural_constraints=copy.deepcopy(self.structural_constraints),
            tokenizer=self.tokenizer,
            parser=None  # Don't copy parser, create new if needed
        )
        
        # Copy all attributes manually
        obj.clauses = [copy.deepcopy(clause) for clause in self.clauses]
        obj.positive_state = copy.deepcopy(self.positive_state) if self.positive_state else None
        obj.hard_negative_state = copy.deepcopy(self.hard_negative_state) if hasattr(self, 'hard_negative_state') and self.hard_negative_state else None
        obj.soft_negative_state = copy.deepcopy(self.soft_negative_state) if hasattr(self, 'soft_negative_state') and self.soft_negative_state else None
        obj.orders = copy.deepcopy(self.orders) if hasattr(self, 'orders') else []
        obj.in_process = copy.deepcopy(self.in_process) if hasattr(self, 'in_process') else []
        obj.seq_len = self.seq_len
        obj.last_processed = self.last_processed
        obj.max_process = self.max_process
        obj._parser = None  # Will be created on demand
        
        # Update negative states
        if obj.hard_negative_state:
            obj.hard_negative_state = obj.hard_negative_state.consume(word_id)
        if obj.soft_negative_state:
            obj.soft_negative_state = obj.soft_negative_state.consume(word_id)
        
        # Update positive state
        if obj.positive_state:
            obj.positive_state = obj.positive_state.advance(word_id)
            
            # Check for newly satisfied lexical constraints
            if obj.positive_state.met_phrases:
                for phrase in obj.positive_state.met_phrases:
                    if not self.positive_state or phrase not in self.positive_state.met_phrases:
                        obj._mark_lexical_constraint_satisfied(phrase)
        
        # Update structural constraints (our extension)
        if text:
            try:
                # Use global parser instance to avoid re-initialization
                obj._parser = get_global_parser()
                if obj._parser is None:
                    raise RuntimeError("Parser is not initialized")
                parse_tree = obj._parser.parse_sentence(text)
                obj._update_structural_constraints(parse_tree)
            except Exception as e:
                raise RuntimeError(f"Parser error during structural constraint update: {e}")
        
        return obj
    
    def _mark_lexical_constraint_satisfied(self, phrase: List[int]):
        """Mark lexical constraint as satisfied"""
        for clause in self.clauses:
            if phrase in clause.positive and not clause.satisfy:
                clause.satisfy = True
                self.orders.append(clause.idx)
                break
    
    def _update_structural_constraints(self, parse_tree: Dict):
        """Update structural constraint satisfaction"""
        for clause in self.clauses:
            if clause.structural_constraints and not clause.satisfy:
                if clause.evaluate_structural(parse_tree, self.tokenizer):
                    clause.satisfy = True
                    self.orders.append(clause.idx)

class EnhancedNeuroStructuralDecoder:
    """
    Enhanced NeuroStructural Decoder with full NeuroLogic base + structural extensions
    """
    def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer,
                 beam_size: int = 5, lambda_penalty: float = 1.0, max_length: int = 50,
                 prune_factor: int = 50, sat_tolerance: int = 2, beta: float = 0.1,
                 early_stop: float = None):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.lambda_penalty = lambda_penalty
        self.max_length = max_length
        self.prune_factor = prune_factor
        self.sat_tolerance = sat_tolerance
        self.beta = beta
        self.early_stop = early_stop
        self.parser = get_global_parser()
    
    def decode(self, input_ids: torch.Tensor, constraints: List[EnhancedConstrainedHypothesis],
               **kwargs) -> List[Dict]:
        """
        Main decoding function with enhanced constraint handling
        """
        batch_size = input_ids.shape[0]
        # Get vocab size from model config, not tokenizer (they might differ for T5)
        vocab_size = self.model.config.vocab_size if hasattr(self.model.config, 'vocab_size') else self.tokenizer.vocab_size
        device = input_ids.device
        
        # Store original encoder input for seq2seq models
        encoder_input_ids = input_ids.clone()
        
        # Initialize beam
        beam_scores = torch.zeros((batch_size, self.beam_size), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        # Initialize decoder input tokens for seq2seq models
        cur_len = 1
        # For T5, use decoder_start_token_id
        if hasattr(self.model.config, 'is_encoder_decoder') and self.model.config.is_encoder_decoder:
            start_token_id = getattr(self.model.config, 'decoder_start_token_id', None)
            if start_token_id is None:
                start_token_id = self.tokenizer.pad_token_id
        else:
            # For decoder-only models
            start_token_id = self.tokenizer.bos_token_id
            if start_token_id is None:
                start_token_id = self.tokenizer.cls_token_id
            if start_token_id is None:
                start_token_id = self.tokenizer.pad_token_id
        
        if start_token_id is None:
            start_token_id = 0  # Fallback to 0
            
        decoder_input_ids = torch.full((batch_size * self.beam_size, cur_len), 
                                      start_token_id,
                                      dtype=torch.long, device=device)
        
        # Initialize constraints for each beam
        beam_constraints = []
        for i in range(batch_size):
            for j in range(self.beam_size):
                beam_constraints.append(copy.deepcopy(constraints[i]))
        
        finished_hypotheses = [[] for _ in range(batch_size)]
        
        while cur_len < self.max_length:
            # Get model predictions
            # For T5 models, we need to provide decoder_input_ids
            if hasattr(self.model.config, 'is_encoder_decoder') and self.model.config.is_encoder_decoder:
                # This is a seq2seq model like T5
                # Expand encoder input for beam search
                expanded_encoder_input = encoder_input_ids.repeat_interleave(self.beam_size, dim=0)
                outputs = self.model(input_ids=expanded_encoder_input, decoder_input_ids=decoder_input_ids)
            else:
                # This is a decoder-only model
                outputs = self.model(decoder_input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(batch_size, self.beam_size * vocab_size)
            
            # Apply enhanced topk selection with constraints
            (new_scores, new_tokens, new_constraints, 
             new_num_mets) = self._enhanced_topk_selection(
                timestep=cur_len,
                batch_size=batch_size,
                scores=next_scores,
                constraints=beam_constraints,
                input_ids=decoder_input_ids  # Use decoder input ids for constraint checking
            )
            
            # Update beam
            beam_scores = torch.tensor(new_scores, device=device).view(-1)
            beam_constraints = new_constraints
            
            # Convert new_tokens to tensor and update decoder_input_ids
            new_tokens_tensor = torch.tensor(new_tokens, device=device).view(batch_size, self.beam_size)
            beam_indices = new_tokens_tensor // vocab_size
            token_indices = new_tokens_tensor % vocab_size
            
            new_decoder_input_ids = []
            for batch_idx in range(batch_size):
                for beam_idx in range(self.beam_size):
                    old_beam_idx = batch_idx * self.beam_size + beam_indices[batch_idx, beam_idx]
                    new_tokens_for_beam = torch.cat([
                        decoder_input_ids[old_beam_idx],
                        token_indices[batch_idx, beam_idx:beam_idx+1]
                    ])
                    new_decoder_input_ids.append(new_tokens_for_beam)
            
            decoder_input_ids = torch.stack(new_decoder_input_ids)
            cur_len += 1
            
            # Check for finished hypotheses
            self._check_finished_hypotheses(
                batch_size, new_tokens_tensor, new_scores, new_constraints,
                decoder_input_ids, finished_hypotheses
            )
        
        return self._format_results(finished_hypotheses)
    
    def _enhanced_topk_selection(self, timestep: int, batch_size: int, 
                               scores: torch.Tensor, constraints: List[EnhancedConstrainedHypothesis],
                               input_ids: torch.Tensor) -> Tuple:
        """
        Enhanced topk selection combining NeuroLogic approach with structural constraints
        """
        # Get vocab size properly
        vocab_size = self.model.config.vocab_size if hasattr(self.model.config, 'vocab_size') else self.tokenizer.vocab_size
        scores_np = scores.detach().cpu().numpy()  # Detach gradients first
        
        all_candidates = []
        
        for batch_idx in range(batch_size):
            batch_candidates = []
            batch_start = batch_idx * self.beam_size
            batch_end = batch_start + self.beam_size
            batch_scores = scores_np[batch_idx].reshape(self.beam_size, vocab_size)
            batch_constraints = constraints[batch_start:batch_end]
            
            # Generate candidates for this batch
            for beam_idx in range(self.beam_size):
                hyp = batch_constraints[beam_idx]
                beam_scores = batch_scores[beam_idx]
                
                # Get allowed words from positive constraints
                allowed_words = hyp.positive_state.allowed() if hyp.positive_state else set()
                
                # Get top-k words
                top_k_indices = np.argsort(beam_scores)[::-1][:self.beam_size * 2]
                
                for word_idx in top_k_indices:
                    if hyp.is_valid(word_idx):
                        # Create candidate
                        score = beam_scores[word_idx]
                        
                        # Generate text for structural analysis
                        current_text = self._decode_tokens(input_ids[batch_start + beam_idx].tolist())
                        new_text = current_text + " " + self.tokenizer.decode([word_idx])
                        
                        # Advance hypothesis
                        new_hyp = hyp.advance(word_idx, new_text)
                        
                        # Apply beta penalty for constraint processing
                        if new_hyp.max_process > 0:
                            score -= self.beta * math.log(new_hyp.max_process)
                        
                        candidate = ConstrainedCandidate(
                            row=batch_start + beam_idx,
                            col=word_idx, 
                            score=score,
                            hypothesis=new_hyp
                        )
                        candidate.rank = score / (timestep + 1)
                        batch_candidates.append(candidate)
            
            # Apply tolerance filtering
            if batch_candidates:
                max_satisfy = max(c.hypothesis.num_met() for c in batch_candidates)
                filtered_candidates = [
                    c for c in batch_candidates 
                    if c.hypothesis.num_met() >= max_satisfy - self.sat_tolerance
                ]
                batch_candidates = filtered_candidates
            
            # Group by constraint satisfaction pattern and select
            selected_candidates = self._group_and_select_candidates(batch_candidates)
            
            # Ensure we have exactly beam_size candidates for this batch
            while len(selected_candidates) < self.beam_size:
                if batch_candidates:
                    # Add the best remaining candidate
                    best_remaining = max(batch_candidates, key=lambda x: x.score)
                    selected_candidates.append(best_remaining)
                    batch_candidates.remove(best_remaining)
                else:
                    # Create dummy candidate if no more available
                    dummy_candidate = ConstrainedCandidate(
                        row=batch_start,
                        col=self.tokenizer.eos_token_id or 1,  # Use EOS or fallback
                        score=float('-inf'),
                        hypothesis=constraints[batch_start]  # Use first constraint as fallback
                    )
                    selected_candidates.append(dummy_candidate)
            
            all_candidates.extend(selected_candidates[:self.beam_size])
        
        # Format outputs
        new_scores = [c.score for c in all_candidates]
        new_tokens = [c.row * vocab_size + c.col for c in all_candidates]
        new_constraints = [c.hypothesis for c in all_candidates]
        new_num_mets = [c.hypothesis.num_met() for c in all_candidates]
        
        return new_scores, new_tokens, new_constraints, new_num_mets
    
    def _group_and_select_candidates(self, candidates: List[ConstrainedCandidate]) -> List[ConstrainedCandidate]:
        """
        Group candidates by constraint satisfaction pattern and select best
        """
        if not candidates:
            return []
        
        # Sort by rank
        candidates.sort(key=attrgetter('rank'), reverse=True)
        
        # Group by constraint satisfaction pattern
        groups = defaultdict(list)
        for candidate in candidates:
            # Use tuple of satisfied constraint indices as grouping key
            key = tuple(sorted([c.idx for c in candidate.hypothesis.clauses if c.satisfy]))
            groups[key].append(candidate)
        
        # Select best from each group
        selected = []
        for group in groups.values():
            group.sort(key=attrgetter('rank'), reverse=True)
            selected.extend(group[:max(1, self.beam_size // len(groups))])
        
        # Sort final selection by score
        selected.sort(key=attrgetter('score'), reverse=True)
        return selected
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Convert token IDs to text"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def _check_finished_hypotheses(self, batch_size: int, new_tokens: torch.Tensor,
                                 new_scores: List[float], new_constraints: List[EnhancedConstrainedHypothesis],
                                 input_ids: torch.Tensor, finished_hypotheses: List[List]):
        """Check for finished hypotheses and add to results"""
        vocab_size = len(new_scores) // (batch_size * self.beam_size)
        
        for batch_idx in range(batch_size):
            for beam_idx in range(self.beam_size):
                idx = batch_idx * self.beam_size + beam_idx
                token_id = new_tokens[batch_idx, beam_idx] % vocab_size
                constraint = new_constraints[idx]
                
                if (token_id == self.tokenizer.eos_token_id or 
                    (constraint.finished() and token_id in constraint.eos_id)):
                    
                    finished_hypotheses[batch_idx].append({
                        'tokens': input_ids[idx].tolist(),
                        'score': new_scores[idx],
                        'constraints_met': constraint.num_met(),
                        'total_constraints': constraint.size()
                    })
    
    def _format_results(self, finished_hypotheses: List[List]) -> List[Dict]:
        """Format final results"""
        results = []
        for batch_hyps in finished_hypotheses:
            if batch_hyps:
                # Sort by score and take best
                batch_hyps.sort(key=lambda x: x['score'], reverse=True)
                best_hyp = batch_hyps[0]
                best_hyp['text'] = self._decode_tokens(best_hyp['tokens'])
                results.append(best_hyp)
            else:
                results.append({
                    'tokens': [],
                    'text': '',
                    'score': float('-inf'),
                    'constraints_met': 0,
                    'total_constraints': 0
                })
        
        return results