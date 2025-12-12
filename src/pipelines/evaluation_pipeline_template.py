#!/usr/bin/env python3
"""
Evaluation Pipeline Template

This pipeline runs the complete evaluation flow on test dataset:
  Test Dataset → Retriever → LLM → Evaluation Metrics

Your teammate will implement the LLM generation and evaluation parts.
This template shows how to integrate the HybridRetriever.

Usage:
    python src/pipelines/evaluation_pipeline_template.py --test-dataset Datasets/Unified_Dataset/test.jsonl
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    # Data paths
    test_dataset_path: str = "Datasets/Unified_Dataset/test.jsonl"
    output_path: str = "data/evaluation/test_results.json"
    
    # Retriever config
    index_dir: str = "data/indexes"
    retrieval_k: int = 5  # Optimal K from validation
    similarity_threshold: float = 0.6  # Optimal threshold from validation
    
    # LLM config (to be filled by your teammate)
    llm_model: str = "gpt-4"  # or whatever model you use
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    
    # Evaluation config
    checkpoint_interval: int = 100
    max_samples: int = None  # None = all samples


class EvaluationPipeline:
    """
    Complete evaluation pipeline: Retrieval → LLM → Metrics
    
    YOU IMPLEMENT:
    - Retriever integration (DONE below)
    
    YOUR TEAMMATE IMPLEMENTS:
    - generate_review_with_llm()
    - compute_evaluation_metrics()
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.retriever = None
        self.results = []
        
    def initialize_retriever(self):
        """Initialize the hybrid retriever."""
        logger.info("Initializing HybridRetriever...")
        self.retriever = HybridRetriever(
            index_dir=self.config.index_dir,
            similarity_threshold=self.config.similarity_threshold,
            use_ivf_index=True,
            parallel_search=True
        )
        logger.info("Retriever initialized successfully")
    
    def load_test_data(self) -> List[Dict]:
        """Load test dataset."""
        logger.info(f"Loading test data from {self.config.test_dataset_path}")
        
        data = []
        with open(self.config.test_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        if self.config.max_samples:
            data = data[:self.config.max_samples]
        
        logger.info(f"Loaded {len(data)} test samples")
        return data
    
    def retrieve_examples(self, query_patch: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant examples using hybrid retriever.
        
        Args:
            query_patch: Code patch to find similar examples for
            
        Returns:
            List of retrieved examples with metadata
        """
        return self.retriever.retrieve(
            patch=query_patch,
            top_k=self.config.retrieval_k,
            apply_similarity_threshold=True
        )
    
    def generate_review_with_llm(self, query_patch: str, retrieved_examples: List[Dict]) -> str:
        """
        Generate review comment using LLM with few-shot examples.
        
        TODO: YOUR TEAMMATE IMPLEMENTS THIS
        
        Args:
            query_patch: The code patch to review
            retrieved_examples: Retrieved examples from retriever
            
        Returns:
            Generated review comment
        """
        # Format examples for prompt
        formatted_examples = self.retriever.format_for_llm_prompt(retrieved_examples)
        
        # TODO: Build prompt with examples
        prompt = f"""You are a code reviewer. Based on the following examples, provide a review comment for the new code patch.

{formatted_examples}

New Code Patch to Review:
{query_patch}

Review Comment:"""
        
        # TODO: Call LLM API
        # response = call_llm(prompt, model=self.config.llm_model, ...)
        
        # PLACEHOLDER - replace with actual LLM call
        generated_review = "[TODO: LLM-generated review comment]"
        return generated_review
    
    def compute_evaluation_metrics(self, generated_review: str, ground_truth_review: str) -> Dict[str, float]:
        """
        Compute evaluation metrics comparing generated vs ground truth.
        
        TODO: YOUR TEAMMATE IMPLEMENTS THIS
        
        Args:
            generated_review: LLM-generated review
            ground_truth_review: Ground truth review from dataset
            
        Returns:
            Dictionary of metrics (BLEU, ROUGE, etc.)
        """
        # TODO: Implement metric computation
        # - BLEU score
        # - ROUGE scores
        # - Semantic similarity
        # - etc.
        
        # PLACEHOLDER
        metrics = {
            'bleu': 0.0,
            'rouge_1': 0.0,
            'rouge_2': 0.0,
            'rouge_l': 0.0,
            'semantic_similarity': 0.0
        }
        return metrics
    
    def process_sample(self, sample: Dict, sample_idx: int) -> Dict[str, Any]:
        """
        Process a single test sample through the complete pipeline.
        
        Args:
            sample: Test sample from dataset
            sample_idx: Index of sample for tracking
            
        Returns:
            Result dictionary with all pipeline outputs
        """
        query_patch = sample.get('original_patch', '') or sample.get('patch', '')
        ground_truth_review = sample.get('review_comment', '')
        
        if not query_patch:
            logger.warning(f"Sample {sample_idx}: Empty patch, skipping")
            return None
        
        # Step 1: Retrieve examples
        retrieved_examples = self.retrieve_examples(query_patch)
        
        # Step 2: Generate review with LLM
        generated_review = self.generate_review_with_llm(query_patch, retrieved_examples)
        
        # Step 3: Compute metrics
        metrics = self.compute_evaluation_metrics(generated_review, ground_truth_review)
        
        # Package result
        result = {
            'sample_idx': sample_idx,
            'query_patch': query_patch,
            'ground_truth_review': ground_truth_review,
            'generated_review': generated_review,
            'num_retrieved_examples': len(retrieved_examples),
            'retrieved_example_ids': [r.get('doc_id') for r in retrieved_examples],
            'metrics': metrics
        }
        
        return result
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("Starting evaluation pipeline...")
        
        # Initialize components
        self.initialize_retriever()
        
        # Load test data
        test_data = self.load_test_data()
        
        # Process samples
        start_time = time.time()
        
        for idx, sample in enumerate(test_data):
            result = self.process_sample(sample, idx)
            
            if result:
                self.results.append(result)
            
            # Progress update
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(test_data) - idx - 1) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {idx + 1}/{len(test_data)} "
                    f"({(idx + 1) / len(test_data) * 100:.1f}%) | "
                    f"Rate: {rate:.2f} samples/sec | "
                    f"ETA: {remaining / 60:.1f} min"
                )
            
            # Checkpoint
            if (idx + 1) % self.config.checkpoint_interval == 0:
                self.save_results(checkpoint=True)
        
        # Final save
        self.save_results(checkpoint=False)
        
        # Compute aggregate metrics
        self.compute_aggregate_metrics()
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation complete in {total_time / 60:.1f} minutes")
    
    def save_results(self, checkpoint: bool = False):
        """Save evaluation results to JSON."""
        output_path = Path(self.config.output_path)
        if checkpoint:
            output_path = output_path.with_suffix('.checkpoint.json')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'num_samples': len(self.results),
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def compute_aggregate_metrics(self):
        """Compute aggregate metrics across all samples."""
        if not self.results:
            logger.warning("No results to aggregate")
            return
        
        # Compute averages
        avg_metrics = {}
        for metric_name in self.results[0]['metrics'].keys():
            values = [r['metrics'][metric_name] for r in self.results]
            avg_metrics[f'avg_{metric_name}'] = sum(values) / len(values)
        
        logger.info("\n" + "=" * 80)
        logger.info("AGGREGATE EVALUATION METRICS")
        logger.info("=" * 80)
        for metric_name, value in avg_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation pipeline for test dataset")
    parser.add_argument('--test-dataset', type=str, 
                       default='Datasets/Unified_Dataset/test.jsonl',
                       help='Path to test dataset')
    parser.add_argument('--output', type=str,
                       default='data/evaluation/test_results.json',
                       help='Path to save evaluation results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Create config
    config = EvaluationConfig(
        test_dataset_path=args.test_dataset,
        output_path=args.output,
        max_samples=args.max_samples
    )
    
    # Run pipeline
    pipeline = EvaluationPipeline(config)
    pipeline.run_evaluation()


if __name__ == '__main__':
    main()
