#!/usr/bin/env python3
"""
Streamlit UI Integration Template

This template shows how to integrate the HybridRetriever with Streamlit UI.
Streamlit has a built-in local server - no Flask/FastAPI needed!

For local demo:
  Streamlit UI (port 8501) → Retriever → LLM → Display Results

Your teammate will implement the LLM generation part.
This template shows how to integrate the HybridRetriever with Streamlit.

Usage:
    streamlit run UI/codeReviewerGUI.py

Note: This template file shows the integration pattern.
      Actual implementation should be in your existing UI/codeReviewerGUI.py
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StreamlitUIConfig:
    """Configuration for Streamlit UI integration."""
    # Retriever config
    index_dir: str = "data/indexes"
    retrieval_k: int = 5  # Optimal K from validation
    similarity_threshold: float = 0.6  # Optimal threshold from validation
    
    # LLM config (to be filled by your teammate)
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512


class StreamlitCodeReviewIntegration:
    """
    Integration helper for Streamlit UI.
    
    YOU IMPLEMENT:
    - Retriever integration (DONE below)
    
    YOUR TEAMMATE IMPLEMENTS:
    - generate_review_with_llm()
    - Streamlit UI components (in UI/codeReviewerGUI.py)
    
    Note: Streamlit runs its own local server automatically.
          Just import this class in your Streamlit code and use it.
    """
    
    def __init__(self, config: StreamlitUIConfig):
        self.config = config
        self.retriever = None
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize retriever once at startup (use with @st.cache_resource)."""
        logger.info("Initializing HybridRetriever for Streamlit...")
        self.retriever = HybridRetriever(
            index_dir=self.config.index_dir,
            similarity_threshold=self.config.similarity_threshold,
            use_ivf_index=True,
            parallel_search=True
        )
        logger.info(f"Retriever ready (device: {self.retriever.device})")
    
    def retrieve_examples(self, code_patch: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant examples for the code patch.
        
        Args:
            code_patch: User's code patch/diff
            
        Returns:
            List of retrieved examples
        """
        return self.retriever.retrieve(
            patch=code_patch,
            top_k=self.config.retrieval_k,
            apply_similarity_threshold=True
        )
    
    def generate_review_with_llm(self, code_patch: str, retrieved_examples: List[Dict]) -> str:
        """
        Generate review comment using LLM with few-shot examples.
        
        TODO: YOUR TEAMMATE IMPLEMENTS THIS
        
        Args:
            code_patch: The code patch to review
            retrieved_examples: Retrieved examples from retriever
            
        Returns:
            Generated review comment
        """
        # Format examples for prompt
        formatted_examples = self.retriever.format_for_llm_prompt(retrieved_examples)
        
        # Build prompt
        prompt = f"""You are a code reviewer. Based on the following examples, provide a constructive review comment for the new code patch.

{formatted_examples}

New Code Patch to Review:
{code_patch}

Review Comment:"""
        
        # TODO: Call LLM API
        # response = call_llm(prompt, model=self.config.llm_model, ...)
        
        # PLACEHOLDER
        generated_review = "[TODO: LLM-generated review comment]"
        return generated_review
    
    def process_review_request(
        self, 
        code_patch: str,
        include_examples: bool = False,
        include_timing: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single code review request (main pipeline method).
        
        Args:
            code_patch: Code patch/diff to review
            include_examples: Include retrieved examples in response
            include_timing: Include timing information
            
        Returns:
            Response dictionary with review and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve examples
        retrieval_start = time.time()
        retrieved_examples = self.retrieve_examples(code_patch)
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generate review with LLM
        generation_start = time.time()
        generated_review = self.generate_review_with_llm(code_patch, retrieved_examples)
        generation_time = time.time() - generation_start
        
        # Build response
        response = {
            'review_comment': generated_review,
            'num_examples_used': len(retrieved_examples),
            'success': True
        }
        
        # Optional: include examples
        if include_examples:
            response['retrieved_examples'] = [
                {
                    'patch': ex.get('original_patch', '')[:200] + '...',
                    'review': ex.get('review_comment', '')[:200] + '...',
                    'score': ex.get('retrieval_score', 0),
                    'similarity': ex.get('semantic_similarity', 0)
                }
                for ex in retrieved_examples
            ]
        
        # Optional: include timing
        if include_timing:
            response['timing'] = {
                'retrieval_ms': retrieval_time * 1000,
                'generation_ms': generation_time * 1000,
                'total_ms': (time.time() - start_time) * 1000
            }
        
        return response


# =============================================================================
# STREAMLIT INTEGRATION EXAMPLE
# =============================================================================

def streamlit_integration_example():
    """
    Example of how to integrate with Streamlit.
    Copy this pattern to your UI/codeReviewerGUI.py file.
    """
    example_code = '''
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.hybrid_retriever import HybridRetriever

# Initialize retriever (cached - loads once)
@st.cache_resource
def load_retriever():
    """Load retriever with indexes. Runs once and cached."""
    with st.spinner("Loading retrieval system (~30s first time)..."):
        retriever = HybridRetriever(
            index_dir="data/indexes",
            similarity_threshold=0.6,
            use_ivf_index=True
        )
    return retriever

retriever = load_retriever()

# UI Components
st.title("🤖 AI Code Reviewer")

code_input = st.text_area("Enter your code patch:", height=200)

if st.button("Review Code"):
    if code_input.strip():
        # Retrieve examples
        with st.spinner("Retrieving similar examples..."):
            results = retriever.retrieve(patch=code_input, top_k=5)
        
        # Display examples
        st.subheader("📚 Retrieved Examples")
        for i, ex in enumerate(results, 1):
            with st.expander(f"Example {i}"):
                st.code(ex.get('original_patch', ''), language='diff')
                st.info(ex.get('review_comment', ''))
        
        # Generate review (TODO: Your teammate implements LLM call)
        # formatted = retriever.format_for_llm_prompt(results)
        # llm_review = call_your_llm(formatted)
        # st.success(llm_review)
    else:
        st.error("Please enter code to review")
'''
    
    print("\n" + "=" * 80)
    print("STREAMLIT INTEGRATION EXAMPLE")
    print("=" * 80)
    print("\nCopy this code to your UI/codeReviewerGUI.py:")
    print(example_code)
    print("\n" + "=" * 80)
    print("\nTo run:")
    print("  streamlit run UI/codeReviewerGUI.py")
    print("\nAccess at:")
    print("  http://localhost:8501")
    print("\n" + "=" * 80)


def demo_usage():
    """Demo of how to use the integration helper."""
    print("\n" + "=" * 80)
    print("Streamlit Integration Demo")
    print("=" * 80)
    
    # Initialize helper
    config = StreamlitUIConfig()
    integration = StreamlitCodeReviewIntegration(config)
    
    # Example code patch
    example_patch = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total
    """
    
    print("\nProcessing review request...")
    response = integration.process_review_request(
        code_patch=example_patch,
        include_examples=True,
        include_timing=True
    )
    
    print("\nResponse:")
    print(json.dumps(response, indent=2))
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Streamlit UI integration template")
    parser.add_argument('--demo', action='store_true',
                       help='Run demo')
    parser.add_argument('--example', action='store_true',
                       help='Show Streamlit integration example code')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_usage()
    elif args.example:
        streamlit_integration_example()
    else:
        print("\n" + "=" * 80)
        print("Streamlit UI Integration Template")
        print("=" * 80)
        print("\nThis template shows how to integrate HybridRetriever with Streamlit.")
        print("Streamlit runs its own local server - no Flask/FastAPI needed!\n")
        print("Options:")
        print("  --demo      Run a demo")
        print("  --example   Show Streamlit integration code\n")
        print("To run your actual Streamlit app:")
        print("  streamlit run UI/codeReviewerGUI.py\n")
        print("=" * 80)


if __name__ == '__main__':
    main()
