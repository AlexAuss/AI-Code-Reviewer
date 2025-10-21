from typing import Dict, List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import torch
from pathlib import Path
import logging

from models.quality_classifier import CodeQualityClassifier
from embeddings.vector_store import CodeReviewVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridCodeReviewer:
    def __init__(
        self,
        quality_model_path: str,
        vector_store_path: str,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize the hybrid code reviewer
        Args:
            quality_model_path: Path to trained CodeBERT model
            vector_store_path: Path to FAISS vector store
            llm_model: Name of the LLM model to use
            temperature: Temperature for LLM generation
        """
        self.model_type = model_type
        self.model_name = model_name
        
        if model_type == "huggingface":
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
    
    def generate_review(self, 
                       diff: str, 
                       quality_score: float, 
                       max_length: int = 500) -> Dict[str, str]:
        """
        Generate a detailed code review using the LLM.
        Args:
            diff: The code diff to review
            quality_score: Quality score from CodeBERT (0 to 1)
            max_length: Maximum length of the generated review
        Returns:
            Dictionary containing review and suggestions
        """
        # Construct the prompt
        prompt = self._construct_prompt(diff, quality_score)
        
        if self.model_type == "openai":
            return self._generate_openai_review(prompt, max_length)
        else:
            return self._generate_hf_review(prompt, max_length)
    
    def _construct_prompt(self, diff: str, quality_score: float) -> str:
        """Construct a detailed prompt for the LLM"""
        return f"""As an expert code reviewer, analyze the following code changes and provide a detailed review. 
The automated quality assessment score is: {quality_score:.2f} (0=poor, 1=excellent)

Code Changes:
{diff}

Please provide:
1. Overall assessment
2. Specific issues or concerns
3. Suggestions for improvement
4. Best practices that could be applied
5. Security considerations (if any)

Review:"""

    def _generate_openai_review(self, prompt: str, max_length: int) -> Dict[str, str]:
        """Generate review using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer with deep knowledge of software engineering best practices."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.7
            )
            review = response.choices[0].message.content
            
            return {
                "review": review,
                "model_used": self.model_name
            }
            
        except Exception as e:
            return {
                "error": f"Error generating review: {str(e)}",
                "model_used": self.model_name
            }
    
    def _generate_hf_review(self, prompt: str, max_length: int) -> Dict[str, str]:
        """Generate review using HuggingFace model"""
        try:
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7
            )
            
            review = outputs[0]['generated_text'].replace(prompt, "").strip()
            
            return {
                "review": review,
                "model_used": self.model_name
            }
            
        except Exception as e:
            return {
                "error": f"Error generating review: {str(e)}",
                "model_used": self.model_name
            }