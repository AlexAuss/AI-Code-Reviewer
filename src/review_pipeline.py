import torch
from pathlib import Path
from models.quality_classifier import CodeQualityClassifier
from llm.code_reviewer import CodeReviewLLM
from transformers import AutoTokenizer
import json

class CodeReviewPipeline:
    def __init__(
        self,
        quality_model_path: str,
        model_name: str = "microsoft/codebert-base",
        llm_type: str = "openai",
        llm_model: str = "gpt-3.5-turbo"
    ):
        # Load quality assessment model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.quality_model = CodeQualityClassifier(model_name).to(self.device)
        self.quality_model.load_state_dict(torch.load(quality_model_path, map_location=self.device))
        self.quality_model.eval()
        
        # Initialize LLM reviewer
        self.llm_reviewer = CodeReviewLLM(model_type=llm_type, model_name=llm_model)
    
    def review_code(self, diff: str, max_length: int = 512) -> dict:
        """
        Perform complete code review including quality assessment and detailed review.
        """
        # 1. Assess code quality
        quality_score = self._assess_quality(diff, max_length)
        
        # 2. Generate detailed review using LLM
        review_result = self.llm_reviewer.generate_review(
            diff=diff,
            quality_score=quality_score
        )
        
        # 3. Combine results
        return {
            "quality_score": float(quality_score),
            "review": review_result.get("review", ""),
            "model_used": review_result.get("model_used", ""),
            "error": review_result.get("error", None)
        }
    
    def _assess_quality(self, diff: str, max_length: int) -> float:
        """
        Assess code quality using CodeBERT model.
        """
        # Tokenize input
        encoding = self.tokenizer(
            diff,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.quality_model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            quality_score = probabilities[:, 1].item()  # Probability of "good" class
        
        return quality_score

def main():
    # Example usage
    base_path = Path(__file__).parent.parent
    model_path = str(base_path / "models/saved/quality_classifier.pth")
    
    # Initialize pipeline
    pipeline = CodeReviewPipeline(
        quality_model_path=model_path,
        llm_type="openai",  # or "huggingface" if you prefer
        llm_model="gpt-3.5-turbo"  # or your preferred model
    )
    
    # Example diff
    diff = """
    def calculate_total(items):
    -    total = 0
    -    for item in items:
    -        total += item
    -    return total
    +    return sum(items)
    """
    
    # Get review
    review_result = pipeline.review_code(diff)
    print(json.dumps(review_result, indent=2))

if __name__ == "__main__":
    main()