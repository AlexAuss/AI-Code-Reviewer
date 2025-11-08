from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeReviewVectorStore:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """Initialize the vector store with a specific embedding model"""
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None

    def create_vector_store(self, dataset_paths):
        """
        Create vector store from multiple dataset files
        Args:
            dataset_paths (list): List of paths to JSONL dataset files
        """
        documents = []
        metadata = []
        
        for path in dataset_paths:
            logger.info(f"Processing dataset: {path}")
            with open(path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Extract relevant information
                    diff = data.get('diff', '')
                    comment = data.get('comment', '')  # From Comment_Generation
                    review = data.get('review', '')    # From Code_Refinement
                    quality = data.get('label', '')    # From Diff_Quality_Estimation
                    
                    # Combine information for embedding
                    text = f"Code Change:\n{diff}\n"
                    if comment:
                        text += f"Review Comment: {comment}\n"
                    if review:
                        text += f"Suggested Changes: {review}\n"
                    if quality:
                        text += f"Quality Assessment: {quality}\n"
                    
                    documents.append(text)
                    metadata.append({
                        "source_file": Path(path).name,
                        "quality_label": quality,
                        "has_review": bool(review),
                        "has_comment": bool(comment)
                    })
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        self.vector_store = FAISS.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadata
        )
        return self.vector_store

    def save_vector_store(self, save_path):
        """Save the vector store to disk"""
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(save_dir))
        logger.info(f"Vector store saved to {save_dir}")

    def load_vector_store(self, load_path):
        """Load a vector store from disk"""
        if not Path(load_path).exists():
            raise ValueError(f"Path does not exist: {load_path}")
        
        self.vector_store = FAISS.load_local(load_path, self.embeddings)
        logger.info(f"Vector store loaded from {load_path}")
        return self.vector_store

    def similarity_search(self, query, k=3):
        """
        Search for similar code reviews
        Args:
            query (str): Code diff to search for
            k (int): Number of results to return
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        return self.vector_store.similarity_search(query, k=k)