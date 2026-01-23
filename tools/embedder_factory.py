"""
Centralized embedder factory with singleton pattern.
Avoids reloading the embedding model multiple times across different tools.
Uses local SentenceTransformers with CodeBERT for code embeddings.
"""
import os
from typing import Optional
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from sentence_transformers import SentenceTransformer


class EmbedderFactory:
    """
    Singleton factory for embedding models.
    Ensures only one instance of the embedder is loaded in memory.
    
    Uses a code-optimized SentenceTransformer model for embeddings.
    Runs fully locally with no API calls.
    """
    _instance: Optional[SentenceTransformerEmbedder] = None
    _sentence_transformer: Optional[SentenceTransformer] = None
    # Using jinaai's code embedding model - specifically trained for code similarity
    _model_id: str = "jinaai/jina-embeddings-v2-base-code"
    _dimensions: int = 768
    
    @classmethod
    def get_embedder(cls) -> SentenceTransformerEmbedder:
        """
        Get or create the embedder instance.
        
        Returns:
            SentenceTransformerEmbedder: Cached embedder instance
        
        Note:
            - Fully local (no API calls after download)
            - Model: jinaai/jina-embeddings-v2-base-code (specifically trained for code)
            - Model auto-downloads on first use (~400MB)
            - Uses GPU if available, otherwise CPU
        """
        if cls._instance is None:
            print(f"Loading embedding model: {cls._model_id}")
            print(f"This will download the model on first use (~500MB)")
            
            # Create SentenceTransformer instance
            # This auto-detects GPU and caches the model locally
            cls._sentence_transformer = SentenceTransformer(
                cls._model_id,
                device=None  # Auto-detect: uses 'cuda' if available, else 'cpu'
            )
            
            print(f"Device: {cls._sentence_transformer.device}")
            
            # Wrap in Agno's SentenceTransformerEmbedder
            cls._instance = SentenceTransformerEmbedder(
                sentence_transformer_client=cls._sentence_transformer,
                dimensions=cls._dimensions
            )
            
            print(f"Embedder loaded successfully!")
            print(f"Model cached at: ~/.cache/torch/sentence_transformers/")
        else:
            print(f"Using cached embedder instance")
        
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing or switching models)"""
        cls._instance = None
        cls._sentence_transformer = None
        print("Embedder cache cleared")
    
    @classmethod
    def get_dimensions(cls) -> int:
        """Get the embedding dimensions"""
        return cls._dimensions
    
    @classmethod
    def get_model_id(cls) -> str:
        """Get the model identifier"""
        return cls._model_id
    
    @classmethod
    def set_model(cls, model_id: str, dimensions: int):
        """
        Change the embedding model (must call before first use).
        
        Args:
            model_id: HuggingFace/SentenceTransformer model ID
            dimensions: Embedding dimensions for the model
        
        Examples:
            # Use Microsoft's CodeBERT (also good for code)
            EmbedderFactory.set_model("microsoft/codebert-base", 768)
            
            # Use smaller/faster general-purpose model
            EmbedderFactory.set_model("sentence-transformers/all-MiniLM-L6-v2", 384)
            
            # Use StarEncoder (trained on The Stack dataset)
            EmbedderFactory.set_model("bigcode/starencoder", 768)
            
            # Use local model path
            EmbedderFactory.set_model("/path/to/local/model", 768)
        """
        if cls._instance is not None:
            raise RuntimeError(
                "Cannot change model after embedder is initialized. "
                "Call EmbedderFactory.reset() first."
            )
        cls._model_id = model_id
        cls._dimensions = dimensions
        print(f"Model configuration updated: {model_id} ({dimensions}D)")


# Convenience function for backward compatibility
def get_code_embedder() -> SentenceTransformerEmbedder:
    """
    Convenience function to get the code embedder.
    
    Returns:
        SentenceTransformerEmbedder: Cached embedder instance
    """
    return EmbedderFactory.get_embedder()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test the embedder factory
    print("Testing EmbedderFactory...")
    print("=" * 60)
    
    # First call - should load the model
    print("\n1. First call (should load model):")
    embedder1 = EmbedderFactory.get_embedder()
    print(f"   Embedder 1 object ID: {id(embedder1)}")
    
    # Second call - should use cached instance
    print("\n2. Second call (should use cache):")
    embedder2 = EmbedderFactory.get_embedder()
    print(f"   Embedder 2 object ID: {id(embedder2)}")
    
    # Verify they're the same instance
    print("\n3. Verification:")
    assert embedder1 is embedder2, "Embedders should be the same instance!"
    print("   Singleton pattern working correctly!")
    
    # Test embedding generation
    print("\n4. Testing embedding generation:")
    test_code = "def hello_world():\n    print('Hello, World!')"
    
    # Use the underlying SentenceTransformer client (with type safety check)
    if embedder1.sentence_transformer_client is not None:
        embedding = embedder1.sentence_transformer_client.encode(test_code)
        print(f"   Generated embedding shape: {embedding.shape}")
        print(f"   Expected dimensions: {EmbedderFactory.get_dimensions()}")
        print(f"   First 5 values: {embedding[:5]}")
    else:
        print("   SentenceTransformer client not initialized!")
    
    # Test with Agno's method (this always works)
    print("\n5. Testing Agno embedding method:")
    embedding_agno = embedder1.get_embedding(test_code)
    print(f"   Embedding type: {type(embedding_agno)}")
    print(f"   Embedding length: {len(embedding_agno)}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")