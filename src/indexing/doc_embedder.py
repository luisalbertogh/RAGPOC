import logging

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Embedder:
    """Class to embed documents.
    """
    # Embedding model
    _embeddings = None

    def __init__(self, model: str, base_url: str):
        """Initialize the class.
        """
        self.model = model
        self.base_url = base_url
        # Use embeddings for Ollama
        self._embeddings = OllamaEmbeddings(model=self.model, base_url=self.base_url)

    def embed_docs_in_memory(self, splits: list) -> list:
        """Embed splits and return embedded docs.

        Args:
            splits (list): List of ckunks to embed

        Returns:
            list: List of embedded docs
        """
        vector_store = InMemoryVectorStore(self._embeddings)
        ids = vector_store.add_documents(documents=splits)
        logger.info(f"Embedded {len(ids)} documents.")
        return ids
