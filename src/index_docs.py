import logging
from os import path

import yaml

from indexing.doc_embedder import Embedder
from indexing.doc_loader import load_markdown
from indexing.doc_splitter import split_doc

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_url_settings(filepath: str) -> list[dict]:
    """Load URLs settings file.

    Args:
        filepath (str): File path to the configuration file.

    Returns:
        list[dict]: List of url settings
    """
    # File not found
    if not path.isfile(filepath):
        logger.error(f"File not found: {filepath}")
        exit(1)

    # Load urls settings file
    with open(filepath, 'r') as file:
        urls = yaml.safe_load(file)

    return urls


# Entry point of the program
if __name__ == "__main__":

    # Load pdfs
    docs = load_markdown(dirpath='../docs')

    # Print documents
    # for doc in docs:
    #     logger.info(doc.page_content[0:50])

    # Split documents
    split_docs = split_doc(docs, chunk_details={'chunk_size': 1000, 'chunk_overlap': 200})

    # Embed documents
    embedder = Embedder(model='llama3.1', base_url='http://localhost:11434')
    doc_ids = embedder.embed_docs_in_memory(split_docs)
