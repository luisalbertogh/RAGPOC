import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_doc(docs: list, chunk_details: dict) -> list:
    """Split document and return blocks

    Args:
        docs (list): List of documents to split.

    Returns:
        list: List of splitted blocks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_details['chunk_size'],  # chunk size (characters)
        chunk_overlap=chunk_details['chunk_overlap'],  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    logger.info(f"Split into {len(all_splits)} sub-documents.")
    return all_splits
