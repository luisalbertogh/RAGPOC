import logging
import os

import bs4
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, UnstructuredMarkdownLoader

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_markdown(dirpath: list[str]) -> list:
    """Load Markdown files.

    Args:
        filepaths (list[str]): List of file paths to load.

    Returns:
        list[Document]: List of generated documents.
    """
    logger.info(f"Loading PDF files from: {dirpath}")

    # Get filepaths
    filepaths = []
    for root, _, files in os.walk(dirpath):
        for file in files:
            if file.endswith(".md"):
                filepaths.append(os.path.join(root, file))

    docs = []
    for filepath in filepaths:
        logger.info(f"Loading: {filepath}")
        loader = UnstructuredMarkdownLoader(filepath)
        for page in loader.lazy_load():
            docs.append(page)

    logger.info(f"Loaded {len(docs)} documents.")
    return docs


def load_pdfs(dirpath: list[str]) -> list:
    """Load PDF files.

    Args:
        filepaths (list[str]): List of file paths to load.

    Returns:
        list[Document]: List of generated documents.
    """
    logger.info(f"Loading PDF files from: {dirpath}")

    # Get filepaths
    filepaths = []
    for root, _, files in os.walk(dirpath):
        for file in files:
            if file.endswith(".pdf"):
                filepaths.append(os.path.join(root, file))

    docs = []
    for filepath in filepaths:
        logger.info(f"Loading: {filepath}")
        loader = PyPDFLoader(filepath)
        for page in loader.lazy_load():
            docs.append(page)

    logger.info(f"Loaded {len(docs)} pages.")
    return docs


def load_web_pages(urls: list[str], classes: set[str]) -> list:
    """Load web page content.

    Args:
        urls (list[str]): List of URLs to load.
        classes (set[str]): HTML classes to keep.

    Returns:
        list[Document]: List of generated documents.
    """
    logger.info(f"Loading web pages: {urls}")
    logger.info(f"Using classes: {classes}")

    # use specific classes.
    if classes:
        bs4_strainer = bs4.SoupStrainer(class_=classes)
    else:
        bs4_strainer = None

    # Load the content
    bs_kwargs = {}
    if bs4_strainer:
        bs_kwargs["parse_only"] = bs4_strainer

    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=bs_kwargs
    )

    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)

    logger.info(f"Loaded {len(docs)} documents.")
    return docs
