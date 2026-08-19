import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

logger = logging.getLogger(__name__)


LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".json": JSONLoader,
}


def load_all_documents(data_dir: str) -> list[Any]:
    data_path = Path(data_dir).resolve()
    documents: list[Any] = []

    if not data_path.exists():
        logger.warning("Data directory not found: %s", data_path)
        return documents

    for suffix, loader_cls in LOADER_REGISTRY.items():
        files = list(data_path.glob(f"**/*{suffix}"))
        logger.info("Found %s files for %s", len(files), suffix)
        for file_path in files:
            try:
                loader = loader_cls(str(file_path))
                loaded = loader.load()
                documents.extend(loaded)
            except Exception as exc:
                logger.exception("Failed loading file %s: %s", file_path, exc)

    logger.info("Total loaded documents: %s", len(documents))
    return documents
