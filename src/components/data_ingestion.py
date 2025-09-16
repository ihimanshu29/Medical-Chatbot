import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

class DataIngestion:
    """
    Handles loading data from the source and splitting it into chunks.
    """
    def __init__(self):
        self.data_path = DATA_PATH
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP

    def load_documents(self) -> List[Document]:
        """Loads documents from the specified data directory."""
        print(f"Loading documents from {self.data_path}")
        loader = DirectoryLoader(
            self.data_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents

    def split_into_chunks(self, documents: List[Document]) -> List[Document]:
        """Splits the loaded documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        return chunks