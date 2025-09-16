from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from config import EMBEDDING_MODEL_REPO_ID, VECTOR_STORE_PATH

class VectorStoreManager:
    """
    Manages the FAISS vector store.
    """
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_REPO_ID,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store_path = VECTOR_STORE_PATH

    def create_and_save_vector_store(self, chunks: List[Document]):
        """Creates a FAISS vector store from document chunks and saves it."""
        print("Creating and saving vector store...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model
        )
        vector_store.save_local(self.vector_store_path)
        print(f"Vector store saved at {self.vector_store_path}")

    def load_vector_store(self) -> FAISS:
        """Loads an existing FAISS vector store."""
        print(f"Loading vector store from {self.vector_store_path}")
        return FAISS.load_local(
            self.vector_store_path,
            self.embedding_model,
            allow_dangerous_deserialization=True 
        )