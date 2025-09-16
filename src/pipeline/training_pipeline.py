# To run this file: python -m src.pipeline.training_pipeline

from components.data_ingestion import DataIngestion
from components.vector_store import VectorStoreManager

def main():
    print("Starting data ingestion and vector store creation pipeline...")
    
    # Step 1: Ingest and chunk data
    data_ingestion = DataIngestion()
    documents = data_ingestion.load_documents()
    chunks = data_ingestion.split_into_chunks(documents)
    
    # Step 2: Create and save vector store
    vector_store_manager = VectorStoreManager()
    vector_store_manager.create_and_save_vector_store(chunks)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()