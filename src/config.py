import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Application Configuration ---
APP_TITLE = "MediBot 🩺 - Your AI Medical Assistant"
APP_FAVICON = "🩺"

# --- LLM and Embedding Model Configuration ---
# LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"  # free, fast Groq-hosted model
LLM_TEMPERATURE = 0.0
GROQ_API_KEY=os.environ["GROQ_API_KEY"]

# --- Vector Store Configuration ---
VECTOR_STORE_PATH = os.path.join("vectorstore", "db_faiss")
DATA_PATH = os.path.join("data")

# --- Data Ingestion Configuration ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50