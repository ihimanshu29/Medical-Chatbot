# 🩺 MediBot: An AI-Powered Medical Chatbot

This is a Retrieval-Augmented Generation (RAG) based chatbot designed to answer medical questions based on a provided set of documents. It's built with LangChain, Groq, and Streamlit.

**Live Demo Link:** [Your Streamlit App URL Will Go Here]

---

## Features

-   **Conversational AI:** Ask medical questions in natural language.
-   **RAG Pipeline:** Retrieves relevant information from medical documents to provide accurate, context-aware answers.
-   **Fast Inference:** Powered by the high-speed Groq LPU™ Inference Engine.
-   **Built with Industry Standards:** Modular code structure, dependency management, and ready for deployment.

## Architecture

The application follows a standard RAG architecture:
1.  **Data Ingestion:** Medical documents (PDFs) are loaded and split into chunks.
2.  **Vector Store:** Text chunks are converted into embeddings (using Hugging Face models) and stored in a FAISS vector store.
3.  **Retrieval:** When a user asks a question, the system retrieves the most relevant chunks from the vector store.
4.  **Generation:** The retrieved context and the user's question are passed to a Large Language Model (Groq's Llama3) to generate a final answer.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    -   Create a `.env` file in the root directory.
    -   Add your API keys to the `.env` file:
        ```
        HUGGINGFACE_API_KEY="your_hf_key"
        GROQ_API_KEY="your_groq_key"
        ```
5.  **Build the knowledge base:**
    (Ensure you have your PDF documents inside the `data/` folder)
    ```bash
    python -m src.pipeline.training_pipeline
    ```
6.  **Run the Streamlit app:**
    ```bash
    python -m streamlit run src/app.py
    ```