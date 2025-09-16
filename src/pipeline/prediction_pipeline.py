from components.vector_store import VectorStoreManager
from components.llm_interface import LLMInterface

class PredictionPipeline:
    def __init__(self):
        self.vector_store = VectorStoreManager().load_vector_store()
        self.qa_chain = LLMInterface(self.vector_store).create_qa_chain()

    def get_response(self, query: str) -> str:
        """
        Gets a response from the QA chain.
        """
        result = self.qa_chain({"query": query})
        return result["result"]