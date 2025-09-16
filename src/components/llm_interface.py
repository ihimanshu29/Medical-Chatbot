from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from config import GROQ_MODEL_NAME, LLM_TEMPERATURE, GROQ_API_KEY

class LLMInterface:
    """
    Handles interactions with the Groq Large Language Model.
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store
        print(f"DEBUG: GROQ API Key Value: {GROQ_API_KEY}")
        print(f"DEBUG: Type of GROQ API Key: {type(GROQ_API_KEY)}")
        self.llm = ChatGroq(
            model_name=GROQ_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            groq_api_key=GROQ_API_KEY
        )
        self.prompt_template = self._create_prompt_template()
        self.memory = ConversationBufferMemory(
            input_key="question",
            memory_key="chat_history"
        )

    @staticmethod
    def _create_prompt_template() -> PromptTemplate:
        """Creates the prompt template for the LLM."""
        template = """
        Use the following pieces of context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        
        Helpful Answer:
        """
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )

    def create_qa_chain(self):
        """Creates the conversational retrieval QA chain."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt_template,
                "memory": self.memory
            }
        )