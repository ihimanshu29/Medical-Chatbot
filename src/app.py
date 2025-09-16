import streamlit as st
from pipeline.prediction_pipeline import PredictionPipeline
from config import APP_TITLE, APP_FAVICON

# Initialize the prediction pipeline
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

# Streamlit page configuration
st.set_page_config(page_title=APP_TITLE, page_icon=APP_FAVICON)

# Header
st.title(APP_TITLE)
st.markdown(
    "Welcome! I'm an AI assistant trained on specific medical documents. "
    "Ask me a question about your health concerns."
)
st.warning("⚠️ **Disclaimer:** I am not a medical professional. My advices are strictly based on Gayle Encyclopedia. Please consult a doctor for any medical advice.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = pipeline.get_response(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})