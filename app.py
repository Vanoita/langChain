import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Langchain Streamlit Chatbot", layout="centered")
st.title("💬 Simple Langchain Chatbot")
st.markdown("---")

# Check for API key
if not openai_api_key:
    st.warning("Please set your OPENAI_API_KEY in the environment variables or `.env` file.")
    st.stop()

# Initialize Langchain components
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])
chain = LLMChain(llm=llm, prompt=prompt)

# Chat history (using Streamlit's session state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.run(question=user_input)
            st.markdown(response)
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("Built with Streamlit and Langchain")