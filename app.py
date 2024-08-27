from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from langchain.vectorstores import FAISS
import os
import streamlit as st
import streamlit.components.v1 as components

# Hide the main menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;} 
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Configure Gemini API LLM
API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")  # Replace with your Gemini API key

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Load and split the PDF
FILEPATH = "Dummy_Data.pdf"  # Path to the uploaded PDF

try:
    if not os.path.exists(FILEPATH):
        raise FileNotFoundError(f"Error: File not found at {FILEPATH}")
    elif not os.access(FILEPATH, os.R_OK):
        raise PermissionError(f"Error: Cannot read file at {FILEPATH}. Check permissions.")
    else:
        loader = PyPDFLoader(FILEPATH)
        data = loader.load()
        print("File loaded successfully!")
except Exception as e:
    st.error(f"Failed to load PDF file: {str(e)}")

# Split the PDF text into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

# Set up FAISS for document retrieval
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedder)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Define a simple prompt template for consistency
template = """You are a knowledgeable assistant. Please answer the following question based on the provided context.

Context: {context}

User: {question}
Assistant:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)
memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question")

# Function to query the PDF and get a response from the LLM
def query_pdf_with_llm(query):
    context = retriever.get_relevant_documents(query)
    if context:
        context_text = context[0].page_content
        formatted_prompt = prompt.format(context=context_text, question=query)
        response = model.generate_content(formatted_prompt)
        return response.text.strip()
    else:
        return "Sorry, I couldn't find any relevant information in the PDF."

# Load the custom CSS
def load_css():
    with open("styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Initialize the session state for chat history
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

# Simulate adding a message to the chat history
def on_click_callback():
    human_prompt = st.session_state.human_prompt
    st.session_state.history.append({"origin": "human", "message": human_prompt})
    ai_response = query_pdf_with_llm(human_prompt)
    st.session_state.history.append({"origin": "AI", "message": ai_response})
    st.session_state.human_prompt = ""  # Clear input box after submission

# Load CSS and initialize session state
load_css()
initialize_session_state()

# Create the title for the app
st.title("AI Chatbot 🤖")

# Container for the chat messages
chat_placeholder = st.container()
with chat_placeholder:
    for chat in st.session_state.history:
        if chat['origin'] == 'AI':
            with st.chat_message(name="assistant"):
                st.write(chat['message'])
        else:
            with st.chat_message(name="user"):
                st.write(chat['message'])

    for _ in range(3):
        st.markdown("")

# Input form for the user to type messages
with st.form("chat-form"):
    cols = st.columns((6, 1))
    with cols[0]:
        user_input = st.text_input(
            "Chat",
            value="Hello!",  # Remove placeholder text
            label_visibility="collapsed",
            key="human_prompt",
            autocomplete="off",  # Disable autocomplete suggestions
        )
    with cols[1]:
        submit_button = st.form_submit_button(
            "Submit",
            on_click=on_click_callback,
        )

# Placeholder for debugging information
st.caption(f"How Can I Help You?")

# JavaScript to allow submitting the form with the Enter key
components.html("""
<script>
const streamlitDoc = window.parent.document;
const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);
streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        submitButton.click();
        e.preventDefault();
    }
});
</script>
""", height=0, width=0)
