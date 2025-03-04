import streamlit as st
import os
import tempfile
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool

# Title
st.title("🩺 ReAct RAG-based Medical Assistant")

# API Key Inputs
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Function to mask API keys for security
def mask_api_key(api_key):
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:] if api_key else ""

if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key

    st.text(f"✅ OpenAI Key Loaded: {mask_api_key(openai_api_key)}")
   
    # File Uploaders
    st.subheader("📂 Upload Patient Data")
    appointment_file = st.file_uploader("📅 Upload Appointments CSV", type=["csv"])
    history_file = st.file_uploader("📜 Upload Patient History CSV", type=["csv"])
    pdf_file = st.file_uploader("📄 Upload Last Appointment Summary PDF", type=["pdf"])

    if appointment_file and history_file and pdf_file:
        embeddings = OpenAIEmbeddings()

        # Function to save uploaded files temporarily
        def save_temp_file(uploaded_file, suffix):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                return temp_file.name

        # Save files temporarily
        temp_csv1_path = save_temp_file(appointment_file, ".csv")
        temp_csv2_path = save_temp_file(history_file, ".csv")
        temp_pdf_path = save_temp_file(pdf_file, ".pdf")

        # Load and Process Documents
        loader1 = CSVLoader(temp_csv1_path)
        loader2 = CSVLoader(temp_csv2_path)
        loader3 = PyPDFLoader(temp_pdf_path)

        try:
            documents1 = loader1.load()
            documents2 = loader2.load()
        except Exception as e:
            st.error(f"Error loading CSV files: {e}")

        try:
            documents3 = loader3.load()
        except Exception as e:
            st.error(f"Error loading PDF file: {e}")

        text_splitter = RecursiveCharacterTextSplitter()
        text_splitter_pdf = RecursiveCharacterTextSplitter(chunk_size=500)

        documents1 = text_splitter.split_documents(documents1)
        documents2 = text_splitter.split_documents(documents2)
        documents3 = text_splitter_pdf.split_documents(documents3)

        # Create Vector Stores
        vectorstore1 = FAISS.from_documents(documents1, embeddings)
        vectorstore2 = FAISS.from_documents(documents2, embeddings)
        vectorstore3 = FAISS.from_documents(documents3, embeddings)

        llm = ChatOpenAI(model="gpt-4")

        # Function to handle retrieval with error handling
        def safe_retrieve(retriever, query):
            try:
                return retriever.run(query)
            except Exception as e:
                return f"⚠️ Error retrieving data: {str(e)}"

        # Create retrieval tools
        tools = [
            Tool(name="appointments_tool", func=lambda q: safe_retrieve(RetrievalQA.from_chain_type(llm, retriever=vectorstore1.as_retriever()), q), description="Search appointments"),
            Tool(name="patient_history_tool", func=lambda q: safe_retrieve(RetrievalQA.from_chain_type(llm, retriever=vectorstore2.as_retriever()), q), description="Search patient history"),
            Tool(name="summary_tool", func=lambda q: safe_retrieve(RetrievalQA.from_chain_type(llm, retriever=vectorstore3.as_retriever()), q), description="Search last appointment summary"),
        ]

        llm_with_tools = llm.bind_tools(tools)
        sys_msg = {"role": "system", "content": "You are a helpful assistant for doctors. Provide a detailed patient summary based on available data."}

        # Patient ID input and question input
        patient_id = st.text_input("🆔 Enter Patient ID")
        patient_query = st.text_area("❓ Enter your question about the patient (e.g., medical conditions, appointment status, etc.)")

        if st.button("📝 Generate Summary") and patient_id and patient_query:
            # Combine the patient ID and query for processing
            query = f"Patient ID: {patient_id}. {patient_query}"

            # Use the tools to process the query
            response_texts = []

            # Appointments Tool
            response_texts.append(f"**Appointments Tool**: {safe_retrieve(RetrievalQA.from_chain_type(llm, retriever=vectorstore1.as_retriever()), query)}")
            # Patient History Tool
            response_texts.append(f"**Patient History Tool**: {safe_retrieve(RetrievalQA.from_chain_type(llm, retriever=vectorstore2.as_retriever()), query)}")
            # Summary Tool
            response_texts.append(f"**Summary Tool**: {safe_retrieve(RetrievalQA.from_chain_type(llm, retriever=vectorstore3.as_retriever()), query)}")

            # Display the response
            st.subheader("📋 Patient Summary")
            st.write("\n\n".join(response_texts))

