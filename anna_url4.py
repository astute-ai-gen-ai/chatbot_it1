import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import psutil
import time
import threading

# Load environment variables from .env file
load_dotenv()

# Retrieve the API token from the environment variable
sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not sec_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")

# Set the API token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

# Define the repository ID and initialize the HuggingFaceEndpoint with Mistral-7B
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=256, temperature=0.5, token=sec_key, timeout=30)

def fetch_website_content(url):
    """Fetch content from the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    else:
        raise ValueError(f"Failed to retrieve content from {url}, status code: {response.status_code}")

def clean_text(text):
    """Clean and pre-process the text."""
    # Example cleaning steps: removing extra spaces, newline characters, etc.
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    return ' '.join(text.split())

# Streamlit UI
st.header("Anna")
with st.sidebar:
    st.title("Web URLs")
    urls = st.text_area("Enter the URLs from which information has to be extracted (one per line)").splitlines()

# Extract the text
if urls:
    combined_content = ""
    url_chunks_map = {}
    chunk_id = 0

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    for url in urls:
        if url.strip():
            content = fetch_website_content(url.strip())
            content = clean_text(content)
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                url_chunks_map[chunk_id] = {'url': url.strip(), 'content': chunk}
                chunk_id += 1

    # Generating embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Creating vector store - FAISS
    vector_store = FAISS.from_texts([chunk['content'] for chunk in url_chunks_map.values()], embeddings)

    # Get user question
    user_question = st.text_input("Type your question here")

    # Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        
        # Find the relevant URL
        relevant_url = None
        for doc in match:
            for chunk_id, chunk in url_chunks_map.items():
                if doc.page_content == chunk['content']:
                    relevant_url = chunk['url']
                    break
            if relevant_url:
                break

        # Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        
        # Post-process the response
        response = response.replace('\n', ' ').strip()

        # Display the response
        st.write(response)
        
        # Display the relevant source URL at the bottom
        if relevant_url:
            st.write("\n\n**Source of the content:**")
            st.write(f"- {relevant_url}")

def log_resource_usage(interval=10):
    """Log the resource usage at the specified interval (in seconds)."""
    while True:
        # Get the current CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        # Log the usage
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {memory_info.percent}%")
        
        # Wait for the next interval
        time.sleep(interval)

# Start logging resource usage in a separate thread
threading.Thread(target=log_resource_usage, daemon=True).start()
