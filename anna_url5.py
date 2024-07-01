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
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    except requests.RequestException as e:
        st.warning(f"Failed to retrieve content from {url}: {e}")
        return None

def clean_text(text):
    """Clean and pre-process the text."""
    # Example cleaning steps: removing extra spaces, newline characters, etc.
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    return ' '.join(text.split())

# Few-shot examples
few_shot_examples = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
    },
    {
        "question": "Who wrote 'To Kill a Mockingbird'?",
        "answer": "'To Kill a Mockingbird' was written by Harper Lee."
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "The largest planet in our solar system is Jupiter."
    }
]

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
            if content:
                content = clean_text(content)
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    url_chunks_map[chunk_id] = {'url': url.strip(), 'content': chunk}
                    chunk_id += 1

    if url_chunks_map:
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

            # Incorporate few-shot examples
            few_shot_prompt = "\n".join([f"Q: {example['question']}\nA: {example['answer']}" for example in few_shot_examples])
            few_shot_prompt += f"\nQ: {user_question}\nA:"

            # Use the retrieved documents and the user question to generate a response
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=few_shot_prompt)
            
            # Post-process the response to remove unnecessary parts
            def clean_response(response):
                response_lines = response.split('\n')
                useful_lines = [line for line in response_lines if 'unhelpful' not in line.lower() and 'i don\'t know' not in line.lower()]
                return ' '.join(useful_lines).strip()

            cleaned_response = clean_response(response)

            # Display the response
            st.write(cleaned_response)
            
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
