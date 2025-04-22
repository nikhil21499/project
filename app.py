import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set Streamlit page configuration
st.set_page_config(page_title="Web Content Q&A Tool")
st.title("Web Content Q&A Tool")

# Input fields for URLs and question
urls = st.text_area("Enter one or more webpage URLs (one per line):").splitlines()
question = st.text_input("Ask a question based on the above content:")

# Extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return ' '.join(soup.stripped_strings)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Unexpected error while processing {url}: {str(e)}")
        return ""

# Get the answer to the question using Langchain and OpenAI
def get_answer(texts, question):
    try:
        # Use SentenceTransformer to get embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        retriever = FAISS.from_texts(texts, embeddings).as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(api_key=openai_api_key, temperature=0), retriever=retriever)
        return qa_chain.run(question)
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't process the question at this time."

# Ingest and process content when the button is clicked
if st.button("Ingest & Answer"):
    if not urls or not question:
        st.warning("Please enter both URLs and a question.")
    else:
        with st.spinner("Scraping and processing content..."):
            docs = [extract_text_from_url(url) for url in urls if url.strip()]
            docs = [doc for doc in docs if doc]  # Filter out empty strings
            if docs:
                answer = get_answer(docs, question)
                st.success("Done!")
                st.markdown(f"**Answer:** {answer}")
            else:
                st.warning("No valid content extracted from the URLs.")
