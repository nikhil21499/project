import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.set_page_config(page_title=" Web Content Q&A Tool")

st.title(" Web Content Q&A Tool")

urls = st.text_area("Enter one or more webpage URLs (one per line):").splitlines()
question = st.text_input("Ask a question based on the above content:")

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return ' '.join(soup.stripped_strings)
    except:
        return ""

def get_answer(texts, question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever = FAISS.from_texts(texts, embeddings).as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)
    return qa_chain.run(question)

if st.button("Ingest & Answer"):
    if not urls or not question:
        st.warning("Please enter both URLs and a question.")
    else:
        with st.spinner("Scraping and processing content..."):
            docs = [extract_text_from_url(url) for url in urls if url.strip()]
            answer = get_answer(docs, question)
        st.success("Done!")
        st.markdown(f"**Answer:** {answer}")
