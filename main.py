import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import textwrap

# Load smaller models for summarization and title generation from Hugging Face
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    title_generator = pipeline("text2text-generation", model="t5-small")
    return summarizer, title_generator

summarizer, title_generator = load_models()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, max_chunk_size=500):
    return textwrap.wrap(text, max_chunk_size)

# Function to summarize large text by processing it in chunks
def summarize_large_text(summarizer, text_chunks):
    summarized_text = ""
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summarized_text += summary + " "
    return summarized_text.strip()

# Streamlit app
st.title("PDF Summarizer and Title Generator (Large PDF Support)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write("Extracting text...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        st.write("Splitting text into chunks for processing...")
        text_chunks = split_text_into_chunks(pdf_text)

        st.write("Generating summary...")
        summary = summarize_large_text(summarizer, text_chunks)

        st.write("Generating title...")
        title = title_generator(summary, max_length=10, min_length=5, do_sample=False)[0]['generated_text']
        
        st.subheader("Generated Title")
        st.write(title)
        
        st.subheader("Summary")
        st.write(summary)
    else:
        st.write("No text found in the PDF.")
