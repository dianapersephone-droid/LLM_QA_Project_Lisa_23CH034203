# app.py
import streamlit as st
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(page_title="LLM Q&A System", page_icon="🤖", layout="centered")

st.title("Question & Answer System")
st.markdown("### Powered by Hugging Face + Mistral-7B | Made by [Your Name]")

# Sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("Enter any question below and get instant answers using open AI models.")
    st.info("Free tier via Hugging Face Inference API.")

# Input
question = st.text_input("Ask anything:", placeholder="e.g., What is the capital of France?")

if st.button("Get Answer", type="primary"):
    if question.strip():
        with st.spinner("Getting answer..."):
            try:
                API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
                headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
                payload = {
                    "inputs": f"<s>[INST] You are a concise and accurate assistant. [/INST] {question} </s>",
                    "parameters": {"max_new_tokens": 300, "temperature": 0.6}
                }
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    answer = result[0]['generated_text'].split('[/INST]')[-1].strip() if result else "No response."
                    st.success("Answer:")
                    st.markdown(f"**{answer}**")

                    with st.expander("Show details"):
                        st.write(f"Original: {question}")
                        st.write(f"Model: Mistral-7B-Instruct-v0.3")
                else:
                    st.error(f"API Error: {response.status_code} - Check your token.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("NLP Project 2 – 21/11/2025 | Free Hugging Face API | Deployed on Render")