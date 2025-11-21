# LLM_QA_CLI.py
import os
import re
import string
import requests
from dotenv import load_dotenv

load_dotenv()


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return " ".join(tokens)


def get_llm_response(question):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

    payload = {
        "inputs": f"<s>[INST] You are a helpful Q&A assistant. Answer concisely. [/INST] {question} </s>",
        "parameters": {"max_new_tokens": 256, "temperature": 0.5}
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text'].split('[/INST]')[-1].strip() if result else "No response."
    else:
        return f"Error: {response.status_code}"


def main():
    print("=== LLM Question & Answer System (CLI) ===")
    print("Type 'quit' or 'exit' to end the program.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a valid question.\n")
            continue

        print(f"\nOriginal: {question}")

        processed = preprocess_text(question)
        print(f"Processed: {processed}\n")

        print("Thinking...")
        try:
            answer = get_llm_response(question)
            print(f"\nAnswer:\n{answer}\n")
            print("-" * 60 + "\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()