import streamlit as st


from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


def get_context(input_dict):
    question = input_dict["question"]
    nodes = retriever.retrieve(question)
    return "\n".join([node.get_content() for node in nodes])

context_retriever = RunnableLambda(get_context)


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key= api_key
)

# Prompt 
system_prompt = """
You are a Senior Financial Analyst.

Instructions:
- Answer ONLY using the provided context.
- Do NOT use prior knowledge.
- Do NOT guess or hallucinate.
- If the answer is not explicitly found, say: "Not found in the provided context."

Formatting Rules:
- Use bullet points for all financial data
- Be precise and concise
- Include numbers, percentages, and dates when available

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(system_prompt)

# Chain 
chain = (
    {
        "context": context_retriever,
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="Financial RAG Chatbot")

st.title("📊 Financial Analyst Chatbot")

# input box
user_question = st.text_input("Ask a question from 10-K:")

# button
if st.button("Ask") and user_question:
    with st.spinner("Analyzing..."):
        response = chain.invoke({
            "question": user_question
        })

    st.write("### 📌 Answer:")
    st.write(response)