import re
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


class PolicyState(TypedDict):
    question: str
    answer: str


retr = None


def loadNotes(path):
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    return [
        Document(page_content=chunk, metadata={"source": str(path), "chunk_id": index})
        for index, chunk in enumerate(chunks)
    ]


def buildRet():
    global retr

    if retr is None:
        p = Path("pdf.pdf")
        docs = loadNotes(p)

        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        store = FAISS.from_documents(docs, emb)
        retr = store.as_retriever(search_kwargs={"k": 3})

    return retr


import os
import httpx
from dotenv import load_dotenv

load_dotenv()


def answerPolicy(question: str) -> str:
    retriever = buildRet()

    try:
        docs = retriever._get_relevant_documents(question, run_manager=None)
    except TypeError:
        docs = retriever._get_relevant_documents(question)

    if not docs:
        return "I cannot answer from the PDF content yet. The retrieved material does not contain enough evidence for a safe answer."

    context = "\n\n".join(
        f"Chunk {doc.metadata.get('chunk_id')}: {doc.page_content}" for doc in docs
    )

    prompt = """You are a careful assistant answering questions from a PDF.
Use only the provided context. If the context does not contain enough evidence to answer the question, say clearly: "I cannot find the answer in the provided context." Do not make things up.

Context:
{context}

Question:
{question}

Answer in 2-3 sentences. Mention the relevant chunk id in brackets at the end of the answer (e.g., [Chunk X])."""

    filled_prompt = prompt.format(context=context, question=question)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return f"{filled_prompt}\n\n[system-note: prompt-based answer step - GROQ_API_KEY not set]"

    try:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a precise technical QA assistant."},
                {"role": "user", "content": filled_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 150
        }
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=10.0)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Error from Groq API (status {response.status_code}): {response.text}"
    except Exception as e:
        return f"Failed to generate answer via Groq LLM: {str(e)}"


def route(state: PolicyState):
    return {"answer": answerPolicy(state["question"])}


graph_builder = StateGraph(PolicyState)
graph_builder.add_node("policy_node", route)
graph_builder.add_edge(START, "policy_node")
graph_builder.add_edge("policy_node", END)
graph = graph_builder.compile()


if __name__ == "__main__":
    sample_questions = [
        "What is boosting in ensemble learning?",
        "How does AdaBoost differ from Gradient Boosting?",
        "What are the differences between XGBoost, LightGBM, and CatBoost?",
        "What is the capital of Egypt?",
    ]

    print("Small RAG system")
    print("==========================================")
    print("Document choice: the provided PDF file at pdf.pdf")
    print("Example questions and answers:")
    for question in sample_questions:
        result = graph.invoke({"question": question, "answer": ""})
        print(f"Q: {question}")
        print(f"A: {result['answer']}\n")
