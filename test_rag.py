# pip install -q sentence-transformers
# pip install ragas datasets
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    context_relevance,
    faithfulness,
    answer_relevance,
    answer_correctness,
    answer_similarity,
    semantic_similarity,
)

import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from openai import OpenAI
import os

# --- Load API Key ---
load_dotenv(override=True)
my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=my_api_key)


# --- Retriever: Get top-k docs ---
def get_top_k_similar(query, k=3):
    documents = [
        {"section": "Pay Policies", "content": "Employees are paid bi-weekly via direct deposit."},
        {"section": "Leave of Absence", "content": "Employees must submit a leave request for approval."},
        {"section": "Internet Use", "content": "Company internet must be used for work-related tasks only."}
    ]

    texts = [doc["content"] for doc in documents]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_vectors = model.encode(texts, convert_to_tensor=True)
    query_vec = model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_vec, doc_vectors)[0].cpu().numpy()
    top_k_idx = np.argsort(similarities)[::-1][:k]

    return [documents[int(idx)] for idx in top_k_idx]


# --- Generator: Use OpenAI with retrieved docs ---
def generate_answer(query, contexts):
    context_text = " ".join(contexts)
    prompt = f"Answer the question based only on the following context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return completion.choices[0].message.content.strip()


# --- Build dataset for Ragas ---
def build_dataset():
    query = "How often do employees get paid?"
    retrieved_docs = get_top_k_similar(query, 3)
    contexts = [d["content"] for d in retrieved_docs]

    # Gold reference
    gold_answer = "Employees are paid bi-weekly via direct deposit."

    # Generate answer using LLM
    model_answer = generate_answer(query, contexts)

    examples = [
        {
            "question": query,
            "answer": model_answer,        # LLM-generated answer
            "contexts": contexts,         
            "reference": gold_answer,     
            "ground_truths": [gold_answer]
        }
    ]
    return Dataset.from_list(examples)


if __name__ == "__main__":
    dataset = build_dataset()

    # --- All metrics across retriever, generator, and end-to-end ---
    all_metrics = [
        context_recall, context_precision, context_relevance,   # Retriever
        faithfulness, answer_relevance, answer_correctness, answer_similarity,  # Generator
        semantic_similarity  # End-to-End
    ]

    results = evaluate(dataset, metrics=all_metrics)

    print("\n🔹 Full RAG Evaluation Results")
    print(results)
