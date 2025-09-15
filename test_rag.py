# pip install -q sentence-transformers
# pip install ragas datasets
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_correctness


import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from openai import OpenAI

import os

load_dotenv(override=True)
my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=my_api_key)


def get_top_k_similar(query, k=3):
    # Step 1: Define Sample Documents
    documents = [
        {"section": "Pay Policies", "content": "Employees are paid bi-weekly via direct deposit."},
        {"section": "Leave of Absence", "content": "Employees must submit a leave request for approval."},
        {"section": "Internet Use", "content": "Company internet must be used for work-related tasks only."}
    ]

    texts = [doc["content"] for doc in documents]

    # Step 2: Generate Embeddings using SentenceTransformers
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_vectors = model.encode(texts, convert_to_tensor=True)

    # Step 3: Encode the query
    query_vec = model.encode(query, convert_to_tensor=True)

    # Step 4: Compute cosine similarities
    similarities = util.cos_sim(query_vec, doc_vectors)[0].cpu().numpy()

    # Step 5: Get top-k indices (sorted by similarity, descending)
    top_k_idx = np.argsort(similarities)[::-1][:k]

    # Step 6: Collect top documents
    top_docs = [documents[int(idx)] for idx in top_k_idx]
    return top_docs


# --- Build dataset for Ragas ---
def build_dataset():
    query = "How often do employees get paid?"
    retrieved_docs = get_top_k_similar(query, 3)

    gold_answer = "Employees are paid bi-weekly via direct deposit."

    examples = [
        {
            "question": query,
            "answer": gold_answer,  # system or gold answer
            "contexts": [d["content"] for d in retrieved_docs],
            "reference": gold_answer,    # gold reference (string)
            "ground_truths": [gold_answer]  # list of valid gold answers
        }
    ]
    return Dataset.from_list(examples)


if __name__ == "__main__":
    query = "How often do employees get paid?"
    top_docs = get_top_k_similar(query, 3)
    dataset = build_dataset()
    results = evaluate(dataset, 
                metrics=[context_precision, context_recall, faithfulness, answer_correctness])
    print(results)

# def test_ragas_retriever_only():
#     dataset = build_dataset()
#     results = evaluate(dataset, metrics=[context_precision, context_recall])
#     print(results)

#     # Assert retrieval quality
#     assert results["context_recall"] > 0.8
#     assert results["context_precision"] > 0.5
