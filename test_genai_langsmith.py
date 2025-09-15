# pip install langchain_openai langsmith pytest python-dotenv
# Run with: pytest -v

import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable, evaluate
from langsmith.evaluation import EvaluationResult


# --- Setup ---
load_dotenv(override=True)
my_api_key = os.getenv("OPENAI_API_KEY")

# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=my_api_key)


# --- Function under test ---
@traceable
def ask_question(user_prompt) -> str:
    """LLM function under test."""
    # If LangSmith passes a dict, extract the actual string
    if isinstance(user_prompt, dict):
        user_prompt = user_prompt.get("input", "")

    system_msg = "You are a helpful assistant. Provide answer in one line."
    response = llm.invoke(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    return response.content


# --- Custom Evaluator ---
def keyword_eval(run, example, **kwargs) -> EvaluationResult:
    """Checks if expected keyword is in model output."""
    expected = example.inputs["expected"].lower()       # dataset expected
    actual = run.outputs["output"].lower()              # model run output

    success = expected in actual
    return EvaluationResult(
        key="keyword_check",
        score=1.0 if success else 0.0,
        explanation=f"Expected '{expected}' in '{actual}'"
    )

# --- Pytest-compatible wrapper ---
def test_langsmith_experiment():
    results = evaluate(
        ask_question,              # function under test
        "ds-2",                    # dataset name in LangSmith
        evaluators=[keyword_eval],  # evaluators
        experiment_prefix="lansmith_eval_test",  # experiment name prefix
    )
    for r in results:
        evals = r["evaluation_results"]["results"]
        print (f"evals {evals}")
        # loop over each evaluator result
        for e in evals:
            assert e.score >= 0.8, f"{e.key} failed for {r['example'].inputs['input']}: {e.explanation}"


# --- Run manually without pytest ---
if __name__ == "__main__":
    test_langsmith_experiment()
    print("All LangSmith evals passed.")
