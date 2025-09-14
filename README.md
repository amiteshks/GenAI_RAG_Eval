# GenAI_RAG_Eval

GenAI_RAG_Eval is a framework for evaluating **Generative AI** and **Retrieval-Augmented Generation (RAG)** applications.  
It provides utilities, metrics, and workflows to test the quality, reliability, and performance of LLM-based systems that depend on external knowledge retrieval.

---

## 🚀 Features

- **Golden Dataset Evaluations**  
  Compare generated responses against curated reference answers.

- **RAG-Specific Metrics**  
  - **Faithfulness** – Does the response stay true to retrieved documents?  
  - **Groundedness** – Are claims supported by evidence?  
  - **Relevance** – How relevant are retrieved chunks to the query?  
  - **Answer Correctness** – Does the response match the expected outcome?  

- **Custom Evaluators**  
  Define rule-based, keyword-based, or LLM-as-judge evaluators.

- **Dataset & Run Tracking**  
  Supports experiment logging and comparison across multiple test runs.

- **Integration Friendly**  
  Works with [LangSmith](https://smith.langchain.com/), LangChain, and OpenAI APIs.  
  Easily extendable for other evaluation pipelines.

