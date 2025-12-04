# %%
##Dependencies

#pip install ragas
#pip install ipykernel
#pip install langsmith
#pip install langchain_huggingface
#pip install langsmith
#pip install openevals
#pip install --pre -U langchain langchain-openai

# %%
# Importing libraries

import os
import uuid
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from chatbot_tools import chat



# %%
# Load variables from .env into the process
load_dotenv()

# Read config values (with defaults where it makes sense)
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "my_chatbot_eval")

# Retrieve the API keys and other secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")          
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# Set as environment variables explicitly (optional but often useful)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY or ""
os.environ["HF_TOKEN"] = HF_TOKEN or ""
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT


print("OpenAI API key loaded and set as environment variable.")
print("Pinecone API key loaded and set as environment variable.")
print("LangSmith API key loaded and set as environment variable.")
print("HF token loaded and set as environment variable.")
print("LangSmith workspace and LangChain tracing config loaded.")

# %% [markdown]
# ## Evaluation code to trace in Langsmith

# %%
client = Client()
project_name = "my_chatbot_eval"

# 1. Test data
questions = [
    "What is a security mindset?",
    "Explain PPP.",
    "What are the domains of the exam with percentages?",
]

ground_truths = [
    "A security mindset is the ability to evaluate risk and constantly seek out and identify potential or actual breaches...",
    "Point-to-point protocol is a data link layer protocol used to establish a direct connection between two networking devices...",
    "The domains of the Security+ exam are: General Security Concepts 12%; Threats, Vulnerabilities, and Mitigations 22%;  Security Architecture 18%; Security Architecture 28%; Security Program Management and Oversight 20%",
]

# 2. Evaluators
correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4o",
)

def relevance_evaluator(inputs, outputs, reference_outputs):
    from difflib import SequenceMatcher
    q = inputs["question"].lower()
    a = outputs["prediction"].lower()
    qw = set(q.split())
    aw = set(a.split())
    overlap = len(qw & aw) / len(qw) if qw else 0
    return {"key": "relevance", "score": overlap}

def similarity_evaluator(inputs, outputs, reference_outputs):
    from difflib import SequenceMatcher
    pred = outputs["prediction"]
    gt = reference_outputs["ground_truth"]
    sim = SequenceMatcher(None, pred, gt).ratio()
    return {"key": "similarity", "score": sim}

# 3. Evaluation loop
results = []

for q, gt in zip(questions, ground_truths):
    pred = chat(q)   # your chatbot function

    row = {
        "question": q,
        "prediction": pred,
        "ground_truth": gt,
    }

    # Compute metrics
    correctness = correctness_evaluator(
        inputs={"question": q},
        outputs={"prediction": pred},
        reference_outputs={"ground_truth": gt}
    ).get("score")

    relevance = relevance_evaluator(
        inputs={"question": q},
        outputs={"prediction": pred},
        reference_outputs={"ground_truth": gt}
    ).get("score")

    similarity = similarity_evaluator(
        inputs={"question": q},
        outputs={"prediction": pred},
        reference_outputs={"ground_truth": gt}
    ).get("score")

    row["correctness"] = correctness
    row["relevance"] = relevance
    row["similarity"] = similarity

    results.append(row)

# Print dataframe
df = pd.DataFrame(results)
print(df)

# 4. Upload to LangSmith and CLOSE all runs immediately
now = datetime.now(timezone.utc)

for row in results:
    run_id = str(uuid.uuid4())

    client.create_run(
        id=run_id,
        project_name=project_name,
        name="my_chatbot_eval",
        run_type="chain",
        start_time=now,
        end_time=now,
        status="success",
        inputs={"question": row["question"]},
        outputs={
            "prediction": row["prediction"],
            "ground_truth": row["ground_truth"],
        },
    )

    client.create_feedback(run_id=run_id, key="correctness", score=row["correctness"])
    client.create_feedback(run_id=run_id, key="relevance", score=row["relevance"])
    client.create_feedback(run_id=run_id, key="similarity", score=row["similarity"])

print("Completed evaluation and uploaded all runs to LangSmith.")


