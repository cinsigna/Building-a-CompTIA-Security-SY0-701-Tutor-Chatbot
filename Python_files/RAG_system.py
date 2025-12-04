# %%
## Dependencies
"""
pip install pinecone
pip install torch torchvision torchaudio
pip install sentence-transformers
pip install pypdf
pip install openai
pip install --upgrade --quiet langchain langchain-community langchain_openai
"""

# %%
# Importing libraries 

import pandas as pd
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time
from openai import OpenAI
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pypdf import PdfReader
from dotenv import load_dotenv
from datetime import datetime
import os
from pypdf import PdfReader

# %%
# Loading the dotevn

load_dotenv()

# %%
#Loading the CSV file containing the video transcripts. 
clean_text = pd.read_csv("clean_text.csv")
print(clean_text.head())


# %% [markdown]
# ## Creating and initializing the vector database index using Pinecone

# %%
# Retrieve the API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY =os.getenv('PINECONE_API_KEY')

# Set as environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

print("OpenAI API key loaded and set as environment variable.")
print("Pinecone API key loaded and set as environment variable.")

# %%
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'

# configure client
pc = Pinecone(api_key=api_key)

# %%
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

# %%
#pc.delete_index("indexfp")

# %%
index_name = "indexfp"

# %%
# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,    
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
    # wait until the index is ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# connect to the index and check stats are all zeros
index = pc.Index(index_name)
print(index.describe_index_stats()) #initialize the index, and insure the stats are all zeros

# %%

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device=device
)

retriever

# %%

# Wait for the index to be ready
print(f"Waiting for index '{index_name}' to be ready...")
while True:
    try:
        # describe_index returns an IndexDescription object which has a status field
        index_description = pc.describe_index(index_name)
        if index_description.status.ready:
            print(f"Index '{index_name}' is ready.")
            break
        else:
            # Index is not ready yet, print current state and wait
            print(f"Index '{index_name}' is not ready yet. Status: {index_description.status.state}. Waiting...")
            time.sleep(10) # Wait for 10 seconds before checking again
    except Exception as e:
        # Catch potential errors if index is still being created/initialized by Pinecone
        print(f"Error checking index status: {e}. Retrying in 10 seconds...")
        time.sleep(10)

# Ensure the stats are all zeros (for a freshly created/empty index)
# Use index.describe_index_stats() to get the current statistics
stats = index.describe_index_stats()

# Check if the total_vector_count is 0
if stats.total_vector_count == 0:
    print(f"Index '{index_name}' is empty (total_vector_count: {stats.total_vector_count}).")
    print(f"Dimension: {stats.dimension}")
else:
    print(f"WARNING: Index '{index_name}' is not empty. Current stats:")
    print(stats) # Print full stats if not empty

# %% [markdown]
# ## Initializing YT Embedding chunks of the YT video transcripts

# %%
df = clean_text.copy()

# unique, stable id per chunk
df["id"] = df.apply(
    lambda r: f"{r['video_id']}_{int(r['chunk_number'])}", axis=1
)

# text to embed
df["passage_text"] = df["Processed_Text_chunk"]

# optional metadata for your template
df["article_title"] = df["video_id"]
df["section_title"] = df["chunk_number"].astype(str)

# %%
batch_size = 64

print(f"\nStarting embedding generation and upsert for {len(df)} documents into Pinecone index '{index_name}'...")

for i in tqdm(range(0, len(df), batch_size), desc="Upserting batches to Pinecone"):
    i_end = min(i + batch_size, len(df))
    batch_df = df.iloc[i:i_end]

    batch_texts = batch_df["passage_text"].tolist()

    # document embeddings using the retriever model
    batch_embeddings = retriever.encode(
        batch_texts,
        show_progress_bar=False,
        device=device
    ).tolist()

    vectors_for_batch = []

    for j, row in enumerate(batch_df.itertuples(index=False)):
        doc_id = str(row.id)

        metadata = {
            "source": "transcript",
            "video_id": row.video_id,
            "passage_text": row.passage_text,
            # "snippet": row.passage_text[:500],
        }

        vectors_for_batch.append({
            "id": doc_id,
            "values": batch_embeddings[j],
            "metadata": metadata,
        })

    try:
        index.upsert(vectors=vectors_for_batch)
    except Exception as e:
        print(f"\nError upserting batch {i}-{i_end}: {e}")
        raise

print("\nFinished generating embeddings and upserting all batches to Pinecone.")

final_index_stats = index.describe_index_stats()
if "" in final_index_stats.namespaces:
    print(f"Total vectors in index: {final_index_stats.namespaces[''].vector_count}")
else:
    print("Total vectors in index: 0 (No default namespace found).")
print(final_index_stats)

# %% [markdown]
# ## Loading PDF and Initializing their embedding chunks in the Vector database.

# %%
# %%  Load PDFs into a pandas DataFrame

def load_pdfs_to_dataframe(pdf_folder="files"):
    """
    Reads all PDF files in a folder and loads them into a pandas DataFrame.
    Columns: pdf_file, page_number, text
    """
    rows = []

    if not os.path.isdir(pdf_folder):
        print(f"Folder '{pdf_folder}' does not exist.")
        return pd.DataFrame(columns=["pdf_file", "page_number", "text"])

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)

        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            continue

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            rows.append({
                "pdf_file": pdf_file,
                "page_number": i,
                "text": text
            })

    return pd.DataFrame(rows)


# %%  Chunking helpers for PDF text

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Word based chunking with overlap.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start += max(chunk_size - overlap, 1)

    return chunks


def chunk_pdf_dataframe(pdf_df, chunk_size=400, overlap=50):
    """
    Takes the page based pdf_df and returns a new DataFrame
    with one row per chunk.

    Columns:
      pdf_file
      page_number
      chunk_index
      passage_text
    """
    chunk_rows = []

    for _, row in pdf_df.iterrows():
        text = row["text"] or ""
        if not text.strip():
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(chunks):
            chunk_rows.append({
                "pdf_file": row["pdf_file"],
                "page_number": row["page_number"],
                "chunk_index": idx,
                "passage_text": chunk,
            })

    if not chunk_rows:
        return pd.DataFrame(columns=["pdf_file", "page_number", "chunk_index", "passage_text"])

    return pd.DataFrame(chunk_rows)

# %%
# %%  Upsert PDF chunks into the same Pinecone index

# Load and chunk PDFs
pdf_df = load_pdfs_to_dataframe("files")  # change folder name if needed
pdf_chunks_df = chunk_pdf_dataframe(pdf_df, chunk_size=400, overlap=80)

print(f"\nFound {len(pdf_chunks_df)} PDF chunks to upsert into Pinecone index '{index_name}'...")

batch_size_pdf = 64

for i in tqdm(range(0, len(pdf_chunks_df), batch_size_pdf), desc="Upserting PDF batches to Pinecone"):
    i_end = min(i + batch_size_pdf, len(pdf_chunks_df))
    batch_df = pdf_chunks_df.iloc[i:i_end]

    batch_texts = batch_df["passage_text"].tolist()

    # same retriever, same embedding space
    batch_embeddings = retriever.encode(
        batch_texts,
        show_progress_bar=False,
        device=device
    ).tolist()

    vectors_for_batch = []

    for j, row in enumerate(batch_df.itertuples(index=False)):
        # unique ID so it does not collide with transcript IDs
        doc_id = f"pdf_{row.pdf_file}_{row.page_number}_{row.chunk_index}"

        metadata = {
            "source": "pdf",
            "pdf_file": row.pdf_file,
            "page_number": int(row.page_number),
            "chunk_index": int(row.chunk_index),
            "passage_text": row.passage_text,
        }

        vectors_for_batch.append({
            "id": doc_id,
            "values": batch_embeddings[j],
            "metadata": metadata,
        })

    try:
        index.upsert(vectors=vectors_for_batch)
    except Exception as e:
        print(f"\nError upserting PDF batch {i}-{i_end}: {e}")
        raise

print("\nFinished generating embeddings and upserting all PDF chunks to Pinecone.")

# Show updated stats including PDFs
final_index_stats = index.describe_index_stats()
if "" in final_index_stats.namespaces:
    print(f"Total vectors in index (transcripts + PDFs): {final_index_stats.namespaces[''].vector_count}")
else:
    print("Total vectors in index: 0 (No default namespace found).")
print(final_index_stats)

# %% [markdown]
# ## Creating a tool to generate the user request and response

# %%
client = OpenAI()

LAST_MOCK_TEST = None  # holds the last mock exam text
EXAM_STATE = {
    "in_exam": False,
    "start_time": None,
}

def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a expert tutor for the CompTIA Security+ SY-701 certification course. Coaching students for their certification preparation.
Use the following rules:

1) Start by using the context below whenever it is relevant. Quote or paraphrase it where needed.
2) If the context does not fully answer the question (for example, the user asks for a 60-day plan, extra practice tests, or “top 10 acronyms”), then:
   - Use your own knowledge to extend or create a helpful answer,
   - Keep everything aligned with Security+ SY0-701 exam preparation.
3) If the question is clearly unrelated to Security+, say: "I don't know based on this course."

Context:
{context}

Question:
{question}

Answer:"""
    response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[
           {
               "role": "system",
               "content": "You are a Security+ SY0-701 tutor. Prefer using course transcripts when they contain relevant information. When transcripts do not contain everything needed, extend the answer using your own knowledge while keeping it aligned with Security+ SY0-701."
           },
           {"role": "user", "content": prompt},
       ],
       temperature=0,
       max_tokens=300,
   )

    return response.choices[0].message.content

# %%
# Defining Pinecone query funtion 
def query_pinecone(query, top_k=15):
    # embed the question
    q_emb = retriever.encode([query], convert_to_tensor=False)[0].tolist()

    # query the Pinecone index
    result = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )

    return result

# %%
# Defining Pinecone matching funtion 
def extract_chunks(query_result):
    chunks = []
    for match in query_result["matches"]:
        chunks.append(match["metadata"]["passage_text"])
    return chunks

# %% [markdown]
# ## Defining the funtion to search and deliver the results to the agent

# %%
def ask(question, top_k=15):
    """
    Main brain of the chatbot.

    Capabilities:
      1) General Security+ questions -> RAG over Pinecone + LLM.
      2) Generate a mock exam / full exam simulation when user asks for it.
      3) After a mock exam is generated, answer requests like:
         - "What is the answer and explanation to question 20?"
         - "Give me all the correct answers."
         - "Here are my answers, please grade me."
      4) When grading, mention how many minutes the learner used since exam start.
    """
    global LAST_MOCK_TEST, EXAM_STATE

    lower_q = question.lower()

    # 1) User asks for a mock test or full exam simulation
    exam_request_phrases = [
        "mock test",
        "mock exam",
        "practice exam",
        "full exam",
        "exam simulation",
        "real exam",
        "real time exam",
        "real-time exam",  # in case the user writes it like this
    ]

    if any(phrase in lower_q for phrase in exam_request_phrases):
        prompt = f"""You are a Security+ SY0-701 tutor.

Create a realistic mock exam similar in style and difficulty to the CompTIA Security+ SY0-701 certification exam.

Requirements:
- Use multiple choice questions only.
- Aim for around 20 to 30 questions that could reasonably take about 90 minutes.
- Cover a balanced mix of domains (threats, architecture, implementation, operations, governance, cryptography).
- Output ONLY the questions and the answer options (A, B, C, D).
- DO NOT include the correct answers or explanations in this response.
- At the end, invite the learner to answer the questions and then ask for the answer key or grading.

Learner request:
{question}

Now output the mock exam questions:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        mock_text = response.choices[0].message.content.strip()

        # Ready-to-copy answer template for the user
        answer_template = (
            "My answers are 1: , 2: , 3: , 4: , 5: , 6: , 7: , 8: , 9: , 10: , "
            "11: , 12: , 13: , 14: , 15: , 16: , 17: , 18: , 19: , 20: . "
            "Please grade my answers and tell me the score."
        )

        mock_text_with_tip = (
            f"{mock_text}\n\n"
            "When you are ready, you can send your answers in a single message. "
            "For example, you can copy this template format for your answers:\n"
            f"{answer_template}"
        )

        # store only the pure mock exam for grading
        LAST_MOCK_TEST = mock_text
        EXAM_STATE["in_exam"] = True
        EXAM_STATE["start_time"] = datetime.utcnow()

        return mock_text_with_tip
    
    # 2) User asks for answers or grading for the last mock exam
    wants_answers_or_grading = any(
        word in lower_q
        for word in [
            "answer",
            "answers",
            "solution",
            "solutions",
            "correct option",
            "correct options",
            "answer key",
            "grade",
            "score",
            "check my answers",
            "mark my answers",
        ]
    )

    if LAST_MOCK_TEST is not None and wants_answers_or_grading:
        # compute how many minutes have passed since exam start (if in exam mode)
        elapsed_minutes_text = ""
        if EXAM_STATE["in_exam"] and EXAM_STATE["start_time"] is not None:
            elapsed = datetime.utcnow() - EXAM_STATE["start_time"]
            used_minutes = int(elapsed.total_seconds() // 60)
            elapsed_minutes_text = f"The learner took about {used_minutes} minutes between exam start and this request."

        prompt = f"""You are a Security+ SY0-701 tutor.

Below is a mock exam that was previously given to the learner:

--- MOCK EXAM START ---
{LAST_MOCK_TEST}
--- MOCK EXAM END ---

{elapsed_minutes_text}

The learner now says:
"{question}"

Follow these rules:

1) If the learner asks for ALL the answers or for an "answer key":
   - Provide a numbered list of correct answers for every question.
   - For each question, show:
     - The correct option (A, B, C, or D).
     - A short explanation.

2) If the learner refers to a specific question number, such as "question 20":
   - Provide the correct option and a short explanation ONLY for those question numbers.

3) If the learner provides their own answers (for example "1:B, 2:C, 3:A..."):
   - Compare their answers to the correct ones.
   - Show which questions are correct and which are incorrect.
   - For incorrect ones, show the correct answer.
   - For incorrect ones, also provide a brief explanation.
   - Provide an overall score at the end (for example "You scored 16 out of 20").

Be very explicit about which question number you are referring to in each line."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )

        # after grading, end the exam session but keep LAST_MOCK_TEST
        EXAM_STATE["in_exam"] = False

        return response.choices[0].message.content.strip()

    # 3) Normal RAG behaviour for all other questions (general chatbot mode)

    # 1. retrieve from Pinecone
    results = query_pinecone(question, top_k=top_k)
    chunks = extract_chunks(results)

    # 2. build context for the LLM
    context = "\n\n".join(chunks)

    prompt = f"""You are a Security+ SY0-701 tutor.

Use the context below when it is helpful.
If the context contains partial information, extend the answer using your own Security+ SY0-701 knowledge.
If the question is clearly unrelated to Security+ study or exam preparation, say: "I don't know based on this course."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1400,
    )

    return response.choices[0].message.content


