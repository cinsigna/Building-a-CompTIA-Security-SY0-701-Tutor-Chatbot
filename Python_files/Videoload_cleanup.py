# %%
## Dependencies
"""
pip install youtube-transcript-api
pip install youtube-transcript-api --upgrade
pip install --upgrade youtube-transcript-api
pip install nltk
pip install pandas
"""

# %%
# Importing libraries 

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

# %% [markdown]
# # Calling API the youtube video transcripts

# %%
api = YouTubeTranscriptApi()

def get_english_transcript(video_id: str) -> str | None:
    try:
        data = api.fetch(video_id, ["en"])
        return " ".join(s.text for s in data)
    except NoTranscriptFound:
        print(f"Skipping {video_id} (no English transcript)")
        return None

# %%
video_ids = ["YBF9c2mCGME", "G0NCHag1rKc", "epgQ-sAr0l8", "3EgYr7jR4NI", "_AadMC3mzSk"]

records = []
for vid in video_ids:
    text = get_english_transcript(vid)
    if text is None:
        continue
    records.append({"video_id": vid, "transcript": text})

df = pd.DataFrame(records)

# %%

print(df.head())

# %%
## Preprocesing, chunking and creation of the dataframe for the youtube video transcripts

# %%
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\ufeff', '')
    text = re.sub(r"\s+", " ", text)
    text = text.replace("[Music]", "").replace("[music]", "").strip()
    # keep your current tokenization / cleaning here if you still want it
    return text

# 1) tiktoken length function (for token-based chunk size)
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text: str) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# 2) create the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,          # about 400 tokens
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

# 3) split each transcript into chunks
df["chunks"] = df["transcript"].apply(lambda text: text_splitter.split_text(text))

# 4) preprocess EACH chunk and keep them as a list
def preprocess_chunk_list(chunk_list):
    return [preprocess_text(chunk) for chunk in chunk_list]

df["Processed_chunks"] = df["chunks"].apply(preprocess_chunk_list)

# 5) explode into multiple rows (one per processed chunk)
df_exploded = df.explode("Processed_chunks")

# 6) add chunk number per video_id
df_exploded["chunk_number"] = df_exploded.groupby("video_id").cumcount() + 1

# 7) rename the exploded column for clarity
df_exploded = df_exploded.rename(columns={"Processed_chunks": "Processed_Text_chunk"})

df_exploded.head()

# %%
# 8) Defining final dataframe
clean_text = df_exploded[["video_id","chunk_number", "Processed_Text_chunk"]]
print(clean_text.head())

# %%
# 9) Obtaining CSV file
clean_text.to_csv("clean_text.csv", index=False)


