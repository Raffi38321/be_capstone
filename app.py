from fastapi import FastAPI, UploadFile, File
import pymupdf
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from utils.preprocessing import *
from utils.skill_extraction import *
from utils.inference import *

app = FastAPI()

model = load_model("model/encoder.h5")
tokenizer = joblib.load("model/tokenizer.pkl")

df = pd.read_csv("data/jobs.csv")
job_clean = df['cleaned_text'].tolist()

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = model.input_shape[1]
job_seq = tokenizer.texts_to_sequences(job_clean)
job_pad = pad_sequences(job_seq, maxlen=max_len)
job_vectors = model.predict(job_pad)

# API ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # baca PDF
    doc = pymupdf.open(stream=await file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")

    # preprocessing
    clean = bersihkan_teks(text)
    skill_section = ambil_bagian_skill(clean)
    skill_section = normalize_synonym(skill_section)

    skills = extract_skills(skill_section)
    user_text = " ".join(skills)

    # inference
    results = recommend_jobs(
        user_text,
        tokenizer,
        model,
        job_clean,
        job_vectors,
        max_len,
        df
    )

    return {
        "skills": skills,
        "recommendations": results
    }