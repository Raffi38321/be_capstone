from fastapi import FastAPI, UploadFile, File,Body,Form
import pymupdf
import pandas as pd
import joblib
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.preprocessing import *
from utils.skill_extraction import *
from utils.inference import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

app = FastAPI()


model = load_model("model/encoder.h5")
tokenizer = joblib.load("model/tokenizer.pkl")
class CosineSimilarityLayer(Layer):
    def call(self, inputs):
        u, v = inputs
        dot = K.sum(u * v, axis=1, keepdims=True)
        norm_u = K.sqrt(K.sum(K.square(u), axis=1, keepdims=True))
        norm_v = K.sqrt(K.sum(K.square(v), axis=1, keepdims=True))
        return dot / (norm_u * norm_v + 1e-8)

model_predict = load_model(
    'model/job_matching_model.keras',
    custom_objects={
        "CosineSimilarityLayer": CosineSimilarityLayer
    }
)
df = pd.read_csv("data/jobs.csv")
job_clean = df['cleaned_text'].tolist()

max_len = model.input_shape[1]
job_seq = tokenizer.texts_to_sequences(job_clean)
job_pad = pad_sequences(job_seq, maxlen=max_len)
job_vectors = model.predict(job_pad)


# API ENDPOINT
@app.post("/topN")
async def topN(file: UploadFile = File(...)):
    
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


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    iloc: int = Form(...)
):
    if iloc < 0 or iloc >= len(df):
        return {"error": "Index out of range"}

    doc = pymupdf.open(stream=await file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")

    clean = bersihkan_teks(text)
    skill_section = ambil_bagian_skill(clean)
    skill_section = normalize_synonym(skill_section)

    skills = extract_skills(skill_section)
    user_clean = " ".join(skills)
    job = df.iloc[iloc]
    job_text = job['cleaned_text']

    user_seq = tokenizer.texts_to_sequences([user_clean])
    job_seq = tokenizer.texts_to_sequences([job_text])

    user_pad = pad_sequences(user_seq, maxlen=max_len)
    job_pad = pad_sequences(job_seq, maxlen=max_len)

    score = model_predict.predict([user_pad, job_pad])[0][0]

    return {
        "skills user": skills,
        "skor": float(score),
        "nama job": job["Judul"],
        "perusahaan":job["Perusahaan"],
        "skill yang dibutuhin": job["Skills"]
    }