import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

def recommend_jobs(text, tokenizer, model, job_clean, job_vectors, max_len, df):
    
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    user_vec = model.predict(pad)

    scores = cosine_similarity(user_vec, job_vectors)[0]

    top_idx = np.argsort(scores)[::-1][:5]

    results = []
    for i in top_idx:
        results.append({
            "judul": df.iloc[i]['Judul'],
            "perusahaan": df.iloc[i]['Perusahaan'],
            "score": float(scores[i])
        })

    return results