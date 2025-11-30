import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from fastapi import FastAPI

app = FastAPI()

# 1) LIDHJE me OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2) URL e file që ke ruajtur në R2 bucket (keq-citiruar më herët)
SERVICES_JSON_URL = "https://3a4c4d0d75c22ad3e96653008476f710.r2.cloudflarestorage.com/servicescache/services_cache_v7_clean.json" 

# 3) Load file 1x në memory (RAM)
SERVICES = requests.get(SERVICES_JSON_URL, timeout=20).json()

print(f"✅ Loaded {len(SERVICES)} services nga R2 në RAM.")

def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search(q: str):
    t0 = time.time()
    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", q.lower()).strip()

    # embed query
    try:
        e = client.embeddings.create(model="text-embedding-3-large", input=cleaned)
        qv = np.array(e.data[0].embedding, dtype=np.float32)
    except:
        return {"results": [], "time_sec": round(time.time()-t0,2)}

    scored = []
    for s in SERVICES:
        emb = np.array(s["embedding_clean"], dtype=np.float32)
        sim_raw = cosine(qv, emb)
        sim01 = scale01(sim_raw)
        scored.append((sim01, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim,s in scored[:4]:
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "uniqueid": s.get("uniqueid",""),
            "score": round(sim,3)
        })

    return {"results": results, "time_sec": round(time.time()-t0,2)}
