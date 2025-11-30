from fastapi import FastAPI
import requests, time, re
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
import os

app = FastAPI()

# 1) Ngarkim JSON 1x nga R2
JSON_URL = "https://3a4c4d0d75c22ad3e96653008476f710.r2.cloudflarestorage.com/servicescache/services_cache_v7_clean.json"

resp = requests.get(JSON_URL, timeout=25)
if resp.status_code != 200:
    print("❌ Nuk u shkarkua JSON, status:", resp.status_code)
    SERVICES = []
else:
    SERVICES = resp.json()

print("✅ U ngarkuan", len(SERVICES), "services në RAM")

# 2) Utils identical si lokale
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def normalize(t: str):
    return re.sub(r"[^a-zA-Z0-9 ëç]+", "", t.lower()).strip()

# 3) Endpoint health
@app.get("/health")
def health():
    return {"status": "ok"}

# 4) Endpoint search
@app.get("/search")
def search(q: str):
    t0 = time.time()
    qc = normalize(q)

    # embed query
    try:
        e = client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).embeddings.create(model="text-embedding-3-large", input=qc)
        qv = np.array(e.data[0].embedding, dtype=np.float32)
    except Exception as err:
        print("Embedding error", err)
        return {"results": [], "time_sec": round(time.time()-t0,2)}

    # similarity në RAM identical si lokale
    scored = []
    for s in SERVICES:
        svec = np.array(s["embedding_clean"], dtype=np.float32)
        sim_raw = cosine(qv, svec)
        sim01 = scale01(sim_raw)
        if sim01 < 0.5:
            continue
        scored.append((sim01, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, s in scored[:4]:
        results.append({
            "id": s["id"],
            "name": s["name"],
            "score": round(sim,3),
            "uniqueid": s.get("uniqueid",""),
            "category": s.get("category","")
        })

    return {"results": results, "time_sec": round(time.time()-t0,2)}
