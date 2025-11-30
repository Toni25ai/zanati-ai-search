import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI

# ========== FASTAPI SETUP ==========
app = FastAPI()

# ========== KEYS ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ========== DATASET NGA R2 → LOAD 1× ==========
R2_PUBLIC_FILE = "https://3a4c4d0d75c22ad3e96653008476f710.r2.cloudflarestorage.com/servicescache/services_cache_v7_clean.json"

print("⬇️ Po shkarkoj dataset 1x…")
try:
    response = requests.get(R2_PUBLIC_FILE, timeout=25)
    if response.status_code != 200:
        print("❌ Nuk u lexua dataset nga R2. Status code:", response.status_code)
        SERVICES_RAW = []
    else:
        SERVICES_RAW = response.json()
except Exception as e:
    print("❌ Dataset download error:", e)
    SERVICES_RAW = []

# Fut në RAM vetëm entries valide
SERVICES = []
for s in SERVICES_RAW:
    emb = s.get("embedding_large") or s.get("embedding_clean")  # merre çfarë të kesh
    if not isinstance(emb, list):
        continue
    SERVICES.append({
        "id": s["id"],
        "name": s["name"],
        "uniqueid": s.get("uniqueid", ""),
        "category": s.get("category", ""),
        "keywords": s.get("keywords", []),
        "embedding": np.array(emb, dtype=np.float32)
    })

print(f"✅ U futën {len(SERVICES)} services në RAM.")

# ========== UTILS (identike si lokali) ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def normalize(t: str):
    return re.sub(r"[^a-zA-Z0-9 ëç]+", "", t.lower()).strip()

# ========== ENDPOINTS ==========
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search_service(q: str):
    t0 = time.time()
    query_clean = normalize(q)

    # embedding për query në cloud
    try:
        e = client.embeddings.create(model="text-embedding-3-large", input=query_clean)
        qvec = np.array(e.data[0].embedding, dtype=np.float32)
    except Exception as er:
        print("❌ Embedding failed:", er)
        return {"results": [], "time_sec": round(time.time()-t0, 2)}

    scored = []
    for s in SERVICES:
        sim_raw = cosine(qvec, s["embedding"])
        sim01 = scale01(sim_raw)
        if sim01 < 0.5:  # këtu është fallback minimal që garanton rezultate si lokale
            continue
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    for sim01,_,s in scored[:4]:
        final.append({
            "id": s["id"],
            "name": s["name"],
            "uniqueid": s["uniqueid"],
            "category": s["category"],
            "score_large": round(sim01, 3)
        })

    return {"results": final, "time_sec": round(time.time()-t0, 2)}
