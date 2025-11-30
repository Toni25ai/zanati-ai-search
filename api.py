import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI

# ========== FastAPI ==========
app = FastAPI()

# ========== Keys ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# R2 keys (vetëm për fetch JSON 1x)
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

if not OPENAI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing critical API keys!")
    exit()

client = OpenAI(api_key=OPENAI_API_KEY)

# ========== Supabase ==========
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== JSON backup from R2 (LOAD 1 HERE!) ==========
R2_BUCKET_URL = os.getenv("R2_BUCKET_URL")  # do e shtojmë në Render si secret

print("⬇️ Loading JSON 1x from bucket to RAM...")
try:
    services_json = requests.get(R2_BUCKET_URL, timeout=20).json()
    SERVICES = services_json
    print(f"✅ Loaded {len(SERVICES)} services into RAM")
except Exception as e:
    print("❌ JSON failed to load:", str(e))
    SERVICES = []

# ========== Cosine similarity ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a,b)/(na*nb))

def scale01(x): return max(0.0, min(1.0, (x + 1.0) / 2.0))

# ========== Search (LOGJIKA IDENTIKE si në PC, pa DB scan) ==========
@app.get("/search")
def search_service(q: str):
    t0 = time.time()

    query_clean = re.sub(r"[^a-zA-Z0-9 ëç]+", "", q.lower()).strip()

    r = client.embeddings.create(model="text-embedding-3-large", input=query_clean)
    qemb = np.array(r.data[0].embedding, dtype=np.float32)

    scored = []
    for s in SERVICES:
        emb = s.get("embedding") or s.get("embedding_clean")
        emb_arr = np.array(emb, dtype=np.float32)
        sim_raw = cosine(qemb, emb_arr)
        sim01 = scale01(sim_raw)

        if sim01 >= 0.60:
            scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = [
        {
            "id": s["id"],
            "name": s["name"],
            "score": round(sim01,3),
            "uniqueid": s.get("uniqueid", "")
        }
        for sim01, sim_raw, s in scored[:4]
    ]

    return {"results": results, "time_sec": round(time.time()-t0, 2)}
