import os, time, json, boto3
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI
from supabase import create_client, Client
from openai import OpenAI

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.60

# ========== load 1x → RAM ==========
services_r2_url = os.getenv("R2_BUCKET_URL")
R2_AK = os.getenv("R2_ACCESS_KEY_ID")
R2_SK = os.getenv("R2_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=R2_AK,
    aws_secret_access_key=R2_SK,
    endpoint_url=services_r2_url,
    region_name="auto"
)

print("⬇️ Loading services from R2 → cloud RAM...")
try:
    obj = s3.get_object(Bucket="servicescache", Key="services_cache_v7_clean.json")
    raw = obj["Body"].read().decode("utf-8")
    ALL = json.loads(raw)
    print("✅ Loaded:", len(ALL))
except:
    ALL = []

RAM_SERVICES = []
for s in ALL:
    vec = s.get("embedding_large") or s.get("embedding_clean")
    if vec is None:
        continue
    arr = np.array(vec, dtype=np.float32)
    if not arr.size:
        continue

    RAM_SERVICES.append({
        "id": s.get("id"),
        "name": s.get("name"),
        "category": s.get("category"),
        "keywords": s.get("keywords",[]),
        "embedding": arr,
        "uniqueid": s.get("uniqueid","")
    })

print(f"🚀 Cached {len(RAM_SERVICES)} services në Render RAM")

# ========== utility functions ==========
def cosine(a,b):
    na,nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a,b)/(na*nb))

def scale01(x):
    return max(0.0, min(1.0, (x+1.0)/2.0))

# ========== endpoint ==========
@app.post("/search")
async def search(body: dict):
    t0 = time.time()
    q  = body.get("q","")

    # 1) local style clean/refine (identike)
    cleaned = q.strip().lower()
    refined = cleaned

    # 2) embedding cache në disk ose R2 (persistente)
    cache_key = f"./.cache_{refined}.npy"

    if os.path.exists(cache_key):
        qemb = np.load(cache_key)
    else:
        r = client.embeddings.create(model="text-embedding-3-large", input=refined)
        qemb = np.array(r.data[0].embedding, dtype=np.float32)
        np.save(cache_key, qemb)

    scored=[]
    for s in RAM_SERVICES:
        sim=cosine(qemb, s["embedding"])
        s01=scale01(sim)
        if s01 < RED_TH:
            continue
        scored.append((s01, sim, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    top4 = scored[:4]

    final=[]
    for sc01, sc, s in top4:
        if sc01 < YELLOW_TH:
            continue
        final.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "score": round(sc01,3),
            "uniqueid": s.get("uniqueid",""),
            "keywords": s.get("keywords",[])
        })

    return {
        "results": final,
        "time_sec": round(time.time()-t0,2),
        "cleaned": cleaned,
        "refined": refined
    }
