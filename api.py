import os, time, re, json
import numpy as np
from numpy.linalg import norm
from supabase import create_client, Client
from openai import OpenAI
from fastapi import FastAPI

app = FastAPI()

# ===== 1) Lidhjet (saktë si në projekt) =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== 2) Load JSON 1x në memory (RAM) =====
# JSON file qëndron në root folder të projektit në Render
# (Nuk përdorim R2 keys për search speed)
SERVICES_JSON = "services_cache_v7_clean.json"

if not os.path.exists(SERVICES_JSON):
    raise Exception(f"❌ File {SERVICES_JSON} nuk ekziston në server! Duhet ta upload-osh te Render.")

with open(SERVICES_JSON, "r", encoding="utf-8") as f:
    RAW_SERVICES = json.load(f)

SERVICES = []
for s in RAW_SERVICES:
    vec = s.get("embedding_clean")
    if vec is None:
        continue
    SERVICES.append({
        "id": s["id"],
        "name": s["name"],
        "category": s.get("category", ""),
        "keywords": s.get("keywords", [])[:5],
        "uniqueid": s.get("uniqueid",""),
        "embedding": np.array(vec, dtype=np.float32)
    })

print(f"✅ U ngarkuan {len(SERVICES)} services në RAM.")

# ===== 3) Funksionet utility identike si lokale =====
def cosine(a, b):
    na, nb = norm(a), norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def normalize(t: str):
    return re.sub(r"[^a-zA-Z0-9 ëç]+", "", t.lower()).strip()

# ===== 4) Endpoint Health =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 5) Endpoint Search (super i shpejt, identical RAM based) =====
@app.get("/search")
def search_service(q: str):
    t0 = time.time()
    query_text = normalize(q)

    # Krijo embedding për query
    try:
        rsp = client.embeddings.create(model="text-embedding-3-large", input=query_text)
        qvec = np.array(rsp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print("❌ Embedding failed:", e)
        return {"results": [], "time_sec": round(time.time()-t0,2)}

    # Llogarit similarity me services në RAM (identical)
    scored = []
    for s in SERVICES:
        sim_raw = cosine(qvec, s["embedding"])
        sim01 = scale01(sim_raw)
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Kthe 4 rezultatet e para identical
    results = []
    for sim01, _, s in scored[:4]:
        results.append({
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "uniqueid": s["uniqueid"],
            "score": round(sim01, 3)
        })

    return {"results": results, "time_sec": round(time.time() - t0, 2)}
