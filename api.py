import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from fastapi import FastAPI

app = FastAPI()

# ===== OpenAI =====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== Load JSON 1x prej R2 =====
JSON_URL = "https://3a4c4d0d75c22ad3e96653008476f710.r2.cloudflarestorage.com/servicescache/services_cache_v7_clean.json"

SERVICES = []
LOADED = False

def cosine(a,b):
    na, nb = norm(a), norm(b)
    return 0.0 if na==0 or nb==0 else float(np.dot(a,b)/(na*nb))

def scale01(x): return max(0.0, min(1.0, (x+1.0)/2.0))
def normalize(t): return re.sub(r"[^a-zA-Z0-9 ëç]", "", t.lower()).replace("ë","e").replace("ç","c").strip()

def load_services_once():
    global SERVICES, LOADED
    if LOADED: return
    print("⬇️ loading JSON në RAM…")
    data = requests.get(JSON_URL, timeout=20).json()
    for s in data:
        if not s.get("embedding_clean"): continue
        SERVICES.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "keywords": s.get("keywords",[]),
            "embedding": np.array(s["embedding_clean"],dtype=np.float32),
            "uniqueid": s.get("uniqueid","")
        })
    LOADED = True
    print(f"✅ loaded {len(SERVICES)} services")

load_services_once()

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/search")
def search(q: str):
    t0 = time.time()
    q = normalize(q)
    # embed query
    e = client.embeddings.create(model="text-embedding-3-large", input=q)
    qv = np.array(e.data[0].embedding,dtype=np.float32)
    scored=[]
    for s in SERVICES:
        sim = scale01(cosine(qv, s["embedding"]))
        if sim<0.60: continue
        scored.append((sim,s))
    scored.sort(key=lambda x:x[0],reverse=True)
    out=[{"id":s["id"],"name":s["name"],"category":s["category"],"uniqueid":s["uniqueid"],"score":sim} for sim,(_,s) in zip([x[0] for x in scored[:4]],[s for _,s in scored[:4]])]
    return {"results":out,"time_sec": round(time.time()-t0,2)}
