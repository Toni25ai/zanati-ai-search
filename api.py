import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# parametra të shpejtë
GREEN_TH  = 0.70
YELLOW_TH = 0.60

def cosine(a,b):
    na, nb = norm(a), norm(b)
    return 0.0 if na==0 or nb==0 else float(np.dot(a,b)/(na*nb))

def scale01(x): return max(0.0, min(1.0,(x+1)/2))

def normalize(t): return re.sub(r"[^a-zA-Z0-9 ëç]","",t.lower()).replace("ë","e").replace("ç","c").strip()

# load 1x on start
SERVICES = []
LOADED = False

JSON_URL = "https://" + os.getenv("SERVICES_JSON_URL")

def load_services_once():
    global SERVICES, LOADED
    if LOADED: return
    print("⬇️ loading JSON from R2 into RAM…")
    try:
        data = requests.get(JSON_URL, timeout=20).json()
        for s in data:
            if not s.get("embedding_clean"): continue
            SERVICES.append({
                "id": s["id"],
                "name": s["name"],
                "category": s.get("category",""),
                "keywords": s.get("keywords",[]),
                "uniqueid": s.get("uniqueid",""),
                "embedding": np.array(s["embedding_clean"],dtype=np.float32)
            })
        LOADED = True
        print(f"✅ loaded {len(SERVICES)} services in RAM")
    except Exception as e:
        print("❌ JSON load error:",e)
        SERVICES = []
        LOADED = True

load_services_once()

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/search")
def search(q: str):
    t0 = time.time()
    q = normalize(q)
    try:
        e = client.embeddings.create(model="text-embedding-3-large",input=q)
        qv = np.array(e.data[0].embedding,dtype=np.float32)
    except:
        return {"results":[],"time_sec": round(time.time()-t0,2)}

    scored=[]
    for s in SERVICES:
        sim = scale01(cosine(qv, s["embedding"]))
        if sim < 0.60: continue
        scored.append((sim,s))
    scored.sort(key=lambda x:x[0],reverse=True)

    out=[{"id":s["id"],"name":s["name"],"category":s["category"],"uniqueid":s["uniqueid"],"score":sim} for sim,s in scored[:4]]
    return {"results":out,"time_sec": round(time.time()-t0,2)}
