import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from supabase import create_client, Client
from openai import OpenAI
from fastapi import FastAPI

app = FastAPI()

# ===== 1) Lidhjet =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== 2) Parametrat (identik si lokal, s’i ndryshojmë) =====
GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.50

SERVICES = []
LOADED = False

# ===== 3) Funksionet =====
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def normalize(t: str):
    return re.sub(r"[^a-zA-Z0-9 ëç]+", "", t.lower()).strip()

def gpt_check(service_name, query):
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për "{query}"? Kthe vetëm: po/jo'
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=3
        )
        ans = rsp.choices[0].message.content.strip().lower()
        return ans.startswith("p")
    except:
        return False

# ===== 4) Ngarko JSON 1× nga R2 në start dhe mbaje në RAM =====
R2_ACCOUNT = "3a4c4d0d75c22ad3e96653008476f710"
JSON_URL = f"https://{R2_ACCOUNT}.r2.cloudflarestorage.com/servicescache/services_cache_v7_clean.json"

def load_services_once():
    global SERVICES, LOADED
    if LOADED:
        return
    print("⬇️ Po shkarkoj JSON 1x në memory…")
    try:
        resp = requests.get(JSON_URL, timeout=20)
        data = resp.json()
    except Exception as e:
        print("❌ JSON load dështoi:", e)
        SERVICES = []
        LOADED = True
        return

    SERVICES.clear()
    for s in data:
        vec = s.get("embedding_clean")
        if vec is None: continue
        uid = s.get("uniqueid","")
        SERVICES.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "keywords": s.get("keywords",[])[:5],
            "uniqueid": uid,
            "embedding": np.array(vec, dtype=np.float32)
        })
    LOADED = True
    print(f"✅ Loaded {len(SERVICES)} services in RAM.")

# thirre direkt në start
load_services_once()

# ===== 5) Endpoints =====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search")
def search_service(q: str):
    t0 = time.time()
    query_text = normalize(q)

    # embed query
    try:
        e = client.embeddings.create(model="text-embedding-3-large", input=query_text)
        qv = np.array(e.data[0].embedding, dtype=np.float32)
    except:
        return {"results": [], "time_sec": round(time.time()-t0,2)}

    scored = []
    for s in SERVICES:
        sim_raw = cosine(qv, s["embedding"])
        sim01 = scale01(sim_raw)
        if sim01 < RED_TH:
            continue
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    for sim01, _, s in scored[:4]:
        final.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "uniqueid": s["uniqueid"],
            "score": round(sim01, 3)
        })

    return {"results": final, "time_sec": round(time.time()-t0,2)}
