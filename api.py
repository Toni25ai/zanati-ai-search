import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm

# ========== IMPORTS FIX ==========
from supabase_py import create_client, Client  # nga supabase-py alternative
from openai import OpenAI  # library e saktë përmes pip install openai
from fastapi import FastAPI

# ========== FASTAPI INIT ==========
app = FastAPI()

# ========== SUPABASE CONNECT ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OPENAI CONNECT ==========
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# ========== CACHE ========== 
SERVICES = []
LOADED = False

# ========== PARAMETRAT ==========
GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.50

# ========== FUNKSIONET ==========
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
    prompt = f'A është shërbimi "{service_name}" i përshtatshëm për "{query}"? Kthe vetëm: po / jo.'
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3
        )
        return r.choices[0].message.content.strip().lower().startswith("po")
    except:
        return False

def load_services_once():
    global SERVICES, LOADED
    if LOADED:
        return

    print("⬇️ Po shkarkoj JSON 1x…")
    R2_ACCOUNT = "3a4c4d0d75c22ad3e96653008476f710"
    url = f"https://{R2_ACCOUNT}.r2.cloudflarestorage.com/servicescache/services_cache_v7_clean.json"

    try:
        data = requests.get(url, timeout=20).json()
    except Exception as e:
        print("❌ JSON failed:", e)
        SERVICES = []
        LOADED = True
        return

    SERVICES.clear()
    for s in data:
        vec = s.get("embedding_clean")
        if not vec:
            continue
        uid = s.get("uniqueid","")
        SERVICES.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "category": s.get("category",""),
            "keywords": s.get("keywords", []),
            "uniqueid": uid,
            "embedding": np.array(vec, dtype=np.float32)
        })

    LOADED = True
    print(f"✅ Loaded {len(SERVICES)} into RAM")

# Thirre 1x në start
load_services_once()

# ========== ENDPOINTS ==========
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/columns")
def columns():
    try:
        s = supabase.table("detailedtable").select("*").limit(1).execute().data
    except:
        return []
    if not s:
        return []
    return list(s[0].keys())

@app.get("/search")
def search(q: str):
    t0 = time.time()
    q = normalize(q)

    # krijo embedding
    try:
        rsp = client.embeddings.create(model="text-embedding-3-large", input=q)
        qv = np.array(rsp.data[0].embedding, dtype=np.float32)
    except:
        return {"results": [], "time_sec": round(time.time()-t0,2)}

    # similarity identical si në PC
    scored = []
    for s in SERVICES:
        sim_raw = cosine(qv, s["embedding"])
        sim01 = scale01(sim_raw)
        if sim01 < RED_TH:
            continue
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    for sim01, sim_raw, s in scored[:4]:
        final.append({
            "id": s["id"],
            "name": s["name"],
            "category": s.get("category",""),
            "keywords": s.get("keywords",[])[:5],
            "uniqueid": s["uniqueid"],
            "score": round(sim01, 3)
        })

    # GPT check vetëm për yellow identical si logjika jote
    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    if greens and 1 <= len(final) < 3 and yellows:
        cand = yellows[0][2]
        if gpt_check(cand["name"], q):
            final.append({
                "id": cand["id"],
                "name": cand["name"],
                "category": cand.get("category",""),
                "keywords": cand.get("keywords",[])[:5],
                "uniqueid": cand["uniqueid"],
                "score": round(yellows[0][0],3)
            })

    return {"results": final, "time_sec": round(time.time()-t0,2)}
