import os, time, re, json
import numpy as np
from numpy.linalg import norm
from supabase import create_client, Client
from openai import OpenAI
from fastapi import FastAPI

app = FastAPI()

# ========== LIDHJET CLOUD ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

SERVICES = []  # do i mbajmë në RAM
LOADED = False

# ========== FUNKSIONE IDENTIKE SI LOKALE ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0: 
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.50

# ========== NGARKE JSON 1× NGA R2 ========== 
def load_services_once():
    global SERVICES, LOADED
    if LOADED:
        return

    print("⬇️ Downloading services JSON 1x from R2...")
    try:
        url = "https://" + os.getenv("SERVICES_JSON_URL")
        resp = requests.get(url, timeout=20)
        data = resp.json()
    except Exception as e:
        print("❌ JSON download failed:", e)
        SERVICES = []
        LOADED = True
        return

    # jokohe pse JSON vjen si list, e ruajmë identik
    SERVICES.clear()
    for s in data:
        vec = s.get("embedding_clean")
        uid = s.get("uniqueid")
        if not vec or not uid:
            continue
        
        SERVICES.append({
            "id": s.get("id"),
            "name": s.get("name"),
            "category": s.get("category", ""),
            "keywords": s.get("keywords", []),
            "uniqueid": uid,
            "embedding": np.array(vec, dtype=np.float32)
        })

    LOADED = True
    print(f"✅ Loaded {len(SERVICES)} services into RAM.")

# Thirre 1× te start
load_services_once()

# ========== ENDPOINTS ==========
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/columns")
def list_columns():
    try:
        sample = supabase.from_("detailedtable").select("*").limit(1).execute().data
    except:
        return []
    if not sample:
        return []
    return list(sample[0].keys())

@app.get("/search")
def search_service(q: str):
    t0 = time.time()

    # 1) Clean input identik si lokale
    cleaned = re.sub(r"[^a-zA-Z0-9 ëç]+", "", q.lower()).strip()
    refined = cleaned  # identik si në PC

    # 2) Krijo embedding për query (identik)
    try:
        rsp = client.embeddings.create(model="text-embedding-3-large", input=refined)
        q_emb = np.array(rsp.data[0].embedding, dtype=np.float32)
    except:
        return {"results": [], "time_sec": round(time.time() - t0, 2)}

    # 3) Load services nga RAM, llogarit similarity identike
    rows = SERVICES
    scored = []
    for r in rows:
        emb = r["embedding"]
        sim_raw = cosine(q_emb, emb)
        sim01 = scale01(sim_raw)
        if sim01 < RED_TH:
            continue
        scored.append((sim01, sim_raw, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 4) Kthe max 4 rezultate identical
    final = []
    for sim01, sim_raw, r in scored[:4]:
        final.append({
            "id": r["id"],
            "name": r["name"],
            "category": r.get("category",""),
            "keywords": r.get("keywords", [])[:5],
            "uniqueid": r["uniqueid"],
            "score": round(sim01, 3)
        })

    # 5) GPT check vetëm për yellows (identik logjika jote)
    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    if greens and 1 <= len(final) < 3 and yellows:
        cand = yellows[0][2]
        if gpt_check(refined, cand["name"]):
            final.append({
                "id": cand["id"],
                "name": cand["name"],
                "category": cand.get("category",""),
                "keywords": cand.get("keywords", [])[:5],
                "uniqueid": cand["uniqueid"],
                "score": round(yellows[0][0],3)
            })

    t_total = time.time() - t0
    return {"results": final, "time_sec": round(t_total, 2)}
