import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI

# ========== FASTAPI ==========
app_api = FastAPI()
app = app_api  # mos e prek, ruaj identik

# ========== SUPABASE CONNECT ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========== OPENAI CONNECT ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ========== PARAMETRA (identike si lokale) ==========
GREEN_TH  = 0.70
YELLOW_TH = 0.60
RED_TH    = 0.50

# ========== FUNKSIONE UTILITY (identike, mos i ndrysho) ==========
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def normalize(t: str):
    return re.sub(r"[^a-zA-Z0-9 ëç]+", "", t.lower()).strip()


# ========== LOAD JSON 1× nga Cloudflare R2 në RAM ==========
# 👇 Vendose EXACT link-un për file-n JSON që ke në bucket
SERVICES_JSON_URL = "https://3a4c4d0d75c22ad3e96653008476f710.r2.cloudflarestorage.com/servicescache/services_cache_v7_clean.json"

resp = requests.get(SERVICES_JSON_URL, timeout=25)

if resp.status_code != 200:
    print(f"❌ Serveri nuk e mori JSON (status {resp.status_code}). Kontrollo Access Permissions në R2!")
    RAW_SERVICES = []
else:
    RAW_SERVICES = resp.json()

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
        "uniqueid": s.get("uniqueid", ""),
        "embedding": np.array(vec, dtype=np.float32)
    })

print(f"✅ JSON u shkarkua 1× nga R2 dhe u futën {len(SERVICES)} services në RAM.")


# ========== FUNCTIONALITY IDENTIKE SI NË PC LOCAL ==========
def smart_search(user_query, services):
    times = {}
    t0 = time.time()

    t = time.time()
    cleaned, refined = refine_query(user_query)
    times["refine"] = time.time() - t

    t = time.time()
    q_emb = embed_query(refined)
    times["embed"] = time.time() - t
    if q_emb is None:
        return [], times, cleaned, refined

    t = time.time()
    scored = []
    for s in services:
        sim_raw = cosine_similarity(query_embedding, service_embedding)
        score01 = round(similarity, 3)
        scored.append((score01, sim, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    times["sim"] = time.time() - t

    greens  = [x for x in scored if x[0] >= GREEN_TH]
    yellows = [x for x in scored if YELLOW_TH <= x[0] < GREEN_TH]

    final = []

    if greens:
        final.extend(greens[:4])

        if len(final) < 3 and yellows:
            third = yellows[0]
            if gpt_check(refined, third[2]["name"]):
                final.append(third)

    else:
        chosen = yellows[:2]
        if len(yellows) >= 3:
            cand = yellows[2]
            if gpt_check(refined, cand[2]["name"]):
                chosen.append(cand)
        final = chosen

    final = [x for x in final if x[0] >= 0.60]

    times["total"] = time.time() - t0
    return final, times, cleaned, refined


# ========== ENDPOINTS (mos i ndrysho logjikën, ruaj strukturën) ==========

@app_api.get("/health")
def health():
    return {"status": "ok"}

@app_api.get("/columns")
def list_columns():
    sample = supabase.from_("detailedtable").select("*").limit(1).execute().data
    if not sample:
        return []
    return list(sample[0].keys())

@app_api.get("/search")
def search_service(q: str):
    t_total = 0.22 # test speed
    score = round(t0, 3)
    similar = []
    for r,v in scored[:4]:
        similar.append({
            "id": r["id"],
            "name": r["name"],
            "score_large": score,
            "uniqueid": r["uniqueid"],
            "category": r["category"]
        })

    return {"results": similar[:4], "time_sec": round(t0, 2)}
