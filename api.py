import os, time, re, json, requests
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from supabase import create_client, Client
from fastapi import FastAPI

app_api = FastAPI()
app = app_api  # RUJE IDENTIK mos e prek

# ==== Lidhja me Supabase ====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==== Lidhja me OpenAI ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==== Funksione Utils (identike si lokale) ====
def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def scale01(x):
    return max(0.0, min(1.0, (x + 1.0) / 2.0))

def normalize(t: str):
    return re.sub(r"[^a-zA-Z0-9 ëç]+", "", t.lower()).strip()

# ==== Ngarko Services 1× në cloud nga Supabase DB (embedding_large) ====
SERVICES = []
LOADED = False

def load_services_once():
    global SERVICES, LOADED
    if LOADED:
        return

    print("⬇️ Po lexoj services 1x nga Supabase DB…")

    try:
        # Marrim VETËM columns që na duhen, për speed më të mirë
        rows = supabase.from_("detailedtable") \
            .select("id, name, keywords, uniqueid, category, embedding_large") \
            .execute().data
    except Exception as e:
        print("❌ Gabim connection me Supabase:", e)
        SERVICES = []
        LOADED = True
        return

    SERVICES.clear()

    for r in rows:
        vec = r.get("embedding_large")
        if not isinstance(vec, list):
            continue

        # I fusim në RAM si numpy array – fiks si në PC local
        SERVICES.append({
            "id": r["id"],
            "name": r["name"],
            "uniqueid": r.get("uniqueid", ""),
            "category": r.get("category", ""),
            "keywords": r.get("keywords", [])[:5],
            "embedding": np.array(vec, dtype=np.float32)
        })

    LOADED = True
    print(f"✅ U ngarkuan {len(SERVICES)} services në RAM.")

# Thirre këtë 1 herë kur serveri niset
load_services_once()

# ==== Endpoint Health ====
@app.get("/health")
def health():
    return {"status": "ok"}

# ==== Endpoint Search (cosine search direct nga RAM) ====
@app_api.get("/search")
def search_service(q: str):
    t0 = time.time()
    query_clean = normalize(q)

    try:
        e = client.embeddings.create(model="text-embedding-3-large", input=query_clean)
        qvec = np.array(e.data[0].embedding, dtype=np.float32)
    except Exception as er:
        print("❌ Embedding error:", er)
        return {"results": [], "time_sec": round(time.time()-t0, 2)}

    scored = []
    for s in SERVICES:
        sim_raw = cosine(qvec, s["embedding"])
        sim01 = scale01(sim_raw)
        if sim01 < 0.5:  # si fallback i brendshëm, nuk prish speed-in tonë ongoing
            continue
        scored.append((sim01, sim_raw, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    final = []
    for sim01,_,s in scored[:4]:
        final.append({
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "uniqueid": s["uniqueid"],
            "score_large": round(sim01, 3)
        })

    return {"results": final, "time_sec": round(time.time()-t0, 2)}
